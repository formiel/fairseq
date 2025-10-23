import os
import argparse
from collections import namedtuple
from tqdm import tqdm

import torch
from fairseq import utils, tasks
from fairseq.data import iterators
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate XNLI with a fine-tuned Roberta model")
    parser.add_argument(
        "--user-dir", type=str, default=None,
    )
    parser.add_argument(
        "--data-path", type=str, required=True, 
        help="Path to the XNLI data directory"
    )
    parser.add_argument(
        "--subset", type=str, default="xnli", 
        help="Subset of XNLI to evaluate on (default: xnli)"
    )
    parser.add_argument(
        "--pretrained-model-path", type=str, required=True, 
        help="Path to the fine-tuned Roberta model checkpoint"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, 
        help="Path to the fine-tuned Roberta model checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    # import user dir
    if args.user_dir is not None:
        user_dir = (Path(os.environ["FAIRSEQ"]) / Path(args.user_dir))
        Arg = namedtuple("Arg", ["user_dir"])
        arg = Arg(user_dir.__str__())
        print(f"Importing user module from {user_dir}")
        utils.import_user_module(arg)

    from fairseq.tasks.sentence_prediction import (
        SentencePredictionConfig, SentencePredictionTask
    )
    task_cfg = SentencePredictionConfig(
        _name="sentence_prediction",
        data=args.data_path,
        init_token=0,
        separator_token=2,
        num_classes=3,
        max_positions=512,
        d2v2_multi=True,
        seed=1,
        regression_target=False,
    )
    print("Setting up task...")
    task = SentencePredictionTask.setup_task(task_cfg)

    # Load the fine-tuned data2vec model
    from examples.data2vec.models.data2vec_text_classification import (
        Data2VecTextClassificationConfig, Data2VecTextClassificationModel
    )

    print("Setting up fine-tuned model...")
    model_cfg = Data2VecTextClassificationConfig(
        model_path=args.pretrained_model_path,
        pooler_dropout=0.1,
    )
    finetuned_model = Data2VecTextClassificationModel.build_model(model_cfg)
    finetuned_model.register_classification_head("sentence_classification_head", num_classes=3)
    print(f"Loading fine-tuned weights from best validation checkpoint {args.model_path}...")
    finetuned_state = torch.load(args.model_path, map_location="cpu")
    finetuned_model.load_state_dict(finetuned_state["model"], strict=True)
    print("Model loaded successfully!")
    finetuned_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    finetuned_model.eval()

    print("Building criterion...")
    from fairseq.criterions.sentence_prediction import SentencePredictionConfig
    criterion_cfg = SentencePredictionConfig()
    criterion = task.build_criterion(criterion_cfg)

    print("Loading test dataset...")
    test_data = task.load_dataset(args.subset, epoch=1)
    print(f"Number of test samples: {len(test_data)}")
    total_correct = 0
    total_samples = 0
    bsz = int(args.batch_size)

    batch_iterator = task.get_batch_iterator(
            dataset=test_data,
            max_sentences=bsz,
            num_workers=1,
        ).next_epoch_itr(shuffle=False)
    progress = tqdm(enumerate(batch_iterator), total=len(batch_iterator))

    # Evaluate in batches
    for batch_idx, sample in progress:
        sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
        # print(f'sample keys: {sample.keys()}')
        labels = utils.move_to_cuda(sample['target']) if torch.cuda.is_available() else sample['target']
        labels = labels.squeeze(-1)
        # print(f'labels: {labels.size()}')

        with torch.no_grad():
            logits, _ = finetuned_model(
                **sample["net_input"], features_only=True, classification_head_name="sentence_classification_head"
            )
            predictions = logits.argmax(dim=-1)
            # print(f'predictions: {predictions.size()}')

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"XNLI {args.subset.upper()} Accuracy: {accuracy * 100:.2f}%")

    # write the accuracy to a file
    run = args.model_path.split('/')[-2]
    results_file = Path(args.model_path).parent / f"xnli_{args.subset}_run{run}.txt"
    with open(results_file, "w") as f:
        f.write(f"XNLI {args.subset.upper()} Accuracy - run {run}: {accuracy * 100:.2f}%\n")
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    main()