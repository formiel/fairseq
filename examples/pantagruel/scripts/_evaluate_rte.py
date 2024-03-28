from pathlib import Path
# import torch
from examples.data2vec.models.data2vec_text_classification import Data2VecTextClassificationModel,Data2VecTextClassificationConfig
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import torch
from fairseq import tasks
from fairseq import utils
from fairseq.tasks.sentence_prediction import SentencePredictionTask,SentencePredictionConfig
# from fairseq.dataclass.utils import omegaconf_no_object_check
# from fairseq.dataclass.configs import FairseqConfig
# from omegaconf import OmegaConf, open_dict
# from fairseq.dataclass.initialize import add_defaults
# from fairseq import models
from fairseq import checkpoint_utils
# from examples.data2vec.models.data2vec_text_classification import Data2VecTextClassificationModel
from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig, GPT2BPE
# from examples.data2vec.models.data2vec2 import Data2VecMultiConfig
from fairseq.data import IdDataset, RightPadDataset, RightPaddingMaskDataset, NumelDataset, NumSamplesDataset

utils.import_user_module("examples/data2vec")

ft_path = "/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/finetuning/rte_pretrained_from_fairseq/checkpoint_best.pt"
pretrained_model = "/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/text/nlp_base_changed_path.pt"
state = checkpoint_utils.load_checkpoint_to_cpu(ft_path, {})
args = state.get("cfg", None)
task = tasks.setup_task(args.task)
cfg = Data2VecTextClassificationConfig()
for attr in cfg._get_all_attributes():
    cfg.attr = args.model[attr]
cfg.model_path = pretrained_model

model = Data2VecTextClassificationModel(cfg)
model.register_classification_head(
    name="sentence_classification_head", num_classes=2
)
model.load_state_dict(state["model"], strict=True)

gpt2_cfg = GPT2BPEConfig(gpt2_encoder_json="/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/text/encoder.json", gpt2_vocab_bpe="/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/text/vocab.bpe")
bpe = GPT2BPE(gpt2_cfg)

# roberta = Data2VecTextClassificationModel.from_pretrained(
#     '/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/finetuning/rte_pretrained_from_fairseq/',
#     checkpoint_file='checkpoint_best.pt',
#     data_name_or_path='/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE-fairseq-pretrained/RTE-bin',
# )
label_fn = lambda label: task.label_dictionary.string(
    [label + task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
model.cuda()
model.eval()
DATA_DIR= "/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE-fairseq-pretrained"

with open(Path(DATA_DIR)/"dev.tsv") as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        # print(f'sent1: {sent1}, sent2: {sent2}, label: {target}')
        bpe_sentence = "<s> " + bpe.encode(sent1) + " </s>"
        bpe_sentence += " </s>"
        bpe_sentence += " " + bpe.encode(sent2) + " </s>"
        src_tokens = task.dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        ).long().cuda()
        net_input = {
                "source": src_tokens,
                "id": IdDataset(),
                "padding_mask": RightPaddingMaskDataset(src_tokens),
            }
        print(f"net_input: {net_input}")
        prediction, _ = model(
            **net_input,
            features_only=True,
            classification_head_name="sentence_classification_head",
        )
        print(f"prediction: {prediction}")

        # prediction = model.predict('sentence_classification_head', tokens).argmax().item()
#         prediction_label = label_fn(prediction)
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print(f'| Accuracy: {float(ncorrect)/float(nsamples)}')


# task = SentencePredictionTask.setup_task(cfg)

# data2vec_text = Data2VecTextClassificationModel(model_cfg)

# ckpt = torch.load("/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/finetuning/rte_pretrained_from_fairseq/checkpoint_best.pt")
# cfg = convert_namespace_to_omegaconf(ckpt["cfg"])
# model = task.build_model(cfg["model"])


# DATA_DIR= "/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE-fairseq-pretrained"
# label_map = {0: 'entailment', 1: 'not entailment'}
# ncorrect, nsamples = 0, 0


# data2vec_text.load_state_dict(state_dict)
# data2vec_text.cuda()
# data2vec_text.eval()
# with open(Path(DATA_DIR)/"dev.tsv", ) as fin:
#     fin.readline()
#     for index, line in enumerate(fin):
#         tokens = line.strip().split('\t')
#         sent1, sent2, target = tokens[1], tokens[2], tokens[3]
#         tokens = data2vec_text.encode(sent1, sent2)
#         prediction = data2vec_text.predict('mnli', tokens).argmax().item()
#         prediction_label = label_map[prediction]
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print('| Accuracy: ', float(ncorrect)/float(nsamples))
# # Expected output: 0.9060

