import argparse
import logging
from pathlib import Path
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import (
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)
from fairseq import tasks
from fairseq import utils
from fairseq import checkpoint_utils
# from examples.data2vec.models.data2vec2 import Data2VecMultiModel, Data2VecMultiConfig
from datasets import load_dataset
from transformers import Wav2Vec2Processor
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME


def compare_tensors(tensor_a, tensor_b):
    max_absolute_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
    # print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    if max_absolute_diff > 0.0:
        raise ValueError(f"max_absolute_diff={max_absolute_diff}")
    success = torch.allclose(tensor_a, tensor_b, atol=1e-3)
    if not success:
        raise ValueError("!!!Something went wrong!!!")


@torch.no_grad()
def test_converted_weights(args):
    checkpoint_path = args.checkpoint_path
    pytorch_dump_folder_path = args.pytorch_dump_folder_path

    hf_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    print(f"Pre-trained weights loaded to HF model!")
    hf_model.eval()

    # fairseq checkpoint
    print(f"Loading fairseq model...")
    utils.import_user_module(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, {})
    w2v_args = state.get("cfg", None)
    assert w2v_args is not None
    w2v_args.criterion = None
    w2v_args.lr_scheduler = None
    task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
    print(f"fairseq model args: {w2v_args.model}")
    fairseq_model = task.build_model(w2v_args.model, from_checkpoint=True)
    fairseq_model.load_state_dict(state["model"], strict=True)
    fairseq_model.remove_pretraining_modules(modality="AUDIO")
    fairseq_model.eval()
    print(f"Pre-trained weights loaded to fairseq model!")

    # compare keys and parameters
    hf_keys = [n for n, _ in hf_model.named_parameters()]
    fairseq_keys = [n for n, _ in fairseq_model.named_parameters()]
    diffs = list(set(fairseq_keys) - set(hf_keys))
    if len(diffs) > 0:
        print(f"diffs: {diffs}")

    for n, p in hf_model.named_parameters():
        compare_tensors(p, fairseq_model.state_dict()[n])

    # print("Loading processor...")
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
    # print("Loading dataset...")
    # mls = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", cache_dir="downloaded_data")
    # input_audio = [x["array"] for x in mls[:4]["audio"]]
    # inputs = processor(input_audio, return_tensors="pt", padding=True)
    # input_values = inputs.input_values
    # attention_mask = inputs.attention_mask

    print(f"Comparing outputs with randomized tensors...")
    input_values = torch.randn((1, 320000), dtype=torch.float32)

    print(f"Forward using HF model...")
    hf_output = hf_model(input_values, padding_mask=None, mode="AUDIO", mask=False)

    print(f"Forward using fairseq model...")
    fairseq_output = fairseq_model(source=input_values, padding_mask=None, mask=False, features_only=True)

    print(f"Comparing x...")
    compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
    print(f'MATCHED!')


def main_old(args):
    pretrained_path = Path(args.pretrained_path)
    fairseq_ckpt = torch.load(pretrained_path) 
    fairseq_model_config = fairseq_ckpt["cfg"]["model"]
    print(f"*** fairseq's configuration***\n{fairseq_model_config}")
    for k, v in fairseq_model_config["modalities"].items():
        print(f"- {k}: {v}")
        print("-"*5)
    
    print(f"Initializing configuration and model...")
    model_config = {k: v for k, v in fairseq_model_config.items() if not "ema" in k and not "decoder" in k and not "loss" in k}
    configuration = Data2Vec2MultiConfig(**model_config)
    print(configuration)

    print(f"Saving pretrained configuration...")
    model_name = pretrained_path.parent.name
    configuration.save_pretrained(f"HuggingFace/{model_name}")

    print("Initializing model using pre-trained configuration")
    hf_model = Data2Vec2MultiModel(configuration)
    print(f'MODEL:\n{hf_model}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--user-dir", default="examples/data2vec")
    args = parser.parse_args()

    test_converted_weights(args)

if __name__ == "__main__":
    main()