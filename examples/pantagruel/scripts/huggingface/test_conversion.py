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
from transformers.utils import CONFIG_NAME


@torch.no_grad()
def test_converted_weights(checkpoint_path, pytorch_dump_folder_path):
    print(f"Initializing HF model with pre-trained config...")
    configuration = Data2Vec2MultiConfig.from_pretrained(f"{pytorch_dump_folder_path}/{CONFIG_NAME}")
    hf_model = Data2Vec2MultiModel(configuration)
    print(f"Loading from pre-trained weights...")
    hf_model.from_pretrained(pytorch_dump_folder_path)

    # print("Loading processor...")
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
    # print("Loading dataset...")
    # mls = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", cache_dir="downloaded_data")
    # input_audio = [x["array"] for x in mls[:4]["audio"]]
    # inputs = processor(input_audio, return_tensors="pt", padding=True)
    # input_values = inputs.input_values
    # attention_mask = inputs.attention_mask

    input_values = torch.randn((1, 320000), dtype=torch.float16)

    # print(f"Forward using HF model...")
    # hf_model.eval()
    # hf_output = hf_model(input_values, padding_mask=None, mode="AUDIO", mask=False)["x"]
    # print(f"hf_output: {hf_output.shape}")

    # fairseq checkpoint
    utils.import_user_module("examples/data2vec")
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, {})
    w2v_args = state.get("cfg", None)
    assert w2v_args is not None
    w2v_args.criterion = None
    w2v_args.lr_scheduler = None
    task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
    print(f"task: {task}")
    print(f"model: {w2v_args.model}")
    model = task.build_model(w2v_args.model, from_checkpoint=True)
    print(model)
    model.load_state_dict(state["model"], strict=True)
    model.remove_pretraining_modules(modality="AUDIO")

    print(f"Forward using fairseq model...")
    fairseq_model = model.eval()
    fairseq_output = fairseq_model(source=input_values, padding_mask=None, mask=False, features_only=True)["x"]
    print(f"fairseq_output: {fairseq_output.shape}")

    # max_absolute_diff = torch.max(torch.abs(hf_output - fairseq_output)).item()
    # print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    # success = torch.allclose(hf_output, fairseq_output, atol=1e-3)
    # if success:
    #     print("Both models output the same tensors.")
    # else:
    #     raise Exception("Something went wrong.")


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
    args = parser.parse_args()

    test_converted_weights(
        args.checkpoint_path, args.pytorch_dump_folder_path,
    )

if __name__ == "__main__":
    main()