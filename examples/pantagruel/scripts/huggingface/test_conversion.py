import argparse
import logging
from pathlib import Path
import torch
from transformers import (
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", type=str)
    args = parser.parse_args()

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

    # print(f"Saving pretrained configuration...")
    # model_name = pretrained_path.parent.name
    # configuration.save_pretrained(f"HuggingFace/{model_name}")

    print("Initializing model using pre-trained configuration")
    hf_model = Data2Vec2MultiModel(configuration)
    print(f'MODEL:\n{hf_model}')


if __name__ == "__main__":
    main()