import argparse
import logging
from pathlib import Path
import torch
from transformers import (
    Data2Vec2AudioConfig, Data2Vec2AudioModel
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", type=str)
    args = parser.parse_args()

    pretrained_path = Path(args.pretrained_path)
    fairseq_ckpt = torch.load(pretrained_path) 
    fairseq_config = fairseq_ckpt["cfg"]
    print(fairseq_config["model"])

    print(f"Initializing configuration and model...")
    configuration = Data2Vec2AudioConfig(
        **fairseq_config["model"]
    )
    print(f"Saving pretrained configuration...")
    model_name = pretrained_path.parent.name
    configuration.save_pretrained(f"HuggingFace/{model_name}")
    hf_model = Data2Vec2AudioModel(configuration)
    print(f'MODEL:\n{hf_model}')

    fairseq_dict = fairseq_ckpt["model"]
    feature_extractor = hf_model.feature_extractor

    for name, value in fairseq_dict.items():
        if name.startswith("modality_encoders"):
            feature_extractor

if __name__ == "__main__":
    main()