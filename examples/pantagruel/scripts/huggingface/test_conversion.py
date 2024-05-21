import argparse
import logging
from pathlib import Path
from transformers import Data2Vec2AudioConfig, Data2Vec2AudioModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", type=str)
    args = parser.parse_args()

    # pretrained_path = Path(args.pretrained_path)

    logging.info(f"Initializing configuration and model...")
    configuration = Data2Vec2AudioConfig()
    model = Data2Vec2AudioModel(configuration)
    

if __name__ == "__main__":
    main()