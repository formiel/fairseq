import argparse
import glob
import os
import string
from pathlib import Path
import pandas as pd
import csv
import unicodedata, six, re
from argparse import Namespace

import soundfile as sf
from fairseq.data import Dictionary, encoders

"""
Read provided ${SPLIT}.tsv files provided by multilingual TEDx
and return manifest and label files ready for training data2vec models 
where SPLIT=["train", "dev", "test"]
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing wav files to index"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="subset to process"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--bpe-model", type=str, help="path to dictionary"
    )
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    audio_root = Path(args.root) / "data" / args.split / "wav_splits" # ~/data/train/wav_splits
    text_root = Path(args.root) / "data" / args.split / "txt"
    segment_file = text_root / "segments"
    label_file = text_root / f"{args.split}.en"

    bpe_tokenizer = {"bpe": "sentencepiece", "sentencepiece_model": args.bpe_model}
    bpe = encoders.build_bpe(Namespace(**bpe_tokenizer))
    suffix = Path( args.bpe_model).stem.split("_")[-1]

    manifest_path = Path(args.dest) / f"{args.split}.tsv"
    removed_indices = []

    audio_fnames = []
    labels = []
    with open(segment_file, "r") as seg:
        for line in seg:
            audio_fnames.append(line.strip().split(" ")[0])
    with open(label_file, "r") as fhanfle:
        for line in fhanfle:
            labels.append(line.strip())

    outputs = [] # list of (audio_fname, frames, label)
    print(f'Checking valid samples: less than 35s and have non-zero labels')
    for au, label in zip(audio_fnames, labels):
        # print(f"au: {au}: {label}")
        filepath = audio_root / f"{au}.wav"
        audiofile = sf.SoundFile(filepath)
        frames = audiofile.frames
        durations = frames / audiofile.samplerate
        if args.split == "train":
            if durations <= 35 and frames >= 3000 and len(label) > 0:
                outputs.append((
                    os.path.relpath(filepath.as_posix(), audio_root),
                    frames,
                    bpe.encode(label)
                ))
            else:
                print(f"- removing audio:{au}, #frames:{frames}, label:{label}")
        else:
            if len(label) > 0:
                outputs.append((
                    os.path.relpath(filepath.as_posix(), audio_root),
                    frames,
                    bpe.encode(label)
                ))
            else:
                print(f"- removing audio:{au}, #frames:{frames}, label:{label}")

    # print(f"Writing manifest file...")
    # with open(manifest_path, "w") as dest:
    #     print(audio_root.as_posix(), file=dest)
    #     for info in outputs:
    #         print(f"{info[0]}\t{info[1]}", file=dest)

    print(f"Preparing label file...")
    with open(
        os.path.join(args.dest, f"{args.split}.{suffix}"), "w"
    ) as label_out:
        for info in outputs:
            print(info[-1], file=label_out)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
