import argparse
from argparse import Namespace
import glob
import os
import string
from pathlib import Path
import pandas as pd
import csv
import unicodedata, six, re
from argparse import Namespace
from examples.speech_to_text.data_utils import load_df_from_tsv

import soundfile
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

    root = Path(args.root)
    split = args.split
    audio_root = root / "clips_wav"
    cv_tsv_path = root / "validated.tsv"
    cv_tsv = load_df_from_tsv(cv_tsv_path)
    covost_tsv = load_df_from_tsv(root / "covost_v2.fr_en.tsv")
    df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
    if split == "train":
        df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
    else:
        df = df[df["split"] == split]
    data = df.to_dict(orient="index").items()
    data = [v for k, v in sorted(data, key=lambda x: x[0])]
    file_paths = []
    translations = []
    for e in data:
        path = audio_root / e["path"].replace(".mp3", ".wav")
        if path.exists():
            file_paths.append(path)
            translations.append(e["translation"])

    # print(f'Preparing manifest files containing root directory on top and relative paths in each line...')
    # with open(Path(args.dest) / f"{args.split}.tsv", "w") as dest:
    #     print(audio_root.as_posix(), file=dest)

    #     for au in file_paths:
    #         fp = audio_root / au
    #         frames = soundfile.info(fp).frames
    #         print(
    #             "{}\t{}".format(os.path.relpath(fp.as_posix(), audio_root), frames), file=dest
    #         )

    # print(f"Preparing label files...")
    # with open(Path(args.dest) / f"{args.split}.ltr", "w"
    # ) as ltr_out, open(
    #     Path(args.dest) / f"{args.split}.wrd", "w"
    # ) as wrd_out:
    #     for line in translations:
    #         # Remove spaces at the beginning and the end of the sentence
    #         line = line.lstrip().rstrip()
    #         # Remove trailing quotation marks
    #         line = str(line.strip('\"'))
    #         print(line, file=wrd_out)
    #         print(
    #             " ".join(list(line.replace(" ", "|"))) + " |",
    #             file=ltr_out
    #         )

    label_file = Path(args.dest) / f"{args.split}.wrd"
    print(f"Preparing label files...")
    bpe_tokenizer = {"bpe": "sentencepiece", "sentencepiece_model": args.bpe_model}
    bpe = encoders.build_bpe(Namespace(**bpe_tokenizer))
    with open(label_file, "r"
    ) as labels, open(
        os.path.join(args.dest, f"{args.split}.spm-char"), "w"
    ) as label_out:
        for line in labels:
            line = bpe.encode(line)
            print(line, file=label_out)
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
