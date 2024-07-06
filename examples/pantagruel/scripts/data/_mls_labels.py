#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}
    transcriptions_data = {}

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        root = next(tsv).strip()
        transcripts_file = Path(root).parent / "transcripts.txt"
        # read transcriptions into a dictionary first
        with open(transcripts_file, "r") as trans_f:
            for tline in trans_f:
                items = tline.strip().split()
                item_name = items[0]
                if item_name not in transcriptions_data:
                    transcriptions_data[item_name] = " ".join(items[1:])
                else:
                    raise ValueError("Should not have two files of the same name!")
        for line in tsv:
            line = line.strip()
            fname = os.path.basename(line).split(".")[0]
            if fname not in transcriptions:
                assert fname in transcriptions_data
                transcriptions[fname] = transcriptions_data[fname]

            print(transcriptions[fname], file=wrd_out)
            print(
                " ".join(list(transcriptions[fname].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
