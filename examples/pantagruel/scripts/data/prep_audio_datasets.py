#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil

import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH, COMMONVOICE

from tqdm import tqdm

from utils import (
    create_zip,
    get_zip_manifest,
)


log = logging.getLogger(__name__)

LIBRISPEECH_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]
COMMONVOICE_SPLITS = [
    "train",
    "dev",
    "test",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "speaker"]


def process(args):
    dataset_name = args.dataset
    print(f"DATASET: {dataset_name}")

    assert Path(args.input_root).exists(), f"{args.input_root} does not exist."
    input_root = Path(args.input_root).absolute()
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    feature_fname = "zipped_audio"
    feature_root = out_root / feature_fname
    feature_root.mkdir(exist_ok=True)

    if dataset_name == "librispeech":
        SPLITS = LIBRISPEECH_SPLITS
    elif dataset_name == "commonvoice":
        SPLITS = COMMONVOICE_SPLITS
    else:
        raise NotImplementedError

    for split in SPLITS:
        print(f"Fetching split {split}...")
        if dataset_name == "librispeech":
            dataset = LIBRISPEECH(input_root.as_posix(), url=split) # channels_first
            for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
                sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                tgt_sample_rate = 16_000
                sf.write(
                    (feature_root / f"{sample_id}.flac").as_posix(),
                    wav.squeeze(0).numpy(), tgt_sample_rate
                )
        elif dataset_name == "commonvoice":
            dataset = COMMONVOICE(input_root.as_posix(), tsv=f"{split}.tsv")
            for wav, sample_rate, dictionary in tqdm(dataset):
                sample_id = dictionary["path"].replace(".mp3", "")
                tgt_sample_rate = 16_000
                sf.write(
                    (feature_root / f"{sample_id}.flac").as_posix(),
                    wav.squeeze(0).numpy(), tgt_sample_rate
                )

    # Pack features into ZIP
    zip_path = out_root / f"{feature_fname}.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path, is_audio=True)
    # Generate TSV manifest
    print("Generating manifest...")

    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if dataset_name == "librispeech":
            dataset = LIBRISPEECH(input_root.as_posix(), url=split)
            for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
                sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
                manifest["id"].append(sample_id)
                manifest["audio"].append(audio_paths[sample_id])
                manifest["n_frames"].append(audio_lengths[sample_id])
                manifest["src_text"].append(utt.lower())
                manifest["speaker"].append(spk_id)
        elif dataset_name == "commonvoice":
            dataset = COMMONVOICE(input_root.as_posix(), tsv=f"{split}.tsv")
            for wav, sample_rate, dictionary in tqdm(dataset):
                sample_id = dictionary["path"].replace(".mp3", "")
                manifest["id"].append(sample_id)
                manifest["audio"].append(audio_paths[sample_id])
                manifest["n_frames"].append(audio_lengths[sample_id])
                manifest["src_text"].append(dictionary["sentence"].lower())
                manifest["speaker"].append(dictionary["client_id"])
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}-with-transcript.tsv"
        )

    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, choices=["librispeech", "commonvoice"])
    parser.add_argument("--input-root", "-i", type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
