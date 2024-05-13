"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 26 April 2024

Prepare manifest ${SPLIT_}${DATASET}.tsv 
(where SPLIT is train/valid/test) files for each dataset 
(same as output .tsv file of examples/wav2vec/wav2vec_manifest.py)
where 
- first line of the manifest is the root to audio files
- each remaining line: relative_path_to_audio \t number_of_frames
(the relative path is red with offset and length information from zipped dataset)

Each dataset is organized into:
- $OUTPUT_DIR/$DATASET/$SPLIT, the corresponding audio files are under these folders 
"""

import argparse
import logging
import glob
from pathlib import Path
import itertools
import random
import math
import shutil

import soundfile as sf

from utils import create_zip, create_manifest_file


log = logging.getLogger(__name__)


DATA_SETS = [
    "mls_french_jz",
    "audiocite_with_metadata", 
    "studios-tamani-kalangou-french",
    "African_Accented_French",
    "Att-HACK_SLR88",
    "CaFE",
    "CFPP_corrected",
    "ESLO",
    "EPAC_flowbert",
    "GEMEP",
    "MPF",
    "Portmedia",
    "TCOF_corrected",
    "MaSS",
    "NCCFr",
    "voxpopuli_unlabeled",
    "voxpopuli_transcribed",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=DATA_SETS)
    parser.add_argument(
        "--valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--audio-dir", required=True, type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--extensions", default="flac,wav", type=str, metavar="EXT", help="list of extensions to look for"
    )
    parser.add_argument("--max-files", default=500000, type=int, metavar="N", 
            help="number of files to be included in a zip file")
    parser.add_argument("--seed", default=1, type=int, metavar="N", help="random seed")
    args = parser.parse_args()

    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    # create directory to save output tsv and zipped audio files
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "zipped_audio" / args.dataset # by datasets and splits
    train_dir, valid_dir, test_dir = (
        dataset_dir / "train", dataset_dir / "valid", dataset_dir / "test"
    )
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    rand = random.Random(args.seed) if args.valid_percent > 0 else None

    # get all audio paths with valid audio extensions
    dir_path = Path(args.audio_dir) / args.dataset
    audio_paths = itertools.chain()
    for ext in args.extensions.split(","):
        audio_paths = itertools.chain(
            audio_paths, 
            glob.glob(f'{dir_path}/**/*.{ext}', recursive=True)
        )

    # convert waveform data so that each dataset has the same structure: audio files under output_dir/args.dataset/split, each audio file are chunks of 50s maximum
    logging.info(f"Restructure audio files under each TRAIN/VALID/TEST splits")
    max_chunk_duration = 50 # in seconds
    n_train, n_val, n_test = 0, 0, 0
    for i, path in enumerate(audio_paths):
        path = Path(path)
        dest_dir = train_dir
        if "valid" in path.as_posix() or "dev" in path.as_posix():
            dest_dir = valid_dir
            n_val += 1
        elif "test" in path.as_posix():
            dest_dir = test_dir
            n_test += 1
        else:
            if rand is not None and rand.random() <= args.valid_percent:
                dest_dir = valid_dir
                n_val += 1
            else:
                n_train += 1

        sample_rate = sf.info(path.as_posix()).samplerate
        max_chunk_frames = max_chunk_duration * sample_rate
        duration = sf.info(path.as_posix()).frames / sample_rate
        num_chunks = math.ceil(duration / max_chunk_duration)
        assert num_chunks >= 1
        if num_chunks > 1:
            logging.warning(f"splitting {path} into {num_chunks} chunks...")
            for start in range(num_chunks):
                waveform, _ = sf.read(
                path, frames=max_chunk_frames, start=start*max_chunk_frames
            ) # T
                idx = "_{:02d}".format(start)
                logging.info(f"writing waveforms idx {idx}")
                sf.write(
                    (dest_dir / f"{path.stem}{idx}.flac").as_posix(),
                    waveform, 16_000
                )
        else:
            sym_path = dest_dir / path.name
            if not sym_path.is_symlink():
                sym_path.symlink_to(path)

    logging.info(f"Total number of audio files: {i+1}")
    logging.info(f"n_train={n_train}, n_val={n_val}, n_test={n_test}")
    assert n_train + n_val + n_test == i+1
    SPLITS = ["train"]
    DATA_DIR = [train_dir]
    if n_val > 0:
        SPLITS.append("valid")
        DATA_DIR.append(valid_dir)
    if n_test > 0:
        SPLITS.append("test")
        DATA_DIR.append(test_dir)

    # create zip file from each split under dataset_dir
    for d in DATA_DIR:
        logging.info(f"Creating zip for audio files in folder {d.as_posix()}")
        create_zip(
            data=d,
            zip_prefix=dataset_dir / f"waveforms_{d.name}",
            max_num_files=int(args.max_files)
        )

    # create manifest file for each split under model_training
    model_training_dir = output_dir / "model_training"
    model_training_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        logging.info(f"Writing manifest file for split:{split.upper()}")
        create_manifest_file(
            dataset_dir,
            model_training_dir,
            split,
        )

    # clean files
    for d in [train_dir, valid_dir, test_dir]:
        shutil.rmtree(d)

if __name__ == "__main__":
    main()