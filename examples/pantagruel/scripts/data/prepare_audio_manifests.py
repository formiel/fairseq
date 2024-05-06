"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 26 April 2024

Input: 

Prepare manifest ${SPLIT_}${DATASET}.tsv 
(where SPLIT is train/valid/test) files for each dataset 
(same as output .tsv file of examples/wav2vec/wav2vec_manifest.py)
where 
- first line of the manifest is the root to audio files
- each remaining line: relative_path_to_audio \t number_of_frames

Each dataset is organized into:
- $OUTPUT_DIR/$DATASET/$SPLIT, the corresponding audio files are under these folders 
"""

import argparse
import time
import logging
import glob
from pathlib import Path
import itertools
import random
import math
import shutil
from multiprocessing import Pool

import soundfile as sf

from utils import (
    create_zip, create_manifest_file, get_paths_from_json, get_paths_from_dir,
    process_audio_file, create_zip_file, get_zip_info,
    TRAIN_FNAME, VALID_FNAME, TEST_FNAME
)


log = logging.getLogger(__name__)

# dictionary of datasets and relative path to json files under audio_root / dataset
DATA_SETS = {
    "mls_french_jz": None,
    "audiocite_with_metadata": "data.json", 
    "studios-tamani-kalangou-french": "v1_10102021/data.json",
    "African_Accented_French": "data.json",
    "Att-HACK_SLR88": "data.json",
    "CaFE": "data.json",
    "CFPP_corrected": "json/data_full.json",
    "ESLO": "data.json",
    "EPAC_flowbert": "data.json",
    "GEMEP": "data.json",
    "MPF": "data.json",
    "Portmedia": "data.json",
    "TCOF_corrected": "output/data_full.json",
    "MaSS": "data.json",
    "NCCFr": "data.json",
    "voxpopuli_unlabeled": "json/data_full.json",
    "voxpopuli_transcribed": None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, choices=list(DATA_SETS.keys())
    )
    parser.add_argument(
        "--valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--audio-root", required=True, type=str, metavar="DIR", 
        help="root to the audio directory",
    )
    parser.add_argument(
        "--output-root", required=True, type=str, metavar="DIR", 
        help="output root where zipped audio \
            and tsv files for model training are saved",
    )
    parser.add_argument(
        "--extensions", default="flac,wav", type=str, metavar="EXT", 
        help="list of audio extensions to look for",
    )
    parser.add_argument(
        "--max-seconds-per-file", default=30, type=int, metavar="N", 
        help="maximum seconds to split each audio file",
    )
    parser.add_argument(
        "--max-files-per-zip", default=500000, type=int, metavar="N", 
        help="number of files to be included in a zip file",
    )
    parser.add_argument(
        "--workers", default=8, type=int, metavar="N", 
        help="number of workers",
    )
    parser.add_argument(
        "--seed", default=1, type=int, metavar="N", help="random seed",
    )
    args = parser.parse_args()

    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    # create directory to save output tsv and zipped audio files
    output_root = Path(args.output_root)
    dataset_dir = output_root / "zipped_audio" / args.dataset # by datasets and splits
    train_dir, valid_dir, test_dir = (
        Path(dataset_dir / TRAIN_FNAME), 
        Path(dataset_dir / VALID_FNAME), 
        Path(dataset_dir / TEST_FNAME)
    )
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    rand = random.Random(args.seed) if args.valid_percent > 0 else None

    # read from json files if available, otherwise 
    # get all audio paths with valid audio extensions
    audio_dir = Path(args.audio_root) / args.dataset
    audio_paths_from_json = None
    if DATA_SETS[args.dataset] is not None:
        audio_paths_from_json = get_paths_from_json(
            audio_dir / DATA_SETS[args.dataset]
        )
    audio_paths_from_dir = get_paths_from_dir(
        audio_dir=audio_dir,
        audio_exts=args.extensions,
    )
    
    # convert waveform data so that each dataset has the same structure: audio files under output_dir/args.dataset/split
    logging.info(f"Restructure audio files under each TRAIN/VALID/TEST splits")
    max_chunk_duration = args.max_seconds_per_file # in seconds
    with Pool(args.workers) as pool:
        arguments = [
            (Path(audio_path), dataset_dir, rand, args.valid_percent, max_chunk_duration)
            for audio_path in audio_paths_from_dir
        ]
        pool.starmap(process_audio_file, arguments)

    logging.info(f'Total number of original audio files: {len(audio_paths_from_dir)}')
    if audio_paths_from_json is not None:
        logging.info(f'Number of audio files from json: {len(audio_paths_from_json)}')

    SPLITS = [TRAIN_FNAME]
    DATA_DIR = [train_dir]
    n_train = len(get_paths_from_dir(train_dir))
    n_val = len(get_paths_from_dir(valid_dir))
    n_test = len(get_paths_from_dir(test_dir))
    logging.info(f"Total number of audio files in TRAIN: {n_train}")
    if n_val > 0:
        SPLITS.append("valid")
        DATA_DIR.append(valid_dir)
        logging.info(f"Total number of audio files in VALID: {n_val}")
    if n_test > 0:
        SPLITS.append("test")
        DATA_DIR.append(test_dir)
        logging.info(f"Total number of audio files in TEST: {n_test}")
    logging.info(f"Total: {n_train} + {n_val} + {n_test}  = {n_train+n_val+n_test}")

    # create zip file from each split under dataset_dir
    for d in DATA_DIR:
        logging.info(f"Creating zip for audio files in folder {d.as_posix()}")
        # create_zip(
        #     data_root=d,
        #     zip_prefix=dataset_dir / f"waveforms_{d.name}",
        #     max_num_files=int(args.max_files_per_zip)
        # )
        zip_prefix = dataset_dir / f"waveforms_{d.name}"
        pool = Pool(args.workers)  # Create a pool of workers
        tasks = []
        paths, num_zip_files, max_num_files = get_zip_info(
            data_root=d,
            max_num_files=args.max_files_per_zip
        )
        for i in range(num_zip_files):
            zip_file_i = f"{zip_prefix.as_posix()}_{i}.zip"
            start_index = i * max_num_files
            end_index = start_index + max_num_files
            paths_zip_i = paths[start_index:end_index]

            # Add task to the list for multiprocessing
            tasks.append((zip_file_i, paths_zip_i))

        # Use pool.starmap to apply the function to each task
        pool.starmap(create_zip_file, tasks)

        # Close the pool to free resources
        pool.close()
        pool.join()

    # create manifest file for each split under model_training
    model_training_dir = output_root / "model_training"
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