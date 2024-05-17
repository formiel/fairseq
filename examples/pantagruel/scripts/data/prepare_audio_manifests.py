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
import random

from utils import (
    create_zip, create_manifest_file, 
    get_paths_from_dir, resolve_path_from_json,
    get_split_duration, compute_duration, process_audio_files_on_fly,
    save_to_json,
)


log = logging.getLogger(__name__)

# dictionary of datasets and relative path to json files under audio_root / dataset
DATA_SETS = {
    "mls_french_jz": None,
    "audiocite_with_metadata": "*.json", 
    "studios-tamani-kalangou-french": "v1_10102021/*.json",
    "African_Accented_French": "*.json",
    "Att-HACK_SLR88": "*.json",
    "CaFE": "*.json",
    "CFPP_corrected": "json/*.json",
    "ESLO": "*.json",
    "EPAC_flowbert": "*.json",
    "GEMEP": "*.json",
    "MPF": "*.json",
    "Portmedia": "*.json",
    "TCOF_corrected": "output/json/*.json",
    "MaSS": "*.json",
    "NCCFr": "*.json",
    "voxpopuli_unlabeled": "json/*.json",
    "voxpopuli_transcribed": "jsons/**/*.json",
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
        "--max-valid-samples",
        default=0,
        type=int,
        metavar="D",
        help="maximum number of samples to be included in validation set"
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
    parser.add_argument(
        "--debug", action="store_true", help="compare zipped data with raw audio"
    )
    args = parser.parse_args()

    start = time.time()
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    # create directory to save output tsv and zipped audio files
    output_root = Path(args.output_root)
    dataset_dir = output_root / "zipped_audio" / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rand = random.Random(args.seed) if args.valid_percent > 0 else None

    # get all audio paths with valid audio extensions
    audio_dir = Path(args.audio_root) / args.dataset
    audio_paths_from_dir = get_paths_from_dir(
        audio_dir=audio_dir,
        audio_exts=args.extensions,
        excl_pattern=None,
        result_as="dict",
    )
    logging.info(f'Number of audio files exist in corpus directory: {len(audio_paths_from_dir.keys())}')

    # get audio paths from json files
    audio_paths_from_json = None
    if DATA_SETS[args.dataset] is not None:
        logging.info(f"Getting paths from json...")
        audio_paths_from_json = resolve_path_from_json(
            audio_dir, DATA_SETS[args.dataset],
            existing_audios=audio_paths_from_dir,
            info_dir=dataset_dir,
        )
        logging.info(f'Number of valid audio files (that exist) read from JSON: {len(audio_paths_from_json.keys())}')

        # reconcile audio_paths_from_dir and audio_paths_from_json
        missing_files_from_json = {}
        duration_discarded = 0
        duration_add_back = 0
        for k, v in audio_paths_from_dir.items():
            if not k in audio_paths_from_json:
                # logging.warning(f"missing audio files from json: {v}")
                duration = compute_duration(v)
                total_split_duration = get_split_duration(k, audio_paths_from_json)
                if total_split_duration == 0:
                    if duration > 1:
                        # logging.warning(f"audio files not split longer than 1 seconds: {v}")
                        missing_files_from_json[k] = {
                            "path": v,
                            "duration": duration,
                        }
                        audio_paths_from_json[k] = v # add to training data
                        duration_add_back += duration
                    else:
                        duration_discarded += duration

        logging.info(f"Number of missing files longer than 1s: {len(missing_files_from_json.keys())}")
        logging.info(f"Total duration discarded (files less than 1s): {duration_discarded} (s)")
        logging.info(f"Total duration added back to training data: {duration_add_back} (s)")
        if len(missing_files_from_json.keys()) > 0:
            save_to_json(missing_files_from_json, dataset_dir / "audio_files_not_in_json.json")

    # now structure into train, valid, and test set
    audio_paths = (
        audio_paths_from_json if audio_paths_from_json is not None else audio_paths_from_dir
    )
    train_dict, valid_dict, test_dict = process_audio_files_on_fly(
            audio_paths=[Path(v) for _, v in audio_paths.items()],
            out_split_dir=dataset_dir / "splits_additional",
            rand=rand, 
            valid_percent=float(args.valid_percent),
            max_valid_samples=int(args.max_valid_samples),
            max_chunk_duration=int(args.max_seconds_per_file),
        )
    n_train, n_val, n_test = len(train_dict), len(valid_dict), len(test_dict)
    logging.info(f"TRAIN:{n_train} / VALID:{n_val} / TEST:{n_test} / TOTAL:{n_train+n_val+n_test}")
    logging.info(f"Number of audio files split or converted: {len(glob.glob(f'{dataset_dir}/splits_additional/*.flac'))}")

    # create zip file from each split under dataset_dir
    SPLITS = {"train": train_dict}
    logging.info(f"Creating zip for TRAIN")
    create_zip(
        data=train_dict,
        zip_prefix=dataset_dir / f"waveforms_train",
        max_num_files=int(args.max_files_per_zip),
    )
    save_to_json(train_dict, dataset_dir / "train.json")
    if n_val > 0:
        logging.info(f"Creating zip for VALID")
        SPLITS["valid"] = valid_dict
        create_zip(
            data=valid_dict,
            zip_prefix=dataset_dir / f"waveforms_valid",
            max_num_files=int(args.max_files_per_zip),
        )
        save_to_json(valid_dict, dataset_dir / "valid.json")
    if n_test > 0:
        logging.info(f"Creating zip for TEST")
        SPLITS["test"] = test_dict
        create_zip(
            data=test_dict,
            zip_prefix=dataset_dir / f"waveforms_test",
            max_num_files=int(args.max_files_per_zip),
        )
        save_to_json(test_dict, dataset_dir / "test.json")

    # create manifest file for each split under model_training
    model_training_dir = output_root / "model_training"
    model_training_dir.mkdir(parents=True, exist_ok=True)

    for split, split_dict in SPLITS.items():
        logging.info(f"Writing manifest file for split:{split.upper()}")
        create_manifest_file(
            dataset_dir,
            model_training_dir,
            split,
            original_paths_to_check=split_dict,
            debug=args.debug,
        )
    logging.info(f'Total running time: {(time.time() - start)/60} (minutes)')

if __name__ == "__main__":
    main()