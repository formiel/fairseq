"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 30 January 2025

Input: a json file including data samples, each sample has path to audio file and
        corresponding transcript, among other info.

Output:
- a manifest file ${DATASET}.tsv for each dataset in which each line includes
the audio path and its transcript 
- a zipped audio file
"""
import os
import argparse
from pathlib import Path
import shutil
import json
import tarfile
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
import io

import torch
from torch.utils.data import Dataset

from examples.pantagruel.data.raw_audio_augment import WaveformAugmentation
from examples.speech_to_text.data_utils import (
    create_zip, save_df_to_tsv
)
from fairseq.data.audio.audio_utils import (
    read_from_stored_zip, parse_path, is_sf_audio_data
)
from utils import (
    save_to_json, get_zip_manifest
)


LB_UNLABEL_SPEECH_DIR = Path("/lustre/fsn1/projects/rech/zbk/commun/Data/LeBenchmark_prepared/zipped_audio")
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]
ZIPPED_AUDIO_SUBFOLDER = "zipped_audio"
TSV_SUBFOLDER = "model_training"


class JSONDataset(Dataset):
    def __init__(
        self, json_path, dataset_name, split, audio_dir, 
        out_root=None, musan_dir=None
    ):
        """
        Args:
            json_path (str): Path to the JSON file.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.split = split
        self.out_root = out_root
        self.dataset_name = dataset_name
        json_data = read_json(json_path)
        try:
            tar_data = read_tar(audio_dir) # {audio_id: waveform}
        except:
            tar_data = {}
        
        zip_data = read_zip(
            LB_UNLABEL_SPEECH_DIR / dataset_name, split=self.split
        ) # {audio_id: path_in_zip}

        valid_ids = get_ids(
                LB_UNLABEL_SPEECH_DIR / dataset_name / "valid.json"
            )
        print(f"- json_data: {len(json_data)}\n"
              f"- valid_ids: {len(valid_ids)}\n"
              f"- tar_data: {len(tar_data)}\n"
              f"- zip_data: {len(zip_data)}\n"
            )
        self.data = self.check_samples(
            json_data, tar_data, zip_data, valid_ids
        )
        self.sample_ids = list(self.data.keys())

        self.audio_transform, self.flac_aug = None, None
        if musan_dir is not None:
            self.audio_transform = WaveformAugmentation(Path(musan_dir))

    def check_samples(self, json_data, tar_data, zip_data, valid_ids):
        data = {}
        json_samples_wo_trans = {}
        json_samples_wo_audio = {}

        for _id, _data in json_data.items():
            if (
                self.split == "train" and _id not in valid_ids
            ) or (
                self.split == "valid" and _id in valid_ids
            ):
                if len(json_data[_id]["trans"]) > 0:
                    if _id in tar_data:
                        data[_id] = (tar_data[_id], _data) # {_id: (wav, json_data_dict)}
                    elif _id in zip_data:
                        data[_id] = (zip_data[_id], _data) # {_id: (zip_path, json_data_dict)}
                    else:
                        json_samples_wo_audio[_id] = _data
                else:
                    json_samples_wo_trans[_id] = _data

        print(f"- json_samples_wo_trans: {len(json_samples_wo_trans)}\n"
              f"- json_samples_wo_audio: {len(json_samples_wo_audio)}"
            )
        out_dir = self.out_root / ZIPPED_AUDIO_SUBFOLDER / self.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(json_samples_wo_trans) > 0:
            save_to_json(
                json_samples_wo_trans, out_dir / f"{self.split}_json_samples_wo_trans.json"
            )
        if len(json_samples_wo_audio) > 0:
            save_to_json(
                json_samples_wo_audio, out_dir / f"{self.split}_json_samples_wo_audio.json"
            )
        return data

    def __len__(self):
        return len(self.sample_ids)  # Number of samples

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={len(self.sample_ids):_}, '
            f"self.audio_transform={self.audio_transform}."
        )

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]  # Get sample ID
        _audio_path_or_wav, _json_data = self.data[sample_id]  # Retrieve data dictionary
        waveform = self.get_waveform(_audio_path_or_wav)

        data = {
            "id": sample_id,
            "audio": waveform,
            "n_frames": waveform.shape[0],
            "tgt_text": _json_data["trans"],
            "speaker": _json_data["spk_id"],
            "spk_gender": getattr(_json_data, "spk_gender", "unk"),
            }

        if self.audio_transform is not None:
            waveform_aug = self.audio_transform(torch.tensor(waveform))
            data["audio_aug"] = waveform_aug
            data["n_frames_aug"] = waveform_aug.shape[0]
        return data

    def get_waveform(self, _audio_path_or_wav):
        if isinstance(_audio_path_or_wav, np.ndarray):
            wav = _audio_path_or_wav
        else:
            wav, _ = get_wav_from_zip_path(_audio_path_or_wav)
        return wav


class MLSFrenchDataset(Dataset):
    def __init__(self, data_root, split, musan_dir=None):

        self.split = split
        self.audio_data = self.load_data_from_tsv(data_root, split)
        self.sample_ids = list(self.audio_data.keys())

        self.audio_transform = None
        if musan_dir is not None:
            self.audio_transform = WaveformAugmentation(Path(musan_dir))

        self.data = {}
        trans_path = Path(data_root) / f"{self.split}_transcripts.txt"
        with open(trans_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            items = line.strip().split("\t")
            self.data[Path(items[0]).stem] = items[1].strip()

    def __len__(self):
        return len(self.sample_ids)  # Number of samples

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={len(self.sample_ids):_}, '
            f"self.audio_transform={self.audio_transform}."
        )

    def load_data_from_tsv(
        self, data_root: Path, split: str
    ):
        manifest_path = Path(data_root) / f"{split}.tsv"
        with open(manifest_path, "r") as f:
            root_dir = Path(f.readline().strip())
            audio_files = [root_dir / line.strip().split("\t")[0] for line in f]

        audio_data = {k.stem: k for k in audio_files}
        return audio_data

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]  # Get sample ID
        waveform, sample_rate = sf.read(self.audio_data[sample_id])
        # Apply transformation if provided
        waveform_aug = None
        if self.audio_transform:
            waveform_aug = self.audio_transform(torch.tensor(waveform))
        assert len(self.data[sample_id]) > 0
        data = {
            "id": sample_id,
            "tgt_text": self.data[sample_id],
            "speaker": sample_id.split("_")[0],
            "audio": waveform,
            "sample_rate": sample_rate,
            "n_frames": waveform.shape[0],
        }
        if waveform_aug is not None:
            data["audio_aug"] = waveform_aug
            data["n_frames_aug"] = waveform_aug.shape[0]
        return data


def get_wav_from_zip_path(path_from_zip):
    _path, slice_ptr = parse_path(path_from_zip)
    byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
    path_or_fp = io.BytesIO(byte_data)
    wav, sr = sf.read(path_or_fp)
    # sf.write(out_dir / f"{sample_id}.flac", wav, 16_000)
    return wav, sr


def read_json(json_path):
    # CFPP_corrected, African_Accented_French, Att-HACK_SLR88, CaFE, MPF, Portmedia, TCOF_corrected
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def read_tar(tar_dir):
    file_size = os.path.getsize(tar_dir)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    print(f"Size of tar file: {file_size_gb:.2f} GB")
    # if file_size_gb >= max_tar_gb:
    #     print("Will read from the zipped file instead.")
    #     raise NotImplementedError

    audio_data = {}
    with tarfile.open(tar_dir) as archive:
        for f in archive.getmembers():
            flac_file = archive.extractfile(f)
            if flac_file == None:
                continue
            wav, sr = sf.read(flac_file)
            audio_data[Path(f.name).stem] = wav
    return audio_data


def get_ids(split_json_path):
    with open(split_json_path, "r") as file:
        ids = json.load(file)
    return ids


def read_zip(zipped_dir, split="train"):
    audio_paths = {}
    zipped_folder = Path(zipped_dir)
    zip_files = list(zipped_folder.glob("*.zip"))  # List all .zip files in the folder
    print(f"zip_files: {zip_files}")
    for _zip_path in zip_files:
        _split = Path(_zip_path).stem.split("_")[1]
        if _split == split:
            _audio_paths, _ = get_zip_manifest(
                    _zip_path,
                    is_audio=True,
                )
            audio_paths.update(_audio_paths) # {audio_id: path_in_zip_file}
    return audio_paths


def process(args):
    SPLITS = ["train", "valid"]
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(parents=True, exist_ok=True)

    feature_root = out_root / ZIPPED_AUDIO_SUBFOLDER / args.dataset_name
    # if feature_root.exists():
    #     shutil.rmtree(feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)

    tsv_root = out_root / TSV_SUBFOLDER
    tsv_root.mkdir(parents=True, exist_ok=True)
    SUFFIXES = [""]

    for split in SPLITS:
        flac_root = feature_root / "flacs" / split
        flac_root.mkdir(parents=True, exist_ok=True)
        if args.musan_dir is not None:
            flac_root_aug = feature_root / "flacs" / f"{split}_aug"
            flac_root_aug.mkdir(parents=True, exist_ok=True)
            SUFFIXES.append("_aug")

        if args.dataset_name not in ["mls_french", "ESLO2"]:
            dataset = JSONDataset(
                json_path=args.json,
                dataset_name=args.dataset_name,
                split=split,
                audio_dir=args.audio_dir,
                out_root=out_root,
                musan_dir=args.musan_dir
            )
        elif args.dataset_name == "mls_french":
            # for MLS, we rely on the existing .tsv and .wrd files
            dataset = MLSFrenchDataset(
                data_root=args.audio_dir,
                split=split,
                musan_dir=args.musan_dir,
            ) 
        else:
            raise NotImplementedError
        print(f"***** {dataset} *****")
        
        # create directory to save original or augmented audio files
        for item in tqdm(dataset):
            for sfx in SUFFIXES:
                if f"audio{sfx}" in item:
                    _flac_root = flac_root_aug if "aug" in sfx else flac_root
                    sf.write(
                        (_flac_root / f"{item['id']}.flac").as_posix(), 
                        item[f'audio{sfx}'], 16_000
                    )

        audio_paths_all = {}
        audio_lengths_all = {}
        for sfx in SUFFIXES:
            _flac_root = flac_root_aug if "aug" in sfx else flac_root
            _zip_path = feature_root / f"waveforms_{split}{sfx}.zip"
            print("ZIPing audios/features...")
            create_zip(_flac_root, _zip_path)
            print("Fetching ZIP manifest...")
            audio_paths, audio_lengths = get_zip_manifest(
                _zip_path,
                is_audio=True,
            ) # len(audio_lengths) >= len(audio_paths)
            audio_paths_all[f"{split}{sfx}"] = audio_paths
            audio_lengths_all[f"{split}{sfx}"] = audio_lengths

        # Generate TSV manifest
        print(f"Generating {split.upper()} manifest...")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for item in tqdm(dataset):
            if item["id"] in audio_lengths_all[split]:
                manifest["id"].append(item["id"])
                manifest["audio"].append(audio_paths_all[split][item["id"]])
                manifest["n_frames"].append(audio_lengths_all[split][item["id"]])
                manifest["tgt_text"].append(item["tgt_text"])
                manifest["speaker"].append(item["speaker"])
                if 'audio_aug' in item:
                    manifest["audio_aug"].append(audio_paths_all[f"{split}_aug"][item["id"]])
                    manifest["n_frames_aug"].append(audio_lengths_all[f"{split}_aug"][item["id"]])
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, tsv_root / f"{split}_{args.dataset_name}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--musan-dir", type=str, default=None)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()