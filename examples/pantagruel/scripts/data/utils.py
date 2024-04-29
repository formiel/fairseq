"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 26 April 2024
"""

import glob
from pathlib import Path
import math
import zipfile
from tqdm import tqdm
import io

import soundfile as sf
import torch

from examples.speech_to_text.data_utils import get_zip_manifest
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)


def create_zip(
    data_root: Path, 
    zip_prefix: Path, 
    extensions="wav,flac,npy",
    max_num_files=None,
):
    """
    Create zip files for all files in a given folder.

    Args:
        data_root: folder to zip
        zip_prefix: prefix to the output zip folder
        extensions: extensions of files to be zipped
        max_num_files: number of files to be included in a zip file
    """
    extensions = extensions.split(",")
    paths = []
    for ext in extensions:
        paths.extend(data_root.glob(f"*.{ext}"))

    num_zip_files = 0
    if max_num_files is not None:
        num_zip_files = math.ceil(len(paths) / max_num_files)

    for i in range(num_zip_files):
        zip_file_i = f"{zip_prefix.as_posix()}_{i}.zip"
        with zipfile.ZipFile(zip_file_i, "w", zipfile.ZIP_STORED) as f:
            for path in tqdm(paths[i*max_num_files : (i+1)*max_num_files]):
                f.write(path, arcname=path.name)


def create_manifest_file(
    dataset_dir: Path,
    tsv_dir:Path,
    split: str,
):
    """
    Create manifest tsv file for each split of the dataset
    """
    zip_paths = glob.glob(f"{dataset_dir.as_posix()}/*_{split}_*.zip")
    data_zip, data_lengths = {}, {}
    for zip_path in zip_paths:
        audio_paths, audio_lengths = get_zip_manifest(
            Path(zip_path),
            is_audio=True,
        )
        data_zip.update(audio_paths)
        data_lengths.update(audio_lengths)

    tsv_f = open(tsv_dir /  f"{split}_{dataset_dir.stem}.tsv", "w")
    print(dataset_dir.as_posix(), file=tsv_f) # writing the root directory
    for utt_id, path in data_zip.items():
        rel_path = Path(path).relative_to(dataset_dir)
        print(
                f"{rel_path}\t{data_lengths[utt_id]}", file=tsv_f
            )
        # check if zipped data is the same as raw audio file # may slow down processing
        _path, slice_ptr = parse_path(path)
        byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        assert is_sf_audio_data(byte_data)
        path_or_fp = io.BytesIO(byte_data)
        wav, _ = sf.read(path_or_fp)
        feats = torch.from_numpy(wav).float()

        try:
            wav_2, _ = sf.read(dataset_dir / split / f"{utt_id}.flac")
        except:
            wav_2, _ = sf.read(dataset_dir / split / f"{utt_id}.wav")
        feats_2 = torch.from_numpy(wav_2).float()
        assert torch.equal(feats, feats_2)

    tsv_f.close()