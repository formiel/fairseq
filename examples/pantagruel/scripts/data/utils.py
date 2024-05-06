"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 26 April 2024
"""

import glob
import logging
from pathlib import Path
import math
import zipfile
from tqdm import tqdm
import io
import json
import itertools
import re
from itertools import islice

import soundfile as sf
import torch

from examples.speech_to_text.data_utils import get_zip_manifest
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
TRAIN_FNAME = "train"
VALID_FNAME = "valid"
TEST_FNAME = "test"


def process_audio_file(
    audio_path: Path, dataset_dir: Path, 
    rand=None, valid_percent=0.0,
    max_chunk_duration=30
):
    """
    Split or create symlink for an input audio path 
    Output(s) is saved under dataset_dir / split, where split depend on audio_path
    """
    dest_dir = dataset_dir / TRAIN_FNAME
    if "valid" in audio_path.as_posix() or "dev" in audio_path.as_posix():
        dest_dir = dataset_dir / VALID_FNAME
    elif "test" in audio_path.as_posix():
        dest_dir = dataset_dir / TEST_FNAME
    else:
        if rand is not None and rand.random() <= valid_percent:
            dest_dir = dataset_dir / TRAIN_FNAME

    sample_rate = sf.info(audio_path).samplerate
    max_chunk_frames = max_chunk_duration * sample_rate
    duration = sf.info(audio_path.as_posix()).frames / sample_rate
    num_chunks = math.ceil(duration / max_chunk_duration)
    if num_chunks > 1:
        logging.warning(f"splitting {audio_path.as_posix()} into {num_chunks} chunks...")
        for start in range(num_chunks):
            waveform, _ = sf.read(
            audio_path, frames=max_chunk_frames, start=start*max_chunk_frames
        ) # T
            idx = "_{:02d}".format(start)
            logging.info(f"writing to {dest_dir / f'{audio_path.stem}{idx}.flac'}")
            sf.write(
                (dest_dir / f"{audio_path.stem}{idx}.flac").as_posix(),
                waveform, 16_000
            )
    else:
        sym_path = dest_dir / audio_path.name
        if not sym_path.is_symlink():
            sym_path.symlink_to(audio_path)


def get_paths_from_dir(
    audio_dir: Path, audio_exts="wav,flac", excl_pattern=r'\/split(s)?\/'
):
    """
    Get all files with specified extensions in all sub-level directories
    except for those whose path containing "split" or "splits"
    """
    audio_paths = itertools.chain()
    for ext in audio_exts.split(","):
        audio_paths = itertools.chain(
            audio_paths, 
            glob.glob(f'{audio_dir}/**/*.{ext}', recursive=True)
        )
    audio_paths = list(audio_paths)
    filtered_paths = [file for file in audio_paths if re.search(excl_pattern, file) is None]
    return filtered_paths


def get_paths_from_json(json_path):
    assert json_path.exists()
    with open(json_path, 'r') as f:
        data = json.load(f)
    audio_paths = []
    if hasattr(data, "corpus"):
        audio_paths = [d["path"] for d in data["corpus"]]
    else:
        for _, v in data.items():
            audio_paths.append(v["path"])
    return audio_paths


def create_zip(
    data_root: Path, 
    zip_prefix: Path, 
    extensions="wav,flac,npy",
    max_num_files=None,
):
    """
    [OLD code without multiprocessing]
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


def get_zip_info(data_root: Path, extensions="wav,flac,npy", max_num_files=None):
    extensions = extensions.split(",")
    paths = []
    for ext in extensions:
        paths.extend(data_root.glob(f"*.{ext}"))

    num_zip_files = 0
    if max_num_files is not None:
        num_zip_files = math.ceil(len(paths) / max_num_files)
    
    return paths, num_zip_files, max_num_files


def create_zip_file(zip_file_i, paths):
    with zipfile.ZipFile(zip_file_i, "w", zipfile.ZIP_STORED) as z:
        for path in tqdm(paths, desc=f"Creating {zip_file_i}"):
            z.write(path, arcname=path.name)


def chunk_paths(paths, chunk_size):
    """Split paths into chunks of given size."""
    it = iter(paths)
    return iter(lambda: list(islice(it, chunk_size)), [])


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