"""
Author: Phuong-Hang Le (hangtp.le@gmail.com)
Date: 26 April 2024
"""

import glob
import logging
from typing import Union, List, Optional
import numpy as np
from pathlib import Path
import math
import zipfile
from tqdm import tqdm
import io
import json
import itertools
import re
from itertools import islice
import subprocess

import soundfile as sf
import torch

from examples.speech_to_text.data_utils import is_npy_data
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
TRAIN_FNAME = "train"
VALID_FNAME = "valid"
TEST_FNAME = "test"
DEFAULT_SAMPLE_RATE = 16000


def convert_audio(input_wav, output_wav):
    command = ['ffmpeg', '-i', input_wav, '-ar', '16000', output_wav]
    subprocess.run(command)


def save_to_json(data: dict, saved_path: Path):
    with open(saved_path.as_posix(), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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


def process_audio_files_on_fly(
    audio_paths: List[Path], 
    out_split_dir: Path,
    rand=None, 
    valid_percent=0.0,
    max_valid_samples=0,
    max_chunk_duration=30,
):
    train_dict, valid_dict, test_dict = {}, {}, {}
    num_file_splits = 0
    num_valid = 0
    if not out_split_dir.exists():
        out_split_dir.mkdir(parents=True, exist_ok=True)
    for audio_path in audio_paths:
        dest_dict = train_dict
        if "valid" in audio_path.as_posix() or "dev" in audio_path.as_posix():
            dest_dict = valid_dict
        elif "test" in audio_path.as_posix():
            dest_dict = test_dict
        else:
            if rand is not None and rand.random() <= valid_percent and num_valid < max_valid_samples:
                dest_dict = valid_dict
                num_valid += 1

        sample_rate = sf.info(audio_path).samplerate
        if sample_rate != DEFAULT_SAMPLE_RATE:
            convert_audio(
                input_wav=audio_path,
                output_wav=(out_split_dir / f"{audio_path.stem}.flac").as_posix()
            )
            audio_path = out_split_dir / f"{audio_path.stem}.flac"
        num_frames = sf.info(audio_path).frames
        if num_frames < 3000:
            continue
        max_chunk_frames = max_chunk_duration * DEFAULT_SAMPLE_RATE
        duration = num_frames / DEFAULT_SAMPLE_RATE
        num_chunks = math.ceil(duration / max_chunk_duration)
        if num_chunks > 2:
            for start in range(num_chunks-2):
                dest_dict = read_and_save(
                    audio_path=Path(audio_path), 
                    frames=max_chunk_frames,
                    offset=start*max_chunk_frames,
                    idx="_{:02d}".format(start),
                    out_dir=out_split_dir,
                    dest_dict=dest_dict,
                )
            # combine last 2 chunks
            dest_dict = read_and_save(
                audio_path=Path(audio_path), 
                frames=max_chunk_frames*2,
                offset=(start+1)*max_chunk_frames,
                idx="_{:02d}".format(start+1),
                out_dir=out_split_dir,
                dest_dict=dest_dict,
                )
            num_file_splits += num_chunks - 1
        else:
            dest_dict[audio_path.stem] = audio_path.as_posix()
    
    return train_dict, valid_dict, test_dict


def read_and_save(audio_path: Path, frames, offset, idx, out_dir, dest_dict):
    waveform, _ = sf.read(audio_path, frames=frames, start=offset) # T
    out_path = out_dir / f'{audio_path.stem}{idx}.flac'
    logging.info(f"writing to {out_path}")
    sf.write(out_path, waveform, 16_000)
    dest_dict[out_path.stem] = out_path.as_posix()
    return dest_dict


def get_paths_from_dir(
    audio_dir: Path, audio_exts="wav,flac", excl_pattern=r'\/split(s)?\/',
    result_as="list",
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
    if excl_pattern is not None:
        filtered_paths = [file for file in audio_paths if re.search(excl_pattern, file) is None]
    else:
        filtered_paths = audio_paths
    if result_as == "list":
        return filtered_paths
    elif result_as == "dict":
        return {Path(p).stem: p for p in filtered_paths}
    else:
        raise NotImplementedError


def get_paths_from_json(json_path):
    if not json_path.exists():
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    audio_paths = []
    if hasattr(data, "corpus"):
        audio_paths = [d["path"] for d in data["corpus"]]
    else:
        for _, v in data.items():
            audio_paths.append(v["path"])
    return audio_paths


def is_text_io_wrapper(obj):
    return isinstance(obj, io.TextIOWrapper)


def compute_duration(audio_path):
    sample_rate = sf.info(audio_path).samplerate
    duration = sf.info(audio_path).frames / sample_rate
    return duration


def is_long_audio(audio_path):
    return True if compute_duration(audio_path) > 30 else False


def get_split_duration(prefix_key, dictionary):
    split_duration = 0
    for k, v in dictionary.items():
        if k.startswith(prefix_key):
            # logging.info(f"{prefix_key} is split into {k}")
            split_duration += compute_duration(v)
    return split_duration


def resolve_path_from_json(
    dataset_root: Path, rel_jsons, existing_audios,
    info_dir,
):
    """
    read paths in json files and include only those that exist
    """
    json_paths = glob.glob(f'{dataset_root}/{rel_jsons}')
    utterances = {} # dict of utt_id: path

    for json_path in json_paths:
        json_path = Path(json_path)
        assert json_path.exists()
        with open(json_path, 'r') as f:
            if is_text_io_wrapper(f):
                for line in f:
                    line = line.strip()
                    if "path" in line:
                        # correct the path
                        audio_fname = line.split(": ")[-1]
                        audio_fname = Path(
                            audio_fname.replace('"', "").strip()
                        ).stem
                        if not audio_fname in utterances:
                            if audio_fname not in existing_audios:
                                with open(info_dir / "invalid_paths.txt", "w") as invalid_f:
                                    print(line, file=invalid_f)
                            else:
                                utterances[audio_fname] = existing_audios[audio_fname]
            else:
                data = json.load(f)
                logging.info(f'Keys in json: {len(data.keys())}')
                raise NotImplementedError

    return utterances


def create_zip(
    data: Union[Path, dict], 
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
    if isinstance(data, Path):
        extensions = extensions.split(",")
        paths = []
        for ext in extensions:
            paths.extend(data.glob(f"*.{ext}"))
    elif isinstance(data, dict):
        paths = [Path(v) for _, v in data.items()]
    else:
        raise NotImplementedError

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

def include_accented_char(word):
    for char in word:
        if ord(char) > 128:
            return True
    return False

def get_zip_manifest(
        zip_path: Path, zip_root: Optional[Path] = None, is_audio=False
):
    _zip_path = Path.joinpath(zip_root or Path(""), zip_path)
    with zipfile.ZipFile(_zip_path, mode="r") as f:
        info = f.infolist()
    paths, lengths = {}, {}
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        fname = i.filename
        if include_accented_char(fname):
            fname = fname.encode('utf-8')
        offset, file_size = i.header_offset + 30 + len(fname), i.file_size
        paths[utt_id] = f"{zip_path.as_posix()}:{offset}:{file_size}"
        with open(_zip_path, "rb") as f:
            f.seek(offset)
            byte_data = f.read(file_size)
            assert len(byte_data) > 1
            if is_audio:
                assert is_sf_audio_data(byte_data), i
            else:
                assert is_npy_data(byte_data), i
            byte_data_fp = io.BytesIO(byte_data)
            if is_audio:
                lengths[utt_id] = sf.info(byte_data_fp).frames
            else:
                lengths[utt_id] = np.load(byte_data_fp).shape[0]
    return paths, lengths


def create_manifest_file(
    dataset_dir: Path,
    tsv_dir:Path,
    split: str,
    original_paths_to_check=None,
    debug=False,
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

    logging.info(f'Total duration of {split.upper()}: {sum([v for _, v in data_lengths.items()])/16000/3600} (h)')
    tsv_f = open(tsv_dir /  f"{split}_{dataset_dir.stem}.tsv", "w")
    print(dataset_dir.parent.as_posix(), file=tsv_f) # writing the audio root directory
    for utt_id, path in data_zip.items():
        rel_path = Path(path).relative_to(dataset_dir.parent).as_posix()
        print(
                f"{rel_path}\t{data_lengths[utt_id]}", file=tsv_f
            )
        if debug:
            # check if zipped data is the same as raw audio file
            _path, slice_ptr = parse_path(path)
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)
            wav, _ = sf.read(path_or_fp)
            feats = torch.from_numpy(wav).float()

            if original_paths_to_check is not None:
                wav_check, _ = sf.read(original_paths_to_check[utt_id])
            else:
                try:
                    wav_check, _ = sf.read(dataset_dir / split / f"{utt_id}.flac")
                except:
                    wav_check, _ = sf.read(dataset_dir / split / f"{utt_id}.wav")
            feats_check = torch.from_numpy(wav_check).float()
            assert torch.equal(feats, feats_check)

    tsv_f.close()