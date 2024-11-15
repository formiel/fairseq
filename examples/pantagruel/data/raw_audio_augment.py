import logging
import os
import time
import io
import glob
from pathlib import Path
import soundfile as sf
from typing import Callable
from scipy.signal import convolve

import numpy as np
import random
import torchaudio
import torchaudio.functional as F

import torch

from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from fairseq.data.data_utils import compute_block_mask_1d
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)


CATEGORIES_SNR_MIN_MAX = {
    "noise": (-10, 35),
    "speech": (5, 20),
    "music": (0, 20)
}
MAX_AUGS = 6


class WaveformAugmentation:

    def __init__(self, musan_dir: Path):
        rir_pattern = musan_dir / "simulated_rirs" / "**" / "**" / "*.wav"
        self.rir_files = glob.glob(str(rir_pattern), recursive=True)

        self.musan_files = {}
        musan_path = musan_dir / "musan_split" / "**" / "**" / "*.wav"
        for file in glob.glob(str(musan_path), recursive=True):
            category = Path(file).parts[-3]  # Get the category from the file path
            self.musan_files.setdefault(category, []).append(file)

        self.augment_types = list(CATEGORIES_SNR_MIN_MAX.keys())
        self.augment_types.append("rir")
        self.augment_types.append("speed_perturb")

        logger.info(self.__repr__())

    def __repr__(self):
        snr_values = ", ".join(
            [f"{cat}: {len(self.musan_files[cat])}" for cat in CATEGORIES_SNR_MIN_MAX]
        )
        return (
            f"{self.__class__.__name__}("
            f"n_samples_rir={len(self.rir_files)}, "
            f"n_samples_musan_noise_speech_music={snr_values}, "
            f"augment_types={self.augment_types}",
            ")"
        )
    
    def apply_rir(self, speech: torch.Tensor):
        rir_file = random.choice(self.rir_files)

        rir, _ = sf.read(rir_file)
        rir = torch.from_numpy(rir).float()
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        augmented_speech = F.fftconvolve(speech, rir)

        return augmented_speech
    
    def get_noise_and_snr(self, target_len, category=None):
        if not category:
            category = random.sample(list(CATEGORIES_SNR_MIN_MAX.keys()), 1)[0]
        musan_file = random.choice(self.musan_files[category])
        noise, _ = sf.read(musan_file)
        noise = torch.from_numpy(noise).float()
        noise_len = noise.size()[0]
        if noise_len < target_len:
            noise = torch.tile(noise, (1, target_len // noise_len + 1))
        noise = noise[: target_len]

        snr_min, snr_max = CATEGORIES_SNR_MIN_MAX[category]
        snr = random.uniform(snr_min, snr_max)
        return noise, torch.tensor(snr, dtype=torch.int8)

    def apply_musan_noise(self, speech: torch.Tensor):
        noise, snr = self.get_noise_and_snr(speech.size()[-1])
        return F.add_noise(speech, noise, snr)
    
    def apply_speed_perturb(self, speech: torch.Tensor, sample_rate=16000):
        speed_perturb = torchaudio.transforms.SpeedPerturbation(sample_rate, [0.9, 1.1, 1.0])
        return speed_perturb(speech)[0]
    
    def get_aug_fn(self, aug_type) -> Callable:
        if aug_type in list(CATEGORIES_SNR_MIN_MAX.keys()):
            return self.apply_musan_noise
        elif aug_type == "speed_perturb":
            return self.apply_speed_perturb
        elif aug_type == "rir":
            return self.apply_rir
        else:
            raise NotImplementedError

    def __call__(self, speech: torch.Tensor):
        num_augs = random.randint(1, MAX_AUGS)
        for _ in range(num_augs):
            aug_type = random.choice(self.augment_types)
            aug_fn = self.get_aug_fn(aug_type)
            speech = aug_fn(speech)
        return speech


class FileAudioAugmentDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        musan_dir=None,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)
        self.audio_transform = WaveformAugmentation(Path(musan_dir))

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        retry = 3
        wav = None
        for i in range(retry):
            try:
                wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
                break
            except Exception as e:
                logger.warning(
                    f"Failed to read {path_or_fp}: {e}. Sleeping for {1 * i}"
                )
                time.sleep(1 * i)

        if wav is None:
            raise Exception(f"Failed to load {path_or_fp}")

        feats = torch.from_numpy(wav).float()

        # apply augmentation
        augmented_audio = self.audio_transform(feats)

        feats = self.postprocess(feats, curr_sample_rate)
        aug_source = self.postprocess(augmented_audio, curr_sample_rate)

        v = {"id": index, 
             "source": (feats, aug_source), 
            }

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )
            v["precomputed_mask"] = mask
        return v
    
    def _collate(self, samples, key=0):
        sources = [s["source"][key] for s in samples] # original audio
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        return collated_sources, padding_mask

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        collated_sources, padding_mask = self._collate(samples, key=0) # original audio
        input = {"source": collated_sources}
        if self.corpus_key is not None:
            input["corpus_key"] = [self.corpus_key] * len(collated_sources)
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self.crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        out["net_input"] = input

        collated_sources_aug, padding_mask_aug = self._collate(samples, key=1) # original audio
        out["net_input"]["source_aug"] = {"source": collated_sources_aug, "padding_mask": padding_mask_aug}

        return out