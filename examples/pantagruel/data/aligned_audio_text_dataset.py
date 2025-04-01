import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from functools import partial

from omegaconf import II

import numpy as np
import io
import soundfile as sf
import time

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("The 'transformers' library is not installed. Please install it by running 'pip install transformers'.")

from fairseq.data import FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks.audio_pretraining import AudioMaskingConfig

from examples.pantagruel.data.raw_audio_augment import WaveformAugmentation


logger = logging.getLogger(__name__)


UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"


@dataclass
class AlignedSpeechTextDatasetItem(object):
    index: int
    audio: Dict[str, torch.Tensor]
    text: Dict[str, torch.Tensor]
    speaker_id: Optional[int] = None


@dataclass
class TextDataConfig(FairseqDataclass):
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with *mask_idx*."}
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"}
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a masked token with a \
            random token from the vocabulary"}
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on \
            word frequencies in the vocab"}
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "standard deviation of masks distribution in case of \
            multiple masking"}
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat each mask index multiple times."}
    )


@dataclass
class AudioDataConfig(FairseqDataclass):
    normalize: bool = field(
        default=True,
        metadata={"help": "if set, normalizes audio waveform to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=True, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=1000000, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=2, metadata={"help": "min sample size to skip small examples"}
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    rebuild_batches: bool = True
    precompute_mask_config: Optional[AudioMaskingConfig] = None
    musan_dir: str = field(
        default="",
        metadata={"help": "musan dir for audio augmentation. None means no augmentation applied"}
    )


@dataclass
class AlignedSpeechTextConfig(FairseqDataclass):
    audio: Optional[AudioDataConfig] = None
    text: Optional[TextDataConfig] = None
    shuffle: bool = field(
        default=True,
        metadata={"help": "shuffle data"},
    )
    seed: int = II("common.seed")


class AlignedSpeechTextDataset(FairseqDataset):

    def __init__(
        self,
        data_root: str,
        split: str,
        cfg: AlignedSpeechTextConfig,
        tokenizer: AutoTokenizer,
        speaker_to_id=None,
    ):
        self.split = split
        self.cfg = cfg
        self.speaker_to_id = speaker_to_id

        self.tokenizer = tokenizer
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.pad_idx = self.tokenizer.pad_token_id
        self.unk_idx = self.tokenizer.unk_token_id
        self.mask_idx = self.tokenizer.mask_token_id
        logger.info(f"self.bos_idx: {self.bos_idx},\t"
                    f"self.eos_idx: {self.eos_idx},\t"
                    f"self.pad_idx: {self.pad_idx},\t"
                    f"self.unk_idx: {self.unk_idx},\t"
                    f"self.mask_idx: {self.mask_idx}."
        )

        data = self._load_data_from_csv(data_root=data_root, split=split)
        self.ids, self.speakers, self.n_frames, self.audios, self.texts = data
        self.n_samples = len(self.audios)

        self.mask_args = cfg.audio.precompute_mask_config or {}
        self.clone_batch = self.mask_args.get("clone_batch", 1)

        self.feature_encoder_spec = eval(self.mask_args["feature_encoder_spec"])
        self._features_size_map = {}

        self.epoch = 0

        self.text_lens = self.get_text_lens_and_check_oov()

        assert self.cfg.audio.enable_padding, "Not supported cropping for aligned dataset yet"

        self.audio_transform = None
        if self.cfg.audio.musan_dir:
            self.audio_transform = WaveformAugmentation(Path(self.cfg.audio.musan_dir))

        logger.info(self.__repr__())

    def get_text_lens_and_check_oov(self):
        if self.texts is None:
            return [0 for _ in range(self.n_samples)]
        text_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_text_item(i)
            oov_tokens = [t for t in tokenized if t == self.unk_idx]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            text_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return text_lens
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"split={self.split}, n_samples={self.n_samples:_}, "
            f"vocab size={len(self.tokenizer)}, "
            f"padded_audio={self.cfg.audio.enable_padding}, "
            f"max_sample_size={self.cfg.audio.max_sample_size}, "
            f"min_sample_size={self.cfg.audio.min_sample_size}, "
            f"normalized_audio={self.cfg.audio.normalize}, "
            # f"audio_transform={self.audio_transform})"
        )

    def __len__(self):
        return self.n_samples

    def _load_data_from_csv(
        self, 
        data_root: str, 
        split: str,
    ):
        # load samples from csv
        tsv_path = Path(data_root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        
        # build data
        ids = [s["id"] for s in samples]
        speakers = [s["speaker"] for s in samples]
        n_frames = [int(s["n_frames"]) for s in samples]
        audios = [s["audio"] for s in samples]
        texts = [s["tgt_text"] for s in samples]
        return (
            ids, speakers, n_frames, audios, texts
        )
    
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_text_item(self, index: int):
        seed = int(hash((self.cfg.seed, self.epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)

        text = self.texts[index]
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens).long()
        assert self.mask_idx not in tokens
        
        # if self.cfg.text.prepend_bos:
        #     tokens = torch.cat(
        #         (torch.tensor([self.bos_idx], dtype=torch.int64), tokens), dim=0)
        # if self.cfg.text.append_eos:
        #     tokens = torch.cat(
        #         (tokens, torch.tensor([self.eos_idx], dtype=torch.int64)), dim=0)
        # if self.cfg.text.remove_bos_eos:
        #     assert tokens[0] == self.bos_idx and tokens[-1] == self.eos_idx
        #     tokens = tokens[1:-1]

        sz = len(tokens)
        text_item = {"id": index, "source": tokens}
        
        mask_generator = partial(
            create_single_mask, sz, self.cfg.text.mask_prob,
            self.cfg.text.mask_multiple_length, 
            self.cfg.text.mask_stdev, rng
        )
        masks = [mask_generator() for _ in range(self.clone_batch)]
        max_length = max(mask.shape[0] for mask in masks)
        padded_masks = torch.stack([F.pad(torch.tensor(mask), (0, max_length - mask.shape[0])) for mask in masks])
        text_item["precomputed_mask"] = torch.tensor(
            padded_masks.clone().detach().requires_grad_(True)
        ) # clone_batch x len

        # mask = create_single_mask(
        #     sz, self.cfg.text.mask_prob, self.cfg.text.mask_multiple_length,
        #     self.cfg.text.mask_stdev, rng
        # )
        # new_item = tokens.clone()
        # new_item[mask] = self.mask_idx

        # text_item["source"] = new_item
        # text_item["mask"] = mask

        return text_item

    def get_audio_item(self, index: int):
        path_or_fp = self.audios[index]
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
        if self.audio_transform is not None:
            feats = self.audio_transform(feats)
        feats = self.postprocess(feats, curr_sample_rate)
        audio_item = {"id": index, "source": feats}

        T = self._get_mask_indices_dims(feats.size(-1))
        mask = fairseq_data_utils.compute_block_mask_1d(
            shape=(self.clone_batch, T),
            mask_prob=self.mask_args.mask_prob,
            mask_length=self.mask_args.mask_length,
            mask_prob_adjust=self.mask_args.mask_prob_adjust,
            inverse_mask=self.mask_args.inverse_mask,
            require_same_masks=True,
            expand_adjcent=False,
            mask_dropout=self.mask_args.mask_dropout,
            non_overlapping=False,
        )
        audio_item["precomputed_mask"] = mask # clone_batch x len

        return audio_item
    
    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self.feature_encoder_spec:
            L_in = size
            for (_, kernel_size, stride) in self.feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.cfg.audio.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.cfg.audio.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.cfg.audio.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats
    
    def __getitem__(self, index: int) -> AlignedSpeechTextDatasetItem:
        audio_item = self.get_audio_item(index)
        text_item = self.get_text_item(index)
        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return AlignedSpeechTextDatasetItem(
            index=index,
            audio=audio_item,
            text=text_item,
            speaker_id=speaker_id,
        )
    
    def collater(
        self, samples: List[AlignedSpeechTextDatasetItem]) -> Dict:
        samples = [
            s for s in samples 
            if torch.numel(s.audio["source"]) > self.cfg.audio.min_sample_size 
            and torch.numel(s.text["source"]) > 2
        ]
        ids = torch.LongTensor([s.index for s in samples])
        if len(samples) == 0:
            return {}
        
        audios = [x.audio["source"] for x in samples]
        sizes = [s.size(0) for s in audios]

        if self.cfg.audio.enable_padding:
            # target_size = min(max(sizes), self.cfg.audio.max_sample_size)
            target_size = max(sizes)
        else:
            target_size = min(min(sizes), self.cfg.audio.max_sample_size)
        
        # frames = _collate_frames(audios, is_audio_input=True)
        collated_sources = audios[0].new_zeros(len(audios), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) 
            if self.cfg.audio.enable_padding else None
        )
        for i, (source, size) in enumerate(zip(audios, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.cfg.audio.enable_padding
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                # collated_sources[i, :size] = source
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        audio_input = {"source": collated_sources, 
                        "src_lengths": torch.tensor(sizes, dtype=torch.long)
                    } # B x max_len
        if self.cfg.audio.enable_padding:
            audio_input["padding_mask"] = padding_mask

        if "precomputed_mask" in samples[0].audio:
            target_size = self._get_mask_indices_dims(target_size)
            if self.cfg.audio.enable_padding:
                padded_tensors = [
                    s.audio["precomputed_mask"] if s.audio["precomputed_mask"].size(1) == target_size
                    else F.pad(s.audio["precomputed_mask"], (0, target_size - s.audio["precomputed_mask"].size(1)))
                    for s in samples
                ]
                collated_mask = torch.cat(padded_tensors, dim=0)
            else:
                collated_mask = torch.cat(
                    [
                        self.crop_to_max_size(s.audio["precomputed_mask"], target_size, dim=1)
                        for s in samples
                    ],
                    dim=0,
                )
            audio_input["precomputed_mask"] = collated_mask
        
        tokens = fairseq_data_utils.collate_tokens(
                [x.text["source"] for x in samples],
                self.pad_idx,
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=False,
            )
        src_txt_lengths = torch.tensor(
                [x.text["source"].size()[0] - 2 for x in samples], dtype=torch.long
            ) # exclude <bos> and <eos>

        padding_mask = fairseq_data_utils.collate_tokens(
                [torch.zeros_like(x.text["source"]).bool() for x in samples],
                self.pad_idx,
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=False,
            )
        text_input = {
            "source": tokens, "padding_mask": padding_mask,
            "src_txt_lengths": src_txt_lengths,
        }

        if "precomputed_mask" in samples[0].text:
            target_size = max([x.text["precomputed_mask"].size(1) for x in samples])
            padded_tensors = [
                s.text["precomputed_mask"] if s.text["precomputed_mask"].size(1) == target_size
                else F.pad(s.text["precomputed_mask"], (0, target_size - s.text["precomputed_mask"].size(1)))
                for s in samples
            ]
            collated_mask = torch.cat(padded_tensors, dim=0)
            text_input["precomputed_mask"] = collated_mask

        target_lengths = torch.tensor(
            [x.text["source"].size(0) for x in samples], dtype=torch.long
        )
        ntokens = sum(x.text["source"].size(0) for x in samples)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .view(-1, 1)
            )

        net_input = {
            "source": {
                "audio": audio_input,  
                "text": text_input,
            },
        }
        out = {
            "id": ids,
            "net_input": net_input,
            "speaker": speaker,
            "text_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }

        return out
    
    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]
        
    def num_tokens(self, index):
        return self.n_frames[index]
    
    def size(self, index):
        return self.n_frames[index], self.text_lens[index]
    
    @property
    def sizes(self):
        return np.array(self.n_frames)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.cfg.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False
    

def create_single_mask(sz, mask_prob, mask_multiple_length, mask_stdev=0.0, rng=None):
    """
    Creates a single random mask for a sequence of size (1, sz) with specified parameters.

    Parameters:
    - sz (int): Length of the sequence (1, sz).
    - mask_prob (float): Probability of masking elements in the sequence.
    - mask_multiple_length (int): Base length for consecutive masked elements.
    - mask_stdev (float): Standard deviation for varying mask lengths (default 0).
    - rng (np.random.Generator): Random number generator (default np.random.default_rng).

    Returns:
    - torch.Tensor: A binary mask tensor of shape (1, sz).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize mask array as False
    mask = np.full(sz, False)

    # Calculate the number of tokens to mask
    num_mask = int(mask_prob * sz / float(mask_multiple_length) + rng.random())
    mask_idc = rng.choice(sz, num_mask, replace=False)

    # Adjust mask lengths based on standard deviation
    if mask_stdev > 0.0:
        lengths = rng.normal(mask_multiple_length, mask_stdev, size=num_mask)
        lengths = [max(0, int(round(x))) for x in lengths]
        mask_idc = np.asarray(
            [mask_idc[j] + offset
             for j in range(len(mask_idc))
             for offset in range(lengths[j])],
            dtype=np.int64,
        )
    else:
        mask_idc = np.concatenate(
            [mask_idc + i for i in range(mask_multiple_length)]
        )
    mask_idc = mask_idc[(mask_idc != 0) & (mask_idc != sz - 1)] # do not mask first and last tokens (<s> and </s>)
    # Ensure indices are within bounds
    mask_idc = mask_idc[mask_idc < len(mask)]
    mask[mask_idc] = True

    return mask