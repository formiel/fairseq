import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import io
import soundfile as sf
import time
import torch
import torch.nn.functional as F

from omegaconf import MISSING

from fairseq.data import encoders
from fairseq.data import FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.speech_to_speech_dataset import _collate_frames
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.dataclass import FairseqDataclass

try:
    from transformers import PreTrainedTokenizerFast
except ImportError:
    raise ImportError("The 'transformers' library is not installed. Please install it by running 'pip install transformers'.")


logger = logging.getLogger(__name__)

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"


@dataclass
class AlignedSpeechTextDatasetItem(object):
    index: int
    audio: torch.Tensor
    text: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None


@dataclass
class TextDataConfig(FairseqDataclass):
    prepend_bos: bool = field(
        default=True,
        metadata={"help": "prepend beginning-of-sentence token"},
    )
    append_eos: bool = field(
        default=True,
        metadata={"help": "append end-of-sentence token (</s> to the end of sentence)"},
    )


@dataclass
class AudioDataConfig(FairseqDataclass):
    normalize: bool = field(
        default=True,
        metadata={"help": "if set, normalizes audio waveform to have 0 mean and unit variance"},
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )


@dataclass
class AlignedSpeechTextConfig(FairseqDataclass):
    audio: Optional[AudioDataConfig] = None
    text: Optional[TextDataConfig] = None
    shuffle: bool = field(
        default=False,
        metadata={"help": "shuffle data"},
    )


class AlignedSpeechTextDataset(FairseqDataset):

    def __init__(
        self,
        split: str,
        cfg: AlignedSpeechTextConfig,
        tokenizer: str,
        audios: List[str],
        texts: List[str],
        n_frames: List[int],
        ids: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        speaker_to_id=None,
    ):
        self.split = split
        self.cfg = cfg
        self.speaker_to_id = speaker_to_id

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)
        self.bos_idx = self.tokenizer.encode(BOS_TOKEN)[0]
        self.eos_idx = self.tokenizer.encode(EOS_TOKEN)[0]
        self.pad_idx = self.tokenizer.encode(PAD_TOKEN)[0]
        self.unk_idx = self.tokenizer.encode(UNK_TOKEN)[0]

        self.ids, self.speakers, self.n_frames, self.audios, self.texts = (
            ids, speakers, n_frames, audios, texts
        )
        self.n_samples = len(self.audios)

        self.text_lens = self.get_text_lens_and_check_oov()

        logger.info(self.__repr__())

    def get_text_lens_and_check_oov(self):
        if self.texts is None:
            return [0 for _ in range(self.n_samples)]
        text_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_text_tokens(i)
            oov_tokens = [t for t in tokenized if t == self.unk_idx]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            text_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return text_lens
    
    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"vocab size={len(self.tokenizer)}, "
            f"prepend_bos={self.cfg.text.prepend_bos}, "
            f"append_eos={self.cfg.text.append_eos}, "
            f"normalized audio={self.cfg.audio.normalize})"
        )
    
    def __len__(self):
        return self.n_samples

    @classmethod
    def _load_data_from_csv(
        cls, data_root: str, split: str, tokenizer_root: str, 
        cfg: AlignedSpeechTextConfig,
        speaker_to_id=None
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
        n_frames = [s["n_frames"] for s in samples]
        audios = [s["audio"] for s in samples]
        texts = [s["src_text"] for s in samples]
        return cls(
            split=split, cfg=cfg, tokenizer=tokenizer_root, audios=audios, texts=texts, n_frames=n_frames, ids=ids, speakers=speakers, speaker_to_id=speaker_to_id
        )
    
    def get_text_tokens(self, index: int):
        text = self.texts[index]
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens).long()
        assert torch.max(tokens) <= len(self.tokenizer) - 1
        if self.cfg.text.prepend_bos:
            tokens = torch.cat(
                (torch.tensor([self.bos_idx], dtype=torch.int64), tokens),
                dim=0
            )
        if self.cfg.text.append_eos:
            tokens = torch.cat(
                (tokens, torch.tensor([self.eos_idx], dtype=torch.int64)),
                dim=0
            )
        return tokens

    def get_audio_frames(self, index: int):
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
        feats = self.postprocess(feats, curr_sample_rate)

        return feats

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
        audio = self.get_audio_frames(index)
        text = self.get_text_tokens(index)
        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return AlignedSpeechTextDatasetItem(
            index=index,
            audio=audio,
            text=text,
            speaker_id=speaker_id,
        )
    
    def collater(self, samples: List[AlignedSpeechTextDatasetItem], return_order: bool = False) -> Dict:
        if len(samples) == 0:
            return {}
        
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        audios = [x.audio for x in samples]
        frames = _collate_frames(audios, is_audio_input=True)

        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in audios], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        tokens = fairseq_data_utils.collate_tokens(
                [x.text for x in samples],
                self.pad_idx,
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=False,
            )
        tokens = tokens.index_select(0, order)
        target_lengths = torch.tensor(
            [x.text.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)
        ntokens = sum(x.text.size(0) for x in samples)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        net_input = {
            "audio": frames,
            "text": tokens,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "text_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out
        
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