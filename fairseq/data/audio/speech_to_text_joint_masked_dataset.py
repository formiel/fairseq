# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple

import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDatasetCreator,
    _collate_frames,
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDataset,
    SpeechToTextJointDatasetItem
)

logger = logging.getLogger(__name__)


class SpeechToTextJointMaskedDatasetItem(NamedTuple):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    masked_src_txt_tokens: Optional[torch.Tensor] = None
    masked_target: Optional[torch.Tensor] = None


class SpeechToTextJointMaskedDataset(SpeechToTextJointDataset):
    def __init__(
        self, 
        split: str, 
        is_train_split: bool, 
        cfg: S2TJointDataConfig, 
        audio_paths: List[str], 
        n_frames: List[int], 
        src_texts: Optional[List[str]] = None, 
        tgt_texts: Optional[List[str]] = None, 
        speakers: Optional[List[str]] = None, 
        src_langs: Optional[List[str]] = None, 
        tgt_langs: Optional[List[str]] = None, 
        ids: Optional[List[str]] = None, 
        tgt_dict: Optional[Dictionary] = None, 
        src_dict: Optional[Dictionary] = None, 
        pre_tokenizer=None, 
        bpe_tokenizer=None, 
        src_pre_tokenizer=None, 
        src_bpe_tokenizer=None,
        mask_sym="<mask>",
        mask_prob=0.15,
        mask_multiple_length=3,
        speech_only=False,
        ):
        super().__init__(
            split, 
            is_train_split, 
            cfg, 
            audio_paths, 
            n_frames, 
            src_texts=src_texts, 
            tgt_texts=tgt_texts, 
            speakers=speakers, 
            src_langs=src_langs, 
            tgt_langs=tgt_langs, 
            ids=ids, 
            tgt_dict=tgt_dict, 
            src_dict=src_dict, 
            pre_tokenizer=pre_tokenizer, 
            bpe_tokenizer=bpe_tokenizer, 
            src_pre_tokenizer=src_pre_tokenizer, 
            src_bpe_tokenizer=src_bpe_tokenizer
        )
        self.mask_sym = mask_sym
        self.mask_prob = mask_prob
        self.mask_multiple_length = mask_multiple_length
        self.speech_only = speech_only

    def __getitem__(self, index: int) -> SpeechToTextJointMaskedDatasetItem:
        s2t_joint_dataset_item = super().__getitem__(index)        
        if self.speech_only:
            s2t_joint_dataset_item = SpeechToTextJointDatasetItem(
                index=index,
                source=s2t_joint_dataset_item.source,
                target=s2t_joint_dataset_item.target,
                src_txt_tokens=None,
                tgt_lang_tag=s2t_joint_dataset_item.tgt_lang_tag,
            )
            
        masked_src_tokens, masked_target = None, None
        if s2t_joint_dataset_item.src_txt_tokens is not None:
            masked_src_tokens, masked_target = self.apply_mask(
                s2t_joint_dataset_item.src_txt_tokens,
                mask_sym=self.mask_sym,
                mask_prob=self.mask_prob,
                mask_multiple_length=self.mask_multiple_length
            )
        return SpeechToTextJointMaskedDatasetItem(
            index=index,
            source=s2t_joint_dataset_item.source,
            target=s2t_joint_dataset_item.target,
            src_txt_tokens=s2t_joint_dataset_item.src_txt_tokens,
            tgt_lang_tag=s2t_joint_dataset_item.tgt_lang_tag,
            masked_src_txt_tokens=masked_src_tokens,
            masked_target=masked_target
        )

    def apply_mask(
        self,
        item,
        mask_sym,
        mask_prob,
        mask_multiple_length,
    ):
        sz = len(item)
        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * sz / float(mask_multiple_length)
            + np.random.rand()
        )
        # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
        mask_idc = np.random.choice(sz, num_mask, replace=False)
        mask_idc = np.concatenate(
            [mask_idc + i for i in range(mask_multiple_length)]
        )
        mask_idc = mask_idc[mask_idc < len(mask)]
        mask[mask_idc] = True
        
        # masked_src_txt_tokens
        new_item = np.copy(item)
        new_item[mask] = self.src_dict.index(mask_sym)
        new_item = np.concatenate(([self.src_dict.bos()], new_item)) # prepend bos token

        # masked_target
        masked_target = np.full(len(mask), self.src_dict.pad())
        masked_target[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
        masked_target = np.concatenate(([self.src_dict.bos()], masked_target))

        return torch.from_numpy(new_item), torch.from_numpy(masked_target)

    def collater(self, samples: List[SpeechToTextJointMaskedDatasetItem], return_order=False) -> Dict:
        # s2t dataset
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size()[0] for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size()[0] for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size()[0] for x in samples)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
        }
        s2t_out = {
            "id": indices,
            "net_input": net_input,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            s2t_out["order"] = order

        # s2t joint dataset
        net_input = s2t_out["net_input"]

        if self.src_texts is not None and not self.speech_only:
            src_txt_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_tokens for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_tokens = src_txt_tokens.index_select(0, order)
            src_txt_lengths = torch.tensor(
                [x.src_txt_tokens.size()[0] for x in samples], dtype=torch.long
            ).index_select(0, order)
            net_input["src_txt_tokens"] = src_txt_tokens
            net_input["src_txt_lengths"] = src_txt_lengths

        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag

        out = {
            "id": s2t_out["id"],
            "net_input": net_input,
            "target": s2t_out["target"],
            "target_lengths": s2t_out["target_lengths"],
            "ntokens": s2t_out["ntokens"],
            "nsentences": len(samples),
        }
        
        # s2t joint masked dataset
        masked_src_txt_tokens, masked_target = None, None
        if self.src_texts is not None and not self.speech_only:
            masked_src_txt_tokens = fairseq_data_utils.collate_tokens(
                    [x.masked_src_txt_tokens for x in samples],
                    self.src_dict.pad(),
                    self.src_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                )
            masked_target = fairseq_data_utils.collate_tokens(
                    [x.masked_target for x in samples],
                    self.src_dict.pad(),
                    self.src_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                )
            masked_src_txt_tokens = masked_src_txt_tokens.index_select(0, order)
        out["net_input"]["masked_src_txt_tokens"] = masked_src_txt_tokens
        out["masked_target"] = masked_target
        return out


class SpeechToTextJointMaskedDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TJointDataConfig,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        mask_sym,
        mask_prob,
        mask_multiple_length,
        speech_only,
    ) -> SpeechToTextJointMaskedDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextJointMaskedDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            src_dict=src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            mask_sym=mask_sym,
            mask_prob=mask_prob,
            mask_multiple_length=mask_multiple_length,
            speech_only=speech_only,
        )

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TJointDataConfig,
        split: str,
        tgt_dict,
        src_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        mask_sym,
        mask_prob,
        mask_multiple_length,
        speech_only,
    ) -> SpeechToTextJointMaskedDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_pre_tokenizer,
            src_bpe_tokenizer,
            mask_sym,
            mask_prob,
            mask_multiple_length,
            speech_only,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TJointDataConfig,
        splits: str,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        mask_sym: str,
        mask_prob: float,
        mask_multiple_length: int,
        speech_only: bool,
    ) -> SpeechToTextJointMaskedDataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                tgt_dict,
                src_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                src_pre_tokenizer,
                src_bpe_tokenizer,
                mask_sym,
                mask_prob,
                mask_multiple_length,
                speech_only,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            logging.info(f"size_ratios: {size_ratios}")
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
