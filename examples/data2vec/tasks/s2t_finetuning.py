import logging
import os
import torch
import json
from pathlib import Path

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq.data.audio.data_cfg import get_config_from_yaml
from fairseq.data import Dictionary, encoders
from fairseq.dataclass import FairseqDataclass
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator
from fairseq.tasks import register_task, FairseqTask

from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig, SpeechToTextDataset, SpeechToTextDatasetCreator
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechToTextFinetuningConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    config_yaml: Optional[Any] = field(
        default=None,
        metadata={
            "help": "config file for speech-to-text dataset"
        }
    )
    seed: int = field(
        default=1,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )


@register_task("data2vec2_st_finetuning", dataclass=SpeechToTextFinetuningConfig)
class S2TFinetuningTask(FairseqTask):
    cfg: SpeechToTextFinetuningConfig

    def __init__(
        self, cfg: SpeechToTextFinetuningConfig, tgt_dict,
    ):
        super().__init__(cfg)

        self.cfg = cfg
        self.data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: SpeechToTextFinetuningConfig, **kwargs):

        data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        dict_path = Path(cfg.data) / data_cfg.vocab_filename

        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        return cls(cfg, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer()
        bpe_tokenizer = self.build_bpe()

        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            root=self.cfg.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed
        )

    def build_tokenizer(self):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def build_model(self, model_cfg: FairseqDataclass):
        logger.info(f"Building model with cfg: {model_cfg}")
        model = super().build_model(model_cfg)
        return model

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        eos_id = self.tgt_dict.index("</s>")
        assert eos_id != self.tgt_dict.unk()
        extra_gen_cls_kwargs["eos"] = eos_id

        logger.info(f"args: {args}")

        return super().build_generator(
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )