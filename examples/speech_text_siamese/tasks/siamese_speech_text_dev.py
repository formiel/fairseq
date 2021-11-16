# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
from argparse import Namespace


from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDatasetCreator,
)
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask


logger = logging.getLogger(__name__)


@register_task("siamese_speech_text_to_text_dev")
class SiameseSpeechTextToTextTaskDev(SpeechToTextTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        # parser.add_argument(
        #     "--max-source-positions",
        #     default=6000,
        #     type=int,
        #     metavar="N",
        #     help="max number of tokens in the source sequence",
        # )
        # parser.add_argument(
        #     "--max-target-positions",
        #     default=1024,
        #     type=int,
        #     metavar="N",
        #     help="max number of tokens in the target sequence",
        # )
        parser.add_argument(
            "--load-speech-only",
            action="store_true",
            help="load speech data only",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        self.speech_only = getattr(args, "load_speech_only", False)
        logging.info(f"self.speech_only: {self.speech_only}")
        if self.src_dict is not None:
            assert self.tgt_dict.pad() == self.src_dict.pad()
            assert self.tgt_dict.eos() == self.src_dict.eos()

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if not os.path.isfile(src_dict_path):
            logging.warning("Dict not found: {}".format(src_dict_path))
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError("Dict not found: {}".format(tgt_dict_path))
        src_dict = Dictionary.load(src_dict_path.as_posix()) if src_dict_path.exists() else None
        if src_dict is not None:
            logger.info(
                f"source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}")
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")

        return cls(args, src_dict, tgt_dict)

    def build_src_tokenizer(self, args):
        logger.info(f"src-pre-tokenizer: {self.data_cfg.src_pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.src_pre_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src-tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.speech_only = True
        src_dict = None if self.speech_only else self.src_dict
        logging.info(f"***** src_dict: {src_dict}")
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        self.datasets[split] = SpeechToTextJointDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=None if self.speech_only else self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None if self.speech_only else self.src_dict