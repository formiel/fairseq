# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
import numpy as np

from fairseq.data import (
    Dictionary, 
    data_utils, 
    TokenBlockDataset,
    PrependTokenDataset,
    MaskTokensDataset,
    SortDataset,
    RightPadDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    IdDataset,
)
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
)
from fairseq.data.audio.speech_to_text_joint_masked_dataset import (
    SpeechToTextJointMaskedDatasetCreator,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task
from examples.speech_text_joint_to_text.tasks.speech_text_joint import SpeechTextJointToTextTask


logger = logging.getLogger(__name__)


@register_task("siamese_speech_text_to_text")
class SiameseSpeechTextToTextTask(SpeechTextJointToTextTask):
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
            "--speech-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq  > 1",
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens per sample for monolingual dataset",
        )
        parser.add_argument(
            "--load-speech-only",
            action="store_true",
            help="load speech data only",
        )
        parser.add_argument(
            "--monolingual-text-data",
            default="",
            help="path to monolingual text data directory",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            default=512,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--max-positions-text",
            type=int,
            metavar="N",
            default=400,
            help="maximum tokens for per encoder text input ",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            metavar="N",
            help="",
        )
        parser.add_argument(
            "--mask-multiple-length",
            default=3,
            type=float,
            metavar="N",
            help="",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        self.speech_only = getattr(args, "load_speech_only", False)
        self.mask_idx = None
        self.mask_sym = "<mask>"
        if self.src_dict is not None:
            assert self.tgt_dict.pad() == self.src_dict.pad()
            assert self.tgt_dict.eos() == self.src_dict.eos()
            if self.args.monolingual_text_data != "":
                self.mask_idx = self.src_dict.add_symbol(self.mask_sym)

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

    def load_monolingual_dataset(self, split, epoch=1, combine=False, **kwargs):
        dataset = None
        split = "train" if split.startswith("train") else "valid"
        path = os.path.join(self.args.monolingual_text_data, split)
        dataset = data_utils.load_indexed_dataset(
                path,
                self.source_dictionary,
                combine=True,
                dataset_impl=None,
            )
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            "",
            None,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="complete",
        )
        logger.info("loaded {} blocks".format(len(dataset)))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=0.1,
            random_token_prob=0.1,
            freq_weighted_replacement=False,
            mask_whole_words=None,
            mask_multiple_length=self.args.mask_multiple_length,
            mask_stdev=0.0,
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_dataset))

        text_dataset = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "masked_src_txt_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "masked_src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "masked_target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )
        return text_dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        s2t_dataset = SpeechToTextJointMaskedDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            mask_sym=self.mask_sym,
            mask_prob=self.args.mask_prob,
            mask_multiple_length=self.args.mask_multiple_length,
            speech_only=self.speech_only, 
        )
        text_dataset = None
        if self.args.monolingual_text_data != "" and is_train_split:
            text_dataset = self.load_monolingual_dataset(split, epoch=epoch)
        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sup_speech",
                    s2t_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_positions_text, self.args.max_target_positions),
                    self.args.max_tokens_text if self.args.max_tokens_text is not None else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            s2t_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = s2t_dataset

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.src_dict