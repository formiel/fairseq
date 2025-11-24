# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import II, MISSING

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class MaskedLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )
    include_index: bool = field(
        default=True,
        metadata={"help": "include index in model input. this is used for data2vec"},
    )
    skip_masking: bool = field(
        default=False,
        metadata={"help": "skip masking at dataset"},
    )
    # subsample_train: float = field(
    #     default=1,
    #     metadata={"help": "shorten training set for debugging"},
    # )
    d2v2_multi: bool = field(
        default=False,
        metadata={"help": "prepare dataset for data2vec_multi"},
    )
    mask_idx_in_vocab: bool = field(
        default=False,
        metadata={"help": "mask index is included in vocab already."}
    )
    pantagruel_multi: bool = field(
        default=False,
        metadata={"help": "prepare dataset for pantagruel training with multimodal encoder"},
    )
    bos_token_prepended: bool = field(
        default=False,
        metadata={"help": "whether the bos token is prepended in the dataset"},
    )
    panta_text_mlm_org_impl: bool = field(
        default=False,
        metadata={"help": "prepare dataset for training textual pantagruel with mlm head"},
    )
    document_sep_len: int = field(
        default=1,
        metadata={"help": "document separator size (required for 'complete_doc' break mode)"},
    )


@register_task("masked_lm", dataclass=MaskedLMConfig)
class MaskedLMTask(FairseqTask):

    cfg: MaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MaskedLMConfig, dictionary=None):
        super().__init__(cfg)
        self.dictionary = dictionary or self.load_dict(cfg)

        # add mask token
        if not cfg.mask_idx_in_vocab:
            self.mask_idx = self.dictionary.add_symbol("<mask>")
        else:
            self.mask_idx = self.dictionary.index("<mask>")
        if self.dictionary:
            logger.info(f'bos:{self.dictionary.index(self.dictionary.bos_word)},\t'
                        f'pad:{self.dictionary.index(self.dictionary.pad_word)},\t'
                        f'eos:{self.dictionary.index(self.dictionary.eos_word)},\t'
                        f'unk:{self.dictionary.index(self.dictionary.unk_word)},\t'
                        f'mask_idx:{self.mask_idx}')
            logger.info("[final] dictionary: {} types".format(len(self.dictionary)))

    @classmethod
    def setup_task(cls, cfg: MaskedLMConfig, **kwargs):
        dictionary = cls.load_dict(cfg)
        return cls(cfg, dictionary)

    @classmethod
    def load_dict(cls, cfg: MaskedLMConfig):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return dictionary

    def _load_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        tokens_per_sample = (
                self.cfg.tokens_per_sample - 1 if not self.cfg.bos_token_prepended 
                else self.cfg.tokens_per_sample
            )
        logger.info(f"tokens_per_sample={tokens_per_sample}")
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            tokens_per_sample,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
            document_sep_len=self.cfg.document_sep_len,
        )
        for i in range(3):
            logger.info(f"First 3 samples in the {split} dataset after TokenBlockDataset:")
            logger.info(f"Sample {i}: {dataset[i]}")
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        if not self.cfg.bos_token_prepended:
            return PrependTokenDataset(dataset, self.source_dictionary.bos())
        return PrependTokenDataset(dataset, None)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_dataset_split(split, epoch, combine)

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            skip_masking=self.cfg.skip_masking,
        )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        if self.cfg.d2v2_multi:
            if not getattr(self.cfg, "panta_text_mlm_org_impl", False):
                dataset = self._d2v2_multi_dataset(
                    src_dataset, 
                    target_dataset=(
                        target_dataset if getattr(self.cfg, "pantagruel_multi", False) else None
                    ),
                )
            else:
                src_dataset_mlm, tgt_dataset_mlm = MaskTokensDataset.apply_mask(
                    dataset,
                    self.source_dictionary,
                    pad_idx=self.source_dictionary.pad(),
                    mask_idx=self.mask_idx,
                    seed=self.cfg.seed,
                    mask_prob=self.cfg.mask_prob,
                    leave_unmasked_prob=self.cfg.leave_unmasked_prob,
                    random_token_prob=self.cfg.random_token_prob,
                    freq_weighted_replacement=self.cfg.freq_weighted_replacement,
                    mask_whole_words=mask_whole_words,
                    mask_multiple_length=self.cfg.mask_multiple_length,
                    mask_stdev=self.cfg.mask_stdev,
                    skip_masking=False, # apply masking at source for mlm head
                )
                dataset = self._pantagruel_mlm_org_impl_dataset(
                    src_dataset_d2v=src_dataset,
                    src_dataset_mlm=src_dataset_mlm,
                    target_dataset_mlm=RightPadDataset(
                        tgt_dataset_mlm,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                )

        else:
            dataset = self._regular_dataset(src_dataset, target_dataset)

        self.datasets[split] = SortDataset(
            dataset, sort_order=[shuffle, src_dataset.sizes]
        )

    def _regular_dataset(self, src_dataset, target_dataset):
        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
        }
        if self.cfg.include_target_tokens:
            input_dict["target_tokens"] = target_dataset
        if self.cfg.include_index:
            input_dict["src_id"] = IdDataset()

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "target": target_dataset,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )
        return dataset

    def _d2v2_multi_dataset(self, src_dataset, target_dataset=None):
        input_dict = {
            "source": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "id": IdDataset(),
            "padding_mask": RightPaddingMaskDataset(src_dataset),
        }
        if getattr(self.cfg, "pantagruel_multi", False) and target_dataset is not None:
            logger.info("Using target dataset for unimodal pantagruel")
            input_dict["target_mlm"] = target_dataset

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )
        return dataset

    def _pantagruel_mlm_org_impl_dataset(
        self, src_dataset_d2v, src_dataset_mlm, target_dataset_mlm,
    ):
        input_dict = {
            "source": RightPadDataset(
                src_dataset_d2v,
                pad_idx=self.source_dictionary.pad(),
            ),
            "id": IdDataset(),
            "padding_mask": RightPaddingMaskDataset(src_dataset_d2v),
            # for mlm head
            "source_mlm": RightPadDataset(
                src_dataset_mlm,
                pad_idx=self.source_dictionary.pad(),
            ),
            "target_mlm": target_dataset_mlm,
        }
        dataset = NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": input_dict,
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset_d2v, reduce=True),
                },
                sizes=[src_dataset_d2v.sizes],
                )
        return dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        tokens_per_sample = (
                self.cfg.tokens_per_sample - 1 if not self.cfg.bos_token_prepended 
                else self.cfg.tokens_per_sample
            )
        logger.info(f"tokens_per_sample={tokens_per_sample}")
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                tokens_per_sample,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        if not self.cfg.bos_token_prepended:
            src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def begin_epoch(self, epoch, model):
        model.set_epoch(epoch)

    def max_positions(self):
        return self.cfg.tokens_per_sample
