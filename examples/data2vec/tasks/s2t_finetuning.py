# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch
import json

from pathlib import Path

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from fairseq.data import AddTargetDataset, Dictionary, encoders
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.data.audio.data_cfg import get_config_from_yaml

from fairseq.tasks import register_task
from fairseq import utils
from fairseq.logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary, s2t_yaml_config):
        self.dictionary = dictionary
        s2t_yaml_config = Path(s2t_yaml_config)
        s2t_yaml_config = get_config_from_yaml(s2t_yaml_config)
        bpe_args = {
            "bpe": s2t_yaml_config["bpe_tokenizer"]["bpe"],
            "sentencepiece_model": s2t_yaml_config["bpe_tokenizer"]["sentencepiece_model"]
        }
        self.bpe_tokenizer = encoders.build_bpe(Namespace(**bpe_args))

    def __call__(self, label):
        tokenized_txt = self.bpe_tokenizer.encode(label)
        target = self.dictionary.encode_line(
                tokenized_txt, add_if_not_exist=False, append_eos=False
            )
        return target.long()


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class Data2vec2STFinetuningConfig(AudioPretrainingConfig):
    s2t_yaml_config: Optional[Any] = field(
        default=None,
        metadata={
            "help": "config file for speech-to-text dataset"
        }
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None,
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
            "required if using --eval-bleu; use 'space' to disable "
            "detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}", metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": "generation args for BLUE scoring, e.g., "
            '\'{"beam": 4, "lenpen": 0.6}\''
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("data2vec2_st_finetuning", dataclass=Data2vec2STFinetuningConfig)
class Data2vec2STFinetuningTask(AudioPretrainingTask):
    """ """

    cfg: Data2vec2STFinetuningConfig

    def __init__(
        self,
        cfg: Data2vec2STFinetuningConfig,
    ):
        super().__init__(cfg)

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    def load_target_dictionary(self):
        # Load the YAML configuration
        s2t_yaml_config = get_config_from_yaml(Path(self.cfg.s2t_yaml_config))
        logger.info(f"Loaded s2t_yaml_config: {s2t_yaml_config}")

        dict_path = (
            Path(
                s2t_yaml_config["bpe_tokenizer"]["sentencepiece_model"]
            ).parent / s2t_yaml_config["vocab_filename"]
        )
        tgt_dict = Dictionary.load(dict_path.as_posix())

        return tgt_dict

    def load_dataset(
        self, split: str, task_cfg: Data2vec2STFinetuningConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(
            self.target_dictionary, self.cfg.s2t_yaml_config
        )

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=True,
            text_compression_level=text_compression_level,
        )

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.eval_bleu:
            metrics = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = metrics.sys_len
            logging_output["_bleu_ref_len"] = metrics.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(metrics.counts) == 4
            for i in range(4):
                logging_output[f"_bleu_counts_{i}"] = metrics.counts[i]
                logging_output[f"_bleu_totals_{i}"] = metrics.totals[i]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_bleu:
            assert self.cfg.eval_bleu_detok is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_bpe(
                Namespace(**detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args = Namespace(**gen_args)
            self.sequence_generator = self.build_generator([model], gen_args)

        return model

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, is_ref):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if is_ref else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"], is_ref=False))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                    is_ref=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("H-{} {}".format(sample["id"][0], hyps[0]))
            logger.info("T-{} {}".format(sample["id"][0], refs[0]))

        eval_tokenization = "none" if self.cfg.eval_tokenized_bleu else "13a"
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=eval_tokenization)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_bleu:
            len_keys = ["_bleu_sys_len", "_bleu_ref_len"]
            count_keys = [f"_bleu_counts_{i}" for i in range(4)]
            total_keys = [f"_bleu_totals_{i}" for i in range(4)]
            for k in len_keys + count_keys + total_keys:
                metrics.log_scalar(k, sum(log.get(k, 0) for log in logging_outputs))

            def compute_bleu(meters):
                import inspect

                try:
                    from sacrebleu.metrics import BLEU

                    comp_bleu = BLEU.compute_bleu
                except ImportError:
                    # compatibility API for sacrebleu 1.x
                    import sacrebleu

                    comp_bleu = sacrebleu.compute_bleu

                fn_sig = inspect.getfullargspec(comp_bleu)[0]
                if "smooth_method" in fn_sig:
                    smooth = {"smooth_method": "exp"}
                else:
                    smooth = {"smooth": "exp"}
                bleu = comp_bleu(
                    correct=[meters[k].sum for k in count_keys],
                    total=[meters[k].sum for k in total_keys],
                    sys_len=meters["_bleu_sys_len"].sum,
                    ref_len=meters["_bleu_ref_len"].sum,
                    **smooth,
                )
                return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)
