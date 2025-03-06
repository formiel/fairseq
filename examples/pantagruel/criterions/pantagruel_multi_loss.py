import os
import logging
import math
from dataclasses import dataclass, field
from typing import List

import editdistance
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import utils
from fairseq.data.data_utils import post_process
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from fairseq.logging.meters import safe_round
from examples.pantagruel.data.utils import get_random_crops, create_negative_pairs

try:
    from geomloss import SamplesLoss
except ImportError:
    raise ImportError("The 'geomloss' library is not installed. Please install it by running 'pip install geomloss'.")

import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


@dataclass
class PantagruelMultiConfig(FairseqDataclass):
    d2v_weight: float = field(
        default=1.0,
        metadata={"help": "weight of data2vec loss"},
    )
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of ctc loss"},
    )
    ot_weight: float = field(
        default=0.0,
        metadata={"help": "weight of OT loss"},
    )
    log_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "additional output keys to log"},
    )


@register_criterion("pantagruel_multi_loss", dataclass=PantagruelMultiConfig)
class PantagruelMultiCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        d2v_weight=1.0,
        ctc_weight=0.0,
        ot_weight=0.0,
        log_keys=None,
    ):
        super().__init__(task)
        
        self.log_keys = log_keys
        self.task = task

        self.ctc_weight = ctc_weight
        self.pad_idx = task.source_dictionary.index(task.source_dictionary.pad_word)
        self.eos_idx = task.source_dictionary.index(task.source_dictionary.eos_word)
        self.blank_idx = task.source_dictionary.index(task.source_dictionary.bos_word)
        logger.info(
            f"blank_idx={self.blank_idx}, pad_idx={self.pad_idx}, eos_idx={self.eos_idx}"
        )

        self.d2v_weight = d2v_weight
        self.ot_weight = ot_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample["net_input"]
        net_output = model(**net_input)
        
        scaled_losses = {}
        for lk, p in net_output["losses"].items():
            scaled_losses[lk] = self.d2v_weight * p.float().sum()

        loss = sum(scaled_losses.values())
        if reduce and loss.numel() > 1:
            loss = loss.sum()

        sample_size = sum(net_output["sample_size"].values())
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "_world_size": 1,
        }

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"loss_d2v_{lk}"] = l.item()
        for k, v in net_output["sample_size"].items():
            logging_output[f"sample_size_{k}"] = v

        self.log_keys = [
            l for l in net_output if any([l.startswith(lk) for lk in self.log_keys])
        ]
        for lk in self.log_keys:
            if net_output[lk] is not None:
                if not torch.is_tensor(net_output[lk]) or net_output[lk].numel() == 1:
                    logging_output[lk] = float(net_output[lk])
                elif lk.startswith("_"):
                    logging_output[lk] = net_output[lk]
                else:
                    for i, v in enumerate(net_output[lk]):
                        logging_output[f"{lk}_{i}"] = float(v)

        ctc_loss, lprobs = None, None
        if self.ctc_weight > 0 and net_output["ctc_out"]:
            ctc_loss, lprobs, input_lengths = self.compute_ctc_loss(
                net_output["ctc_out"], net_input
            )
            logging_output["loss_ctc"] = self.ctc_weight * ctc_loss

            if not net_output["ctc_out"]["is_frozen"]:
                loss = loss + self.ctc_weight * ctc_loss
                if not model.training and model.num_updates % 200000 == 0: 
                    _ctc_logging = self.compute_wer(
                        lprobs, net_output["ctc_out"], net_input, input_lengths
                    )
                    for k, v in _ctc_logging.items():
                        logging_output[k] = v

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        if self.ot_weight > 0 and net_output["dual_encoders_out"]:
            if not net_output["dual_encoders_out"]["is_frozen"]:
                ot_loss = self.compute_ot_loss(net_output["dual_encoders_out"])
                loss = loss + self.ot_weight * ot_loss
                logging_output["loss_ot"] = self.ot_weight * ot_loss

        logging_output["loss"] = loss.data # update loss

        return loss, sample_size, logging_output

    def compute_ot_loss(self, dual_encoders_out):
        #TODO: OT with positional encoding!
        ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=1.0)

        audio_enc = dual_encoders_out["audio"] # BxTxD
        text_enc = dual_encoders_out["text"]
        text_enc = text_enc[:, 1:-1] # remove bos and eos token
        loss = ot_loss(
                audio_enc.float().contiguous(),
                text_enc.float().contiguous()
            ).sum()
        return loss

    def compute_ctc_loss(self, ctc_out, net_input):
        lprobs = utils.log_softmax(ctc_out["x"], dim=-1).contiguous()  # (T, B, C) from the encoder
        src_text = net_input["source"]["text"]["source"]
        encoder_padding_mask = ctc_out["padding_mask"]
        if isinstance(encoder_padding_mask, torch.Tensor) and encoder_padding_mask.any():
            if lprobs.size(0) > encoder_padding_mask.size(1):
                encoder_padding_mask = F.pad(
                    encoder_padding_mask, (1, 0), value=False
                )
            non_padding_mask = ~encoder_padding_mask
            offset = 0
            input_lengths = non_padding_mask.long().sum(-1)
            if torch.max(input_lengths) < lprobs.size(0):
                offset = 1
            input_lengths = input_lengths + offset
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )

        pad_mask = (src_text != self.pad_idx) & (src_text != self.eos_idx) & (src_text != self.blank_idx)
        targets_flat = src_text.masked_select(pad_mask)
        target_lengths = net_input["source"]["text"]["src_txt_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )

        # ntokens = target_lengths.sum().item()
        # sample_size_ctc = src_text.size(0)

        return ctc_loss, lprobs, input_lengths

    def compute_wer(self, lprobs, ctc_out, net_input, input_lengths):
        src_text = net_input["source"]["text"]["source"]
        target_lengths = net_input["source"]["text"]["src_txt_lengths"]

        _logging_output = {}
        with torch.no_grad():
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu() # BxTxC
            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            wv_errs = 0
            for lp, t, inp_l in zip(lprobs_t, src_text, input_lengths):
                lp = lp[:inp_l].unsqueeze(0)
                decoded = None
                p = (t != self.task.source_dictionary.pad()) & (
                        t != self.task.source_dictionary.eos()
                    )
                targ = t[p]
                targ_units = self.task.source_dictionary.string(targ)
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

                # targ_words = post_process(targ_units, self.post_process).split()
                targ_words = self.task.source_tokenizer.decode(targ_units_arr).split()

                pred_units = self.task.source_dictionary.string(pred_units_arr)
                # pred_words_raw = post_process(pred_units, self.post_process).split()
                pred_words_raw = self.task.source_tokenizer.decode(pred_units_arr).split()
                

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    w_errs += editdistance.eval(pred_words, targ_words)
                    wv_errs += editdistance.eval(pred_words_raw, targ_words)
                else:
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                w_len += len(targ_words)

            # printing out decoding results
            logger.info(f"[TGT]: {' '.join(targ_words)}")
            logger.info(f"[HYP]: {' '.join(pred_words_raw)}")

            _logging_output["wv_errors"] = wv_errs
            _logging_output["w_errors"] = w_errs
            _logging_output["w_total"] = w_len
            _logging_output["c_errors"] = c_err
            _logging_output["c_total"] = c_len
        return _logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_keys = list(set([k for log in logging_outputs for k in log if k.startswith("loss_")]))

        if len(loss_keys) >= 1:
            for lk in loss_keys:
                mod = (
                    "AUDIO" if "ctc" in lk or "audio" in lk 
                    else "AUDIO_TEXT" if "ot" in lk 
                    else "TEXT"
                )
                _loss = utils.item(sum(log.get(lk, 0) for log in logging_outputs))
                _sample_size = (
                    utils.item(
                        sum(
                            log.get(f"sample_size_{mod.lower()}", 0) 
                            for log in logging_outputs
                        )
                    ) if "_" not in mod else utils.item(
                        sum(log.get("nsentences", 0) for log in logging_outputs)
                    )
                )
                metrics.log_scalar(lk, _loss / _sample_size, _sample_size, round=3)
                metrics.log_scalar(f"sample_size_{mod}", _sample_size)

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("sample_size", sample_size)

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "_world_size",
        }

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        # WER
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        True