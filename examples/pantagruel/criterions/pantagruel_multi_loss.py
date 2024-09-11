import logging
import math
from dataclasses import dataclass, field
from typing import List

import torch

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from fairseq.logging.meters import safe_round


@dataclass
class PantagruelMultiConfig(FairseqDataclass):
    d2v_weight: float = field(
        default=1.0,
        metadata={"help": "weight of data2vec loss"},
    )
    ctr_weight: float = field(
        default=1.0,
        metadata={"help": "weight of contrastive loss"},
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
        d2v_weight,
        ctr_weight,
        log_keys=None,
    ):
        super().__init__(task)
        self.d2v_weight = d2v_weight
        self.ctr_weight = ctr_weight
        self.log_keys = log_keys
        assert self.d2v_weight >=0 or self.ctr_weight >=0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        
        scaled_losses = {}

        d2v_loss = net_output["losses"]
        for lk, p in d2v_loss.items():
            scaled_losses[lk] = self.d2v_weight * p.float().sum()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "_world_size": 1,
        }

        for lk in self.log_keys:
            if lk in net_output and net_output[lk] is not None:
                if not torch.is_tensor(net_output[lk]) or net_output[lk].numel() == 1:
                    logging_output[lk] = float(net_output[lk])
                elif lk.startswith("_"):
                    logging_output[lk] = net_output[lk]
                else:
                    for i, v in enumerate(net_output[lk]):
                        logging_output[f"{lk}_{i}"] = float(v)

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"loss_{lk}"] = l.item()

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
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

        for k in logging_outputs[0]:
            if k not in builtin_keys and not k.startswith("_"):
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        total = sum(log.get("count", 0) for log in logging_outputs)

        if total > 0:
            metrics.log_scalar("_correct", correct)
            metrics.log_scalar("_total", total)

            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        True