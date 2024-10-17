import os
import logging
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from fairseq.logging.meters import safe_round
from examples.pantagruel.data.utils import get_random_crops


@dataclass
class PantagruelMultiConfig(FairseqDataclass):
    d2v_weight: float = field(
        default=1.0,
        metadata={"help": "weight of data2vec loss"},
    )
    lim_weight: float = field(
        default=0.0,
        metadata={"help": "weight of local info max loss"},
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
        lim_weight=0.0,
        log_keys=None,
    ):
        super().__init__(task)
        
        self.log_keys = log_keys
        self.d2v_weight = d2v_weight
        self.lim_weight = lim_weight
        if self.lim_weight > 0:
            self.projector = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 384),
                nn.ReLU(),
            )

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

        if self.lim_weight > 0:
            lim_weight = self.compute_lim_weight(net_output["local_features"])
            logging_output["loss_lim"] = lim_weight

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        return loss, sample_size, logging_output
    
    def compute_lim_weight(self, local_features, eps=1e-9):
        # local_features: B x T x C
        for mode, feature in local_features.items():
            if feature is not None:
                crops1, crops2 = get_random_crops(feature) # B x crop_size x C
                Y1 = self.projector(crops1).mean(dim=1) # B x D
                Y2 = self.projector(crops2).mean(dim=1) # B x D
                B, _ = Y1.size()
                # Y2_all = (GatherLayer.apply(Y2) if (
                #     dist.is_available() and dist.is_initialized()
                # ) else Y2)
                Y2_all = concat_all_gather(Y2) if dist.is_available() and dist.is_initialized() else Y2
                neg_idx = torch.randint(0, Y2_all.size(0), size=(B,))
                Y_R = Y2_all[neg_idx]
                pos = F.cosine_similarity(Y1, Y2, dim=-1)
                neg = F.cosine_similarity(Y1, Y_R, dim=-1)
                pos = torch.clamp(torch.sigmoid(pos), eps, 1.0 - eps)
                neg = torch.clamp(torch.sigmoid(neg), eps, 1.0 - eps)
                loss = -torch.mean(torch.log(pos)) - torch.mean(torch.log(1 - neg))
                return loss

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

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        # output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        # dist.all_gather(output, x, async_op=False)
        # return torch.cat(output, dim=0)
    
        tensors_gather = [torch.ones_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, x, async_op=False)

        return torch.cat(tensors_gather, dim=0)


    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output