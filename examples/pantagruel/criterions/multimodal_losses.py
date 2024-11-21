import logging
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from omegaconf import II

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import fairseq.distributed.utils as distributed_utils

from fairseq.logging.meters import safe_round
from examples.pantagruel.data.utils import get_random_crops, create_negative_pairs

logger = logging.getLogger(__name__)


@dataclass
class PantagruelMultiConfig(FairseqDataclass):
    d2v_weight: float = II("model.d2v_loss")
    ctr_weight: float = field(
        default=0.0,
        metadata={"help": "weight of MoCo-v3 loss"},
    )
    ctr_temperature: float = field(
        default=0.0,
        metadata={"help": "temperature for MoCo-v3 loss"},
    )
    log_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "additional output keys to log"},
    )
    clone_batch: int = II("model.clone_batch")
    use_all_clones: bool = field(
        default=True,
        metadata={"help": "use all clone to increase number of positive examples"}
    )
    embed_dim: int = II("model.embed_dim")
    start_step_train_aux_loss: int = II("model.start_step_train_aux_loss")


@register_criterion("pantagruel_multimodal_loss", dataclass=PantagruelMultiConfig)
class PantagruelMultiCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        d2v_weight,
        ctr_weight=0.0,
        ctr_temperature=0.0,
        log_keys=None,
        clone_batch=8,
        use_all_clones=True,
        embed_dim=768,
        start_step_train_aux_loss=0,
    ):
        super().__init__(task)
        
        self.log_keys = log_keys
        self.d2v_weight = d2v_weight
        self.ctr_weight = ctr_weight
        self.ctr_temperature = ctr_temperature
        self.clone_batch = clone_batch
        self.use_all_clones = use_all_clones
        self.start_step_train_aux_loss = start_step_train_aux_loss
        logger.info(f"self.embed_dim={embed_dim}, self.start_step_train_aux_loss={start_step_train_aux_loss}")
        logger.info(f"self.clone_batch={self.clone_batch}, self.use_all_clones={self.use_all_clones}")
        self.step_counter = 0

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        self.step_counter += 1
        
        scaled_losses = {}

        d2v_loss = net_output["losses"]
        for lk, p in d2v_loss.items():
            # logger.info(f"lk={lk}, loss={p.float().sum()}")
            scaled_losses[lk] = self.d2v_weight * p.float().sum()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        nsentences = sample["id"].numel()
        loss_d2v = loss
        logging_output = {
            "loss_d2v": loss_d2v,
            "ntokens": sample_size,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "_world_size": 1,
        }

        for lk in self.log_keys:
            for _out_k in net_output.keys():
                if _out_k.startswith(lk) and net_output[_out_k] is not None:
                    if not torch.is_tensor(net_output[_out_k]) or net_output[_out_k].numel() == 1:
                        logging_output[_out_k] = float(net_output[_out_k])
                    else:
                        for i, v in enumerate(net_output[_out_k]):
                            logging_output[f"{_out_k}_{i}"] = float(v)

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"loss_{lk}"] = l.item()

        if self.ctr_weight > 0:
            ctr_loss = self.compute_ctr_loss(
                net_output["proj_s"], net_output["proj_t"], use_all_clones=self.use_all_clones
            )
            logging_output["loss_ctr"] = ctr_loss
            loss = loss + ctr_loss

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        logging_output["loss"] = loss.data

        return loss, sample_size, logging_output

    def compute_ctr_loss(self, proj_s, proj_t, use_all_clones=True):
        nclone = self.clone_batch
        if not use_all_clones:
            M, D = proj_s.size()
            B = M // nclone
            proj_s = proj_s.reshape(B, nclone, D)
            indices = torch.randint(0, nclone, (B,))
            proj_s = proj_s[torch.arange(B), indices, :] # B x D
            nclone = 1
        proj_s = F.normalize(proj_s,dim=1,p=2) # M x D (M=Bxclone_batch)
        proj_t = F.normalize(proj_t,dim=1,p=2) # B x D
        _n = 0
        if dist.is_initialized():
            proj_t_gathered = concat_all_gather(proj_t)
            proj_t = proj_t_gathered.type(proj_s.dtype).to(device=proj_s.device) # num_gpu*B x D
            _n = dist.get_rank()

        logits = torch.matmul(proj_s, proj_t.transpose(1, 0))
        logits = logits / self.ctr_temperature
        bs = logits.size(0) // nclone
        label = (torch.arange(bs, dtype=torch.long) +
                 bs * _n).repeat_interleave(nclone, 0).to(device=logits.device)
        return self.ctr_weight * 2 * self.ctr_temperature * nn.CrossEntropyLoss()(logits, label)
    
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


def concat_all_gather(z: torch.Tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    group = distributed_utils.get_global_group()
    rank = distributed_utils.get_rank(group=group)
    world_size = distributed_utils.get_world_size(group=group)
    z = utils.move_to_cpu(z)
    gathered_zs = [
        torch.zeros_like(z, device="cpu") for _ in range(world_size)
    ]
    gathered_zs[rank] = z.clone()
    gathered_zs = torch.cat(gathered_zs, dim=0)

    return gathered_zs