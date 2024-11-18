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
    d2v_weight: float = field(
        default=1.0,
        metadata={"help": "weight of data2vec loss"},
    )
    ncp_weight: float = field(
        default=0.0,
        metadata={"help": "weight of next chunk prediction loss"},
    )
    ncp_loss_fn: str = field(
        default="none",
        metadata={"help": "type of loss function"},
    )
    moco_weight: float = field(
        default=0.0,
        metadata={"help": "weight of MoCo-v3 loss"},
    )
    moco_temperature: float = field(
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
        ncp_weight=0.0,
        ncp_loss_fn=None,
        moco_weight=0.0,
        moco_temperature=0.0,
        log_keys=None,
        clone_batch=8,
        use_all_clones=True,
        embed_dim=768,
        start_step_train_aux_loss=0,
    ):
        super().__init__(task)
        
        self.log_keys = log_keys
        self.d2v_weight = d2v_weight
        self.ncp_weight = ncp_weight
        self.ncp_loss_fn = ncp_loss_fn
        if self.ncp_weight > 0:
            self.nc_projector = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim//2, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim//2, embed_dim*2, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim*2, 1, bias=False),
                nn.Sigmoid(),
            )
            assert ncp_loss_fn in ["bce", "custom"]
            for param in self.nc_projector.parameters():
                param.requires_grad = False
        self.moco_weight = moco_weight
        self.moco_temperature = moco_temperature
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

        if self.ncp_weight > 0:
            if self.step_counter >= self.start_step_train_aux_loss:
                if all(not param.requires_grad for param in self.nc_projector.parameters()):
                    for param in self.nc_projector.parameters():
                        param.requires_grad = True
                ncp_loss, ncp_acc_logs = self.compute_ncp_loss(
                    net_output["local_features"], 
                    loss_fn=self.ncp_loss_fn,
                    use_all_clones=self.use_all_clones,
                    )
                loss += self.ncp_weight * ncp_loss
                logging_output["loss_ncp"] = self.ncp_weight * ncp_loss
                for k, v in ncp_acc_logs.items():
                    logging_output[k] = v * 100

        if self.moco_weight > 0:
            moco_loss = self.compute_moco_loss(
                net_output["proj_s"], net_output["proj_t"], use_all_clones=self.use_all_clones
            )
            logging_output["loss_moco"] = moco_loss
            loss = loss + moco_loss

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        logging_output["loss"] = loss.data

        return loss, sample_size, logging_output
    
    def compute_ncp_loss(self, local_features, loss_fn="bce", use_all_clones=True):
        # local_features: B x T x C or B x clone_batch x T x C
        for _, feature in local_features.items():
            if feature is not None:
                if feature.dim() == 4:
                    B, nclone, T, C = feature.size()
                    if not use_all_clones:
                        indices = torch.randint(0, nclone, (B,))
                        feature = feature[torch.arange(B), indices, :, :]
                        nclone = 1
                    feature = feature.reshape(B*nclone, T, C) # M = Bxnclone
                else:
                    B, T, C = feature.size()
                    nclone = 1

                crops1, crops2 = get_random_crops(feature) # M x crop_size x D
                Y1 = crops1.mean(dim=1) # M x D
                Y2 = crops2.mean(dim=1) # M x D

                X_pos = torch.cat(
                    (Y1, Y2), dim=-1
                ) # M x 2D
                X_neg = create_negative_pairs(Y1, Y2, nclone=nclone) # M(M-nclone) x 2D

                logits_pos = self.nc_projector(X_pos) # (M, 1)
                logits_neg = self.nc_projector(X_neg) # M(M-nclone) x 1
                logits = torch.cat([logits_pos, logits_neg], dim=0)  # Shape (M + M*(M-nclone), 1)
                num_pos, num_neg = logits_pos.numel(), logits_neg.numel()
                pos_weight = torch.tensor([num_neg / num_pos], device=logits.device)

                if loss_fn == "bce":
                    targets_pos = torch.ones_like(logits_pos)
                    targets_neg = torch.zeros_like(logits_neg)
                    targets = torch.cat([targets_pos, targets_neg], dim=0)
                    loss = nn.BCELoss(weight=pos_weight)(logits, targets)
                    loss = loss.sum()
                else:
                    loss = ((1 - logits_pos)**2).sum() + (logits_neg ** 2).sum()

                # Calculate the total accuracy for both positive and negative logits
                correct_pos  = (logits[:logits_pos.numel()] >= 0.7).sum()
                correct_neg = (logits[logits_pos.numel():] < 0.3).sum()

                return loss, {"acc_ncp_pos": correct_pos/num_pos, "acc_ncp_neg": correct_neg/num_neg}
            
    def compute_moco_loss(self, proj_s, proj_t, use_all_clones=True):
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
        logits = logits / self.moco_temperature
        bs = logits.size(0) // nclone
        label = (torch.arange(bs, dtype=torch.long) +
                 bs * _n).repeat_interleave(nclone, 0).to(device=logits.device)
        return self.moco_weight * 2 * self.moco_temperature * nn.CrossEntropyLoss()(logits, label)
    
    def compute_lim_weight_gather(self, local_features, eps=1e-9):
        # local_features: B x T x C
        for _, feature in local_features.items():
            if feature is not None:
                crops1, crops2 = get_random_crops(feature) # B x crop_size x C
                Y1 = self.nc_projector(crops1).mean(dim=1) # B x D
                Y2 = self.nc_projector(crops2).mean(dim=1) # B x D
                B, _ = Y1.size()
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
    # distributed_utils.all_reduce(gathered_zs, group=group)

    return gathered_zs