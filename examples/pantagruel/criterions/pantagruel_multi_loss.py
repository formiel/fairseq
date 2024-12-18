import os
import logging
import math
from dataclasses import dataclass, field
from typing import List

import editdistance

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
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of ctc loss"},
    )


@register_criterion("pantagruel_multi_loss", dataclass=PantagruelMultiConfig)
class PantagruelMultiCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        d2v_weight,
        ncp_weight=0.0,
        ncp_loss_fn=None,
        moco_weight=0.0,
        moco_temperature=0.0,
        ctc_weight=0.0,
        log_keys=None,
    ):
        super().__init__(task)
        
        self.log_keys = log_keys
        self.d2v_weight = d2v_weight
        self.ncp_weight = ncp_weight
        self.ncp_loss_fn = ncp_loss_fn
        if self.ncp_weight > 0:
            self.nc_projector = nn.Sequential(
                nn.Linear(768*2, 384),
                nn.Tanh(),
                nn.Linear(384, 1),
            )
            assert ncp_loss_fn in ["bce", "custom"]
            self.ncp_weight_learned = nn.Parameter(torch.tensor(ncp_weight), requires_grad=True)
        self.moco_weight = moco_weight
        self.moco_temperature = moco_temperature
        self.task = task
        self.ctc_weight = ctc_weight
        self.pad_idx = task.source_dictionary.pad()
        self.eos_idx = task.source_dictionary.eos()
        self.blank_idx = task.source_dictionary.bos()
        logger.info(f"blank_idx={self.blank_idx}, pad_idx={self.pad_idx}, eos_idx={self.eos_idx}")

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

        d2v_loss = net_output["losses"] # data2vec loss is computed in model
        for lk, p in d2v_loss.items():
            # logger.info(f"{lk}: {self.d2v_weight * p.float().sum()}")
            scaled_losses[lk] = self.d2v_weight * p.float().sum()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        nsentences = sample["id"].numel()
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "_world_size": 1,
        }

        suffix = list(scaled_losses.keys())
        if len(suffix) > 1:
            for sfx in suffix:
                logging_output[f"loss_d2v_{sfx}"] = scaled_losses[sfx]

        self.log_keys = [l for l in net_output if any([l.startswith(lk) for lk in self.log_keys])]
        for lk in self.log_keys:
            if net_output[lk] is not None:
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
            _ncp_loss, ncp_acc = self.compute_ncp_loss(
                net_output["local_features"], loss_fn=self.ncp_loss_fn
            )
            ncp_loss = F.softplus(self.ncp_weight_learned) * _ncp_loss
            loss = loss + ncp_loss
            logging_output["loss_ncp"] = ncp_loss
            logging_output["acc_ncp"] = ncp_acc * 100
            logging_output["weight_ncp"] = self.ncp_weight_learned.data
        if self.moco_weight > 0:
            moco_loss = self.compute_moco_loss(
                net_output["q"], net_output["k"]
            )
            logging_output["loss_moco"] = moco_loss
            loss = loss + self.moco_weight * moco_loss

        ctc_loss, lprobs = None, None
        if self.ctc_weight > 0 and net_output["ctc_out"]:
            ctc_loss, lprobs, sample_size_ctc, input_lengths = self.compute_ctc_loss(
                net_output["ctc_out"], net_input
            )
            logging_output["loss_ctc"] = ctc_loss
            logging_output["sample_size_ctc"] = sample_size_ctc
            if not net_output["ctc_out"]["is_frozen"]:
                loss = loss + ctc_loss

            if not model.training:
                _ctc_logging = self.compute_wer(
                    lprobs, net_output["ctc_out"], net_input, input_lengths
                )
                for k, v in _ctc_logging.items():
                    logging_output[k] = v

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        logging_output["loss"] = loss.data # update loss

        return loss, sample_size, logging_output

    def compute_ctc_loss(self, ctc_out, net_input):
        lprobs = utils.log_softmax(ctc_out["x"], dim=-1).contiguous()  # (T, B, C) from the encoder

        src_text = net_input["source"]["text"]["source"] # no special bos and eos tokens here
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

        pad_mask = (src_text != self.pad_idx) & (src_text != self.eos_idx)
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
        sample_size_ctc = src_text.size(0)

        return ctc_loss, lprobs, sample_size_ctc, input_lengths

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
    
    def compute_ncp_loss(self, local_features, loss_fn="bce"):
        # local_features: B x T x C or B x clone_batch x T x C
        for _, feature in local_features.items():
            if feature is not None:
                if feature.dim() == 4:
                    B, nclone, T, C = feature.size()
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

                if loss_fn == "bce":
                    targets_pos = torch.ones_like(logits_pos)
                    targets_neg = torch.zeros_like(logits_neg)
                    targets = torch.cat([targets_pos, targets_neg], dim=0)
                    # logits = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
                    loss = F.binary_cross_entropy_with_logits(logits, targets).sum()
                else:
                    loss = ((1 - logits_pos)**2).sum() + (logits_neg ** 2).sum()

                # Calculate the total accuracy for both positive and negative logits
                num_samples = logits_pos.numel() + logits_neg.numel()
                accuracy = (
                    (logits_pos >= 0.5).sum() + (logits_neg < 0.5).sum()
                ) / (num_samples)
 
                return loss, accuracy
            
    def compute_moco_loss(self, q, k):
        if q is None or k is None:
            return 0
        # q: B x nclone x C, k: BxC
        B, nclone, C = q.size()
        q = q.reshape(-1, C) # MxC
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.T) / self.moco_temperature # MxB
        logits = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
        labels = torch.arange(B, dtype=torch.long, device=q.device)
        labels = labels.repeat_interleave(nclone, 0)

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.moco_temperature)
    
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
        subloss_types = list(set([k for log in logging_outputs for k in log if k.startswith("loss_d2v_") or k=="loss_ctc"]))

        if len(subloss_types) >= 1:
            for key_full in subloss_types:
                mod = key_full.split("_")[-1]
                _loss = utils.item(sum(log.get(key_full, 0) for log in logging_outputs))
                _sample_size = utils.item(
                        sum(log.get(f"sample_size_{mod}", 0) for log in logging_outputs)
                    )
                metrics.log_scalar(key_full, _loss / _sample_size, _sample_size, round=3)
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

        # for k in logging_outputs[0]:
        #     if k not in builtin_keys and not k.startswith("_"):
        #         val = sum(log.get(k, 0) for log in logging_outputs)
        #         if k.startswith("loss_"):
        #             metrics.log_scalar(k, val / sample_size, sample_size, round=3)
        #         else:
        #             metrics.log_scalar(k, val / world_size, round=3)

        # correct = sum(log.get("correct", 0) for log in logging_outputs)
        # total = sum(log.get("count", 0) for log in logging_outputs)

        # if total > 0:
        #     metrics.log_scalar("_correct", correct)
        #     metrics.log_scalar("_total", total)

        #     metrics.log_derived(
        #         "accuracy",
        #         lambda meters: safe_round(
        #             meters["_correct"].sum / meters["_total"].sum, 5
        #         )
        #         if meters["_total"].sum > 0
        #         else float("nan"),
        #     )

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


@torch.no_grad()
def concat_all_gather(z: torch.Tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    gathered_zs = [torch.zeros_like(z)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=gathered_zs, tensor=z.contiguous())
    gathered_zs[dist.get_rank()] = z

    output = torch.cat(gathered_zs, dim=0)
    return output