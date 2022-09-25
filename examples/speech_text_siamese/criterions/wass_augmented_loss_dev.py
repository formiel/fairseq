# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.dataclass.constants import ChoiceEnum

from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

# import optimal_transport as ot
from geomloss import SamplesLoss

from .soft_dtw_cuda import SoftDTW

WASS_METRIC_CHOICES = ChoiceEnum(["euclidean", "lp", "dot", "dotexp", "cosine", "none"])
SAMPLE_LOSS_CHOICES = ChoiceEnum(["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"])

@dataclass
class CtcWassersteinCriterionConfig(CtcCriterionConfig):
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    attn_weight_speech: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    attn_weight_text: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    mlm_weight: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    ot_weight: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    ot_weight_embed: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    ot_weight_st: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    ot_weight_st_ctc: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    ot_weight_mt: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + ot_weight * ot_loss"},
    )
    dtw_weight: float = field(
        default=0.0,
        metadata={"help": "loss = ctc_weight * ctc_loss + attn_weight_speech * attn_loss_speech \
            + attn_weight_text * attn_loss_text \
            + (1 - ctc_weight - attn_weight_speech - attn_weight_text - dtw_weight) * ot_loss \
            + dtw_weight * dtw_loss"},
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "lambda for KL divergence, 0 means no label smoothing"},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    copy_mechanism: bool = field(
        default=False,
        metadata={"help": "Use copy mechanism"},
    )
    do_bs1: bool = field(
        default=False,
        metadata={"help": "Compute Wasserstein loss for each example (because of padding)"},
    )
    zero_padding_weights: bool = field(
        default=False,
        metadata={"help": "Compute Wasserstein loss for each example (because of padding)"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy"},
    )
    compute_dist_decoder: bool = field(
        default=False,
        metadata={"help": "compute_dist_decoder"},
    )
    # ctc_zero_epoch: int = field(
    #     default=0,
    #     metadata={"help": "keeps ctc_weight constant for this number of epoch, starting from 0"},
    # )
    # ctc_warmup_epoch: int = field(
    #     default=0,
    #     metadata={"help": "increase ctc_weight linearly to ctc_weight for this number of epoch"},
    # )
    # use_wass_loss: bool = field(
    #     default=False,
    #     metadata={"help": "Use Wasserstein loss"},
    # )
    # use_wass_loss_st: bool = field(
    #     default=False,
    #     metadata={"help": "Use Wasserstein loss between speech out and prediction"},
    # )
    # use_wass_loss_st_ctc: bool = field(
    #     default=False,
    #     metadata={"help": "Use Wasserstein loss between speech out and prediction"},
    # )
    # use_wass_loss_mt: bool = field(
    #     default=False,
    #     metadata={"help": "Use Wasserstein loss between text out and prediction"},
    # )
    # wass_metric: WASS_METRIC_CHOICES = field(
    #     default="none", 
    #     metadata={"help": "type of distance measure between X_i and Y_j"}
    # )
    # wass_pos_cost: float = field(
    #     default=0.0,
    #     metadata={"help": "penalty to enforce alignment"},
    # )
    # wass_pos_epoch: int = field(
    #     default=0,
    #     metadata={"help": "Epoch at which the position cost decreases to 0"},
    # )
    use_cross_attentive_loss: bool = field(
        default=False,
        metadata={"help": "Use cross-attentive loss"},
    )
    use_soft_dtw_loss: bool = field(
        default=False,
        metadata={"help": "Use soft DTW loss"},
    )
    norm_before_ot: bool = field(
        default=False,
        metadata={"help": "Normalize before computing OT"},
    )
    ot_loss: SAMPLE_LOSS_CHOICES = field(
        default="sinkhorn", 
        metadata={"help": "type of distance measure between X_i and Y_j"}
    )
    ot_p: int = field(
        default=2,
        metadata={"help": "p in SampleLoss"},
    )
    ot_blur: float = field(
        default=0.05,
        metadata={"help": "blur in SampleLoss"},
    )
    ot_scaling: float = field(
        default=0.5,
        metadata={"help": "blur in SampleLoss"},
    )
    ot_position_weight: float = field(
        default=0.0,
        metadata={"help": "weight for positional embedding in OT"},
    )


@register_criterion("wasserstein_augmented_loss_dev", dataclass=CtcWassersteinCriterionConfig)
class CtcWassersteinCriterion(CtcCriterion):
    def __init__(self, cfg: CtcWassersteinCriterionConfig, task: FairseqTask):
        super().__init__(cfg, task)
        self.ctc_weight = cfg.ctc_weight
        self.attn_weight_speech = cfg.attn_weight_speech
        self.attn_weight_text = cfg.attn_weight_text
        self.mlm_weight = cfg.mlm_weight
        self.ot_weight = cfg.ot_weight
        self.ot_weight_st = cfg.ot_weight_st
        self.ot_weight_st_ctc = cfg.ot_weight_st_ctc
        self.ot_weight_mt = cfg.ot_weight_mt
        self.dtw_weight = cfg.dtw_weight
        self.eps = cfg.label_smoothing
        self.gamma = cfg.gamma
        self.copy_mechanism = cfg.copy_mechanism
        self.do_bs1 = cfg.do_bs1
        self.zero_padding_weights = cfg.zero_padding_weights
        self.ignore_prefix_size = cfg.ignore_prefix_size
        self.report_accuracy = cfg.report_accuracy
        self.compute_dist_decoder = cfg.compute_dist_decoder
        # self.ctc_zero_epoch = cfg.ctc_zero_epoch
        # self.ctc_warmup_epoch = cfg.ctc_warmup_epoch
        # self.ctc_weight_val = cfg.ctc_weight \
        #     if self.ctc_zero_epoch == 0 and self.ctc_warmup_epoch == 0 else 0.0
        # self.use_wass_loss = cfg.use_wass_loss
        # self.use_wass_loss_st = cfg.use_wass_loss_st
        # self.use_wass_loss_st_ctc = cfg.use_wass_loss_st_ctc
        # self.use_wass_loss_mt = cfg.use_wass_loss_mt
        self.ot_loss = cfg.ot_loss
        self.ot_p = cfg.ot_p
        self.ot_blur = cfg.ot_blur
        self.ot_scaling = cfg.ot_scaling
        self.ot_weight_embed = cfg.ot_weight_embed
        self.norm_before_ot = cfg.norm_before_ot
        self.ot_position_weight = cfg.ot_position_weight
        if self.norm_before_ot or (self.ot_position_weight > 0.0):
            assert self.do_bs1
        # self.wass_metric = cfg.wass_metric
        # self.wass_pos_cost = cfg.wass_pos_cost
        # self.wass_pos_epoch = cfg.wass_pos_epoch
        # self.wass_pos_cost_val = cfg.wass_pos_cost
        self.use_cross_attentive_loss = cfg.use_cross_attentive_loss
        self.use_soft_dtw_loss = cfg.use_soft_dtw_loss
        logging.info(f"*** Weights in loss function ***")
        logging.info(f"ctc_weight = {self.ctc_weight}, gamma = {self.gamma}")
        logging.info(f"mlm_weight = {self.mlm_weight}")
        logging.info(f"ot_weight = {self.ot_weight}")
        logging.info(f"ot_weight_st = {self.ot_weight_st}, ot_weight_mt = {self.ot_weight_mt}")
        logging.info(f"ot_loss = {self.ot_loss}, ot_p = {self.ot_p}, ot_blur = {self.ot_blur}, ot_scaling = {self.ot_scaling}")
        logging.info(f"attn_weight_speech = {self.attn_weight_speech}")
        logging.info(f"attn_weight_text = {self.attn_weight_text}")
        logging.info(f"label smoothing eps = {self.eps}")
        # Initialize loss
        if self.dtw_weight > 0.0:
            self.soft_dtw_loss = SoftDTW(use_cuda=True, gamma=0.1)
        if self.ot_weight > 0.0 or self.ot_weight_mt > 0.0 or self.ot_weight_st or self.ot_weight_st_ctc:
            self.ot_loss = SamplesLoss(loss=cfg.ot_loss, 
                            p=self.ot_p, 
                            blur=self.ot_blur,
                            scaling=self.ot_scaling)
        if self.ot_weight_embed > 0.0:
            self.ot_loss_embed = SamplesLoss(loss=cfg.ot_loss, 
                            p=self.ot_p, 
                            blur=self.ot_blur,
                            scaling=self.ot_scaling)


    def forward(self, model, sample, reduce=True):
        net_input = sample["net_input"]
        text_mode = True if "src_tokens" not in net_input else False
        masked_tokens = None
        if sample["masked_target"] is not None:
            masked_tokens = sample["masked_target"].ne(self.pad_idx)

        net_output, encoder_out = model(
            **net_input, 
            masked_tokens=masked_tokens,
            use_encoder_outputs=True
        )
        if text_mode:
            sample_size = (net_input["masked_src_txt_tokens"].size(0) 
                                if self.sentence_avg else sample["ntokens"])
        else:
            sample_size = (net_input["src_tokens"].size(0) 
                                if self.sentence_avg else sample["ntokens"])
        
        loss = 0.0
        extra = {"ce_loss_speech": 0.0, "nll_loss_speech": 0.0,
                "ce_loss_text": 0.0, "nll_loss_text": 0.0,
                "ctc_loss": 0.0, 
                "mlm_loss": 0.0,
                "wass_loss": 0.0, # between speech enc_out and text enc_out
                "wass_loss_st": 0.0, # between speech enc_out and pred
                "wass_loss_st_ctc": 0.0, # between ctc_out and pred
                "wass_loss_mt": 0.0, # between text enc_out and pred
                "wass_loss_embed": 0.0,
                "dtw_loss": 0.0, 
                "cross_attn_loss": 0.0,
                }
        if not text_mode:
            if self.attn_weight_speech > 0.0:
                ce_loss_speech, extra = self.compute_ce_loss(
                    model, net_output, sample, extra, reduce=reduce, idx=0,
                )
                loss += self.attn_weight_speech * ce_loss_speech
            if self.ctc_weight > 0.0:
                ctc_loss, extra = self.compute_ctc_loss(
                    model, net_output, encoder_out, net_input, extra
                )
                loss += self.ctc_weight * ctc_loss

        if self.attn_weight_text > 0.0:
            ce_loss_text, extra = self.compute_ce_loss(
                model, net_output, sample, extra, reduce=reduce, idx=2,
            )
            loss += self.attn_weight_text * ce_loss_text

        if self.mlm_weight > 0.0:
            mlm_loss, extra = self.compute_mlm_loss(
                net_output,
                sample,
                masked_tokens,
                extra,
            )
            loss += self.mlm_weight * mlm_loss
        
        if not text_mode:
            # if isinstance(encoder_out, tuple) and encoder_out[0] is not None:
            if self.dtw_weight > 0.0 :
                dtw_loss = self.compte_soft_dtw(self.soft_dtw_loss, 
                                                encoder_out,
                                                model=model,
                                                net_output=net_output,
                                                net_input=net_input,
                                                compute_dist_decoder=self.compute_dist_decoder,
                                                )
                loss += self.dtw_weight * dtw_loss
                extra["dtw_loss"] = dtw_loss
            if self.ot_weight > 0.0:
                if not self.compute_dist_decoder:
                    assert model.encoder.text_encoder_aux is not None
                wass_loss = self.compute_wass_loss(self.ot_loss, 
                                                    encoder_out,
                                                    model=model,
                                                    net_output=net_output,
                                                    net_input=net_input,
                                                    compute_dist_decoder=self.compute_dist_decoder,
                                                    encoder_out_key="encoder_out",
                                                    )
                loss += self.ot_weight * wass_loss
                extra["wass_loss"] = wass_loss
            if self.ot_weight_embed > 0.0:
                wass_loss_embed = self.compute_wass_loss(self.ot_loss_embed, 
                                                    encoder_out,
                                                    model=model,
                                                    net_output=net_output,
                                                    net_input=net_input,
                                                    compute_dist_decoder=self.compute_dist_decoder,
                                                    encoder_out_key="embed_src_tokens",
                                                    )
                loss += self.ot_weight_embed * wass_loss_embed
                extra["wass_loss_embed"] = wass_loss_embed
            if self.use_cross_attentive_loss:
                cross_attn_loss = torch.sum(self.cross_attentive_loss(encoder_out))
                loss += cross_attn_loss
                extra["cross_attn_loss"] = cross_attn_loss
            if self.ot_weight_mt > 0.0 or self.ot_weight_st > 0.0:
                if self.ot_weight_mt > 0.0:
                    assert model.encoder.text_encoder_aux is not None
                speech_out = None
                text_out = None 
                if isinstance(encoder_out, tuple):
                    speech_out = encoder_out[0]["encoder_out"][0] # S x B x D
                    text_out = encoder_out[-1]["encoder_out"][0] # T x B x D
                else:
                    speech_out = encoder_out["encoder_out"][0]
                if self.ot_weight_st > 0.0: # between speech enc_out and dec_out
                    # wloss_st = SamplesLoss(loss=self.ot_loss, 
                    #                         p=self.ot_p, 
                    #                         blur=self.ot_blur,
                    #                         scaling=self.ot_scaling)
                    if not self.compute_dist_decoder:
                        wass_loss_st = self.ot_loss(
                            speech_out.float().transpose(0, 1).contiguous(),
                            net_output[1]["before_out_proj"].transpose(0, 1).contiguous()
                        ).sum()
                    else:
                        lprobs = model.get_normalized_probs(net_output, log_probs=False)
                        target = F.one_hot(net_input["prev_output_tokens"], 
                                    num_classes=lprobs.size()[-1]) # BxT
                        wass_loss_st = self.ot_loss(
                                lprobs.float().contiguous(),
                                target.float().contiguous()
                            ).sum()

                    loss += wass_loss_st * self.ot_weight_st
                    extra["wass_loss_st"] = wass_loss_st
                if self.ot_weight_mt > 0.0: # between text enc_out and dec_out
                    # wloss_mt = SamplesLoss(loss=self.ot_loss, 
                    #                         p=self.ot_p, 
                    #                         blur=self.ot_blur,
                    #                         scaling=self.ot_scaling)
                    wass_loss_mt = self.ot_loss(
                        text_out.float().transpose(0, 1).contiguous(),
                        net_output[1]["before_out_proj"].transpose(0, 1).contiguous()
                    ).sum()
                    loss += wass_loss_mt * self.ot_weight_mt
                    extra["wass_loss_mt"] = wass_loss_mt
            if self.ot_weight_st_ctc:
                assert isinstance(net_output[0], tuple)
                # ctc_out = net_output[0][1] # T x B x D
                # wloss_st_ctc = SamplesLoss(loss=self.ot_loss, 
                #                             p=self.ot_p, 
                #                             blur=self.ot_blur,
                #                             scaling=self.ot_scaling)
                # wass_loss_st_ctc = self.ot_loss(
                #     ctc_out.float().transpose(0, 1).contiguous(),
                #     net_output[0][0] # B x T x C
                # ).sum()
                wass_loss_st_ctc = self.compute_wass_loss(self.ot_loss, 
                                                    encoder_out,
                                                    model=model,
                                                    net_output=net_output,
                                                    net_input=net_input,
                                                    compute_dist_decoder=self.compute_dist_decoder,
                                                    ot_spch_dec_and_ctc_out=True
                                                    )
                extra["wass_loss_st_ctc"] = wass_loss_st_ctc

        logging_output = {
            "loss": utils.item(loss.data) if loss != 0.0 else 0.0,  # * sample['ntokens'],
            "ce_loss_speech": utils.item(extra["ce_loss_speech"].data) if extra["ce_loss_speech"] != 0.0 else 0.0,
            "nll_loss_speech": utils.item(extra["nll_loss_speech"].data) if extra["nll_loss_speech"] != 0.0 else 0.0,
            "ce_loss_text": utils.item(extra["ce_loss_text"].data) if extra["ce_loss_text"] != 0.0 else 0.0,
            "nll_loss_text": utils.item(extra["nll_loss_text"].data) if extra["nll_loss_text"] != 0.0 else 0.0,
            "ctc_loss": utils.item(extra["ctc_loss"].data) if extra["ctc_loss"] != 0.0 else 0.0,
            "mlm_loss": utils.item(extra["mlm_loss"].data) if extra["mlm_loss"] != 0.0 else 0.0,
            "wass_loss": utils.item(extra["wass_loss"].data) if extra["wass_loss"] != 0.0 else 0.0,
            "wass_loss_embed": utils.item(extra["wass_loss_embed"].data) if extra["wass_loss_embed"] != 0.0 else 0.0,
            "wass_loss_st": utils.item(extra["wass_loss_st"].data) if extra["wass_loss_st"] != 0.0 else 0.0,
            "wass_loss_st_ctc": utils.item(extra["wass_loss_st_ctc"].data) if extra["wass_loss_st_ctc"] != 0.0 else 0.0,
            "wass_loss_mt": utils.item(extra["wass_loss_mt"].data) if extra["wass_loss_mt"] != 0.0 else 0.0,
            "dtw_loss": utils.item(extra["dtw_loss"].data) if extra["dtw_loss"] != 0.0 else 0.0,
            "cross_attn_loss": utils.item(extra["cross_attn_loss"].data) if extra["cross_attn_loss"] != 0.0 else 0.0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["id"].numel(),
            "sample_size": net_input["src_tokens"].size(0) if self.sentence_avg else sample["ntokens"],
            # "wass_pos_cost": self.wass_pos_cost_val,
            # "ctc_weight": self.ctc_weight,
            # "reduced_speech_output": net_output[-1].get("reduced_speech_output", 0.0),
        }

        if not model.training and self.ctc_weight > 0.0:
            logging_output = self.compute_wer(
                extra["lprobs_ctc"], sample, net_input, extra["input_lengths"], logging_output)

        if self.report_accuracy:
            if not text_mode and self.attn_weight_speech > 0.0:
                n_correct, total = self.compute_accuracy(extra["lprobs_ce_speech"], extra["target"])
                logging_output["n_correct_speech"] = utils.item(n_correct.data)
                logging_output["total_speech"] = utils.item(total.data)
            if self.attn_weight_text > 0.0:
                n_correct, total = self.compute_accuracy(extra["lprobs_ce_text"], extra["target"])
                logging_output["n_correct_text"] = utils.item(n_correct.data)
                logging_output["total_text"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, idx=0):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, idx=idx)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_ce_loss(self, model, net_output, sample, extra, reduce=True, idx=0):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, idx=idx)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        suffix = "_speech" if idx == 0 else "_text" if idx==2 else ""
        extra[f"ce_loss{suffix}"] = loss
        extra[f"nll_loss{suffix}"] = nll_loss
        extra[f"lprobs_ce{suffix}"] = lprobs
        extra["target"] = target
        return loss, extra

    def compute_mlm_loss(self, net_output, sample, masked_tokens, extra):
        logits = net_output[0][-1]
        targets = sample["masked_target"]
        if masked_tokens is not None:
            targets = targets[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.pad_idx,
        )   
        extra["mlm_loss"] = loss
        return loss, extra

    def compute_accuracy(self, lprobs, target):
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def compute_ctc_loss(self, model, net_output, encoder_out, net_input, extra):
        lprobs = model.get_normalized_probs(
                net_output, log_probs=True, idx=1,
            ).contiguous()  # (T, B, C) from the encoder
        
        spch_encoder_out = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
        if spch_encoder_out["encoder_padding_mask"]:
            non_padding_mask = ~spch_encoder_out["encoder_padding_mask"][0]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )
        pad_mask = (net_input["src_txt_tokens"] != self.pad_idx) & (
                net_input["src_txt_tokens"] != self.eos_idx)
        targets_flat = net_input["src_txt_tokens"].masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        # label smoothing
        kldiv_loss = 0.0
        if self.gamma > 0:
            kldiv_loss = F.kl_div(
                lprobs.transpose(0, 1), 
                torch.full_like(lprobs.transpose(0, 1), 1 / (lprobs.size(-1) - 1)), 
                reduction="batchmean",
            )
        ctc_loss = (1. - self.gamma) * ctc_loss + self.gamma * kldiv_loss

        extra["ctc_loss"] = ctc_loss
        extra["lprobs_ctc"] = lprobs
        extra["input_lengths"] = input_lengths

        return ctc_loss, extra

    def compute_wer(self, lprobs, sample, net_input, input_lengths, logging_output):
        import editdistance
        with torch.no_grad():
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            wv_errs = 0
            for lp, t, inp_l in zip(
                lprobs_t,
                sample["target_label"]
                if "target_label" in sample 
                else net_input["src_txt_tokens"],
                input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)
                decoded = None
                if self.w2l_decoder is not None:
                    decoded = self.w2l_decoder.decode(lp)
                    if len(decoded) < 1:
                        decoded = None
                    else:
                        decoded = decoded[0]
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                
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

                targ_words = post_process(targ_units, self.post_process).split()

                pred_units = self.task.source_dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units, self.post_process).split()

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    w_errs += editdistance.eval(pred_words, targ_words)
                    wv_errs += editdistance.eval(pred_words_raw, targ_words)
                else:
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                w_len += len(targ_words)
            logging_output["wv_errors"] = wv_errs
            logging_output["w_errors"] = w_errs
            logging_output["w_total"] = w_len
            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len
        return logging_output

    def compute_wass_loss(self, ot_loss, encoder_out,
                            model=None,
                            net_output=None,
                            net_input=None,
                            compute_dist_decoder=False,
                            ot_spch_dec_and_ctc_out=False,
                            encoder_out_key="encoder_out"):
        if compute_dist_decoder:
            if not ot_spch_dec_and_ctc_out:
                assert net_output is not None
                assert net_input is not None
                # pred = net_output[0].transpose(0, 1) # TxBxD -> BxTxD
                lprobs = model.get_normalized_probs(net_output, log_probs=False, idx=1)
                target = F.one_hot(net_input["src_txt_tokens"], 
                            num_classes=lprobs.size()[-1]) # BxT
                wass_loss = ot_loss(
                        lprobs.float().contiguous(),
                        target.float().contiguous()
                    ).sum()
            else:
                spch_dec_out = net_output[0][0] # BxTxD
                ctc_dec_out = net_output[0][1].transpose(0, 1) # TxBxD -> BxTxD
                wass_loss = ot_loss(
                        spch_dec_out.float().contiguous(),
                        ctc_dec_out.float().contiguous()
                    ).sum()
            return wass_loss
        
        speech_out = encoder_out[0][encoder_out_key][0] # S x B x D
        text_out = encoder_out[-1][encoder_out_key][0] # T x B x D
        # wloss = SamplesLoss(loss=self.ot_loss, 
        #                     p=self.ot_p, 
        #                     blur=self.ot_blur,
        #                     scaling=self.ot_scaling)
        if not self.do_bs1:
            if not self.zero_padding_weights:
                wass_loss = ot_loss(
                    speech_out.float().transpose(0, 1).contiguous(),
                    text_out.float().transpose(0, 1).contiguous()
                ).sum()  # use constant weights = 1/number of samples
            else:
                B, S, T = speech_out.size()[1], speech_out.size()[0], text_out.size()[0]
                non_padding_speech = (torch.ones(B, S) > 0).to(device=speech_out.device)
                non_padding_text = (torch.ones(B, T) > 0).to(device=text_out.device)
                if encoder_out[0]["encoder_padding_mask"]:
                    non_padding_speech = ~encoder_out[0]["encoder_padding_mask"][0] # B x S
                if encoder_out[1]["encoder_padding_mask"]:
                    non_padding_text = ~encoder_out[1]["encoder_padding_mask"][0] # B x T
                speech_weights = (
                    torch.ones_like(non_padding_speech) / 
                    torch.sum(non_padding_speech, dim=-1).unsqueeze(-1) *
                    non_padding_speech
                )
                text_weights = (
                    torch.ones_like(non_padding_text) / 
                    torch.sum(non_padding_text, dim=-1).unsqueeze(-1) *
                    non_padding_text
                )
                wass_loss = ot_loss(
                    speech_weights.float(),
                    speech_out.float().transpose(0, 1).contiguous(),
                    text_weights.float(),
                    text_out.float().transpose(0, 1).contiguous()
                ).sum()
        else:
            speech_lens =encoder_out[0]["input_lengths"][0]
            text_lens = encoder_out[1]["src_lengths"][0].squeeze(-1)
            # logging.info(f"speech_lens: {speech_lens.size()}\n{speech_lens}")
            # logging.info(f"text_lens: {text_lens.size()}\n{text_lens}")
            # logging.info(f"BEFORE NORM: speech_out: {speech_out.size()}, text_out: {text_out.size()}")
            if self.norm_before_ot:
                speech_out = speech_out / torch.linalg.norm(speech_out, dim=-1, keepdim=True)
                text_out = text_out / torch.linalg.norm(text_out, dim=-1, keepdim=True)
                # logging.info(f"AFTER NORM: speech_out: {speech_out.size()}, text_out: {text_out.size()}")
            # compute Wasserstein loss for each example
            wass_loss = 0.0
            device = speech_out.device
            for i in range(speech_out.size()[1]):
                # un-padded sequence lengths 
                S = speech_lens[i]
                T = text_lens[i]
                speech_feat = speech_out[:S, i, :] # S x D
                text_feat = text_out[:T, i, :] # T x D
                # logging.info(f"speech_feat={speech_feat.size()}, text_feat={text_feat.size()}")
                if self.ot_position_weight > 0.0 and S > 1 and T > 1:
                    pos_S = self.ot_position_weight * (torch.tensor(range(S), device=device)/(S-1)).unsqueeze(-1)
                    pos_T = self.ot_position_weight * (torch.tensor(range(T), device=device)/(T-1)).unsqueeze(-1)
                    # logging.info(f"pos_S: {pos_S.size()}\n{pos_S}")
                    # logging.info(f"pos_T: {pos_T.size()}\n{pos_T}")
                    speech_feat = torch.cat((speech_feat, pos_S), dim=-1)
                    text_feat = torch.cat((text_feat, pos_T), dim=-1)
                    # logging.info(f"speech_feat={speech_feat.size()}, text_feat={text_feat.size()}")
                    # logging.info(f"speech_feat: {speech_feat}")
                    # logging.info(f"text_feat: {text_feat}")
                wass_loss += ot_loss(speech_feat, text_feat).sum()
        if self.copy_mechanism:
            assert not self.do_bs1 # not implemented for copy mechanism yet
            x1 = encoder_out[0]["embed_src_tokens"][0]
            x2 = encoder_out[1]["embed_src_tokens"][0]
            wass_loss += (ot_loss(x1.float().transpose(0, 1).contiguous(), 
                        text_out.float().transpose(0, 1).contiguous()).sum() +
                        ot_loss(x2.float().transpose(0, 1).contiguous(), 
                        speech_out.float().transpose(0, 1).contiguous()).sum())
        return wass_loss

    # def compute_wass_loss_old(self, pred, target):
    #     # pred, target: T x B x D
    #     pred = pred.transpose(0, 1).contiguous() # B x T x D
    #     target = target.transpose(0, 1).contiguous() # B x T x D
    #     loss, Z = ot.wasserstein_dist(pred, target, self.wass_metric, p=1, 
    #                                   position_cost=self.wass_pos_cost_val)
    #     return loss, Z

    def compte_soft_dtw(self, sdtw, 
                        encoder_out,
                        model=None, 
                        net_output=None, 
                        net_input=None,
                        compute_dist_decoder=False):
        if not compute_dist_decoder:
            pred = encoder_out[0]["encoder_out"][0].transpose(0, 1).contiguous() # TxBxD -> BxTxD
            target = encoder_out[1]["encoder_out"][0].transpose(0, 1).contiguous()
            return sdtw(pred, target).sum()
        else: # compute dtw between predictions and one-hot groundtruth
            assert net_output is not None
            assert net_input is not None
            # lprobs = net_output[0].transpose(0, 1) # TxBxD -> BxTxD
            lprobs = model.get_normalized_probs(net_output, log_probs=False, idx=1).transpose(0, 1)
            # logging.info(f"lprobs: {lprobs.size()}")
            target = F.one_hot(net_input["src_txt_tokens"], 
                        num_classes=lprobs.size()[-1]) # src_txt_tokens: BxT
            # logging.info(f"target: {target.size()}")
            return sdtw(lprobs, target).sum()
        

    def cross_attentive_loss(self, encoder_out, 
        teacher_masking=[], student_masking=[], eps=1e-6,
        cross_attentive_loss_with_norm=True,
    ):
        x = encoder_out[0]["encoder_out"][0].transpose(0, 1)  # from T X B X D to B X T X D
        y = encoder_out[1]["encoder_out"][0].transpose(0, 1)
        if cross_attentive_loss_with_norm:
            x = x / (x.norm(dim=2, keepdim=True) + eps)
            y = y / (y.norm(dim=2, keepdim=True) + eps)
        dim = x.size(-1)
        # lengths: batch X seqLen
        sim_scores_xy = torch.bmm(x, y.transpose(1, 2))  # batch X lenx X leny ]
        if y.dtype == torch.float16:
            sim_scores_xy = sim_scores_xy.float()
            y = y.float()
            x = x.float()
        if teacher_masking != []:
            assert len(teacher_masking) == 1
            sim_scores_xy = sim_scores_xy.masked_fill(
                teacher_masking[0].unsqueeze(-1), float("-inf")
            )
        if student_masking != []:
            sim_scores_xy = sim_scores_xy.masked_fill(
                student_masking[0].unsqueeze(1), float("-inf")
            )
        # do masking
        y_weights = utils.softmax(sim_scores_xy, dim=-1)
        if teacher_masking != []:
            y_weights = y_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)
        x_reconstruct_from_y = torch.bmm(y_weights, y)

        sim_scores_xx = torch.bmm(x, x.transpose(1, 2))  # batch X lenx X lenx ]
        x_weights = utils.softmax(sim_scores_xx, dim=-1)
        if teacher_masking != []:
            x_weights = x_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)

        # no gradient for teacher state
        x_reconstruct_from_x = torch.bmm(x_weights, x).detach()
        cost = (x_reconstruct_from_x - x_reconstruct_from_y).norm(dim=2)
        if teacher_masking != []:
            cost = cost.masked_fill(teacher_masking[0], 0)

        if not cross_attentive_loss_with_norm:
            cost = cost / dim
        return cost

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ce_loss_speech_sum = utils.item(sum(log.get("ce_loss_speech", 0) for log in logging_outputs))
        ce_loss_text_sum = utils.item(sum(log.get("ce_loss_text", 0) for log in logging_outputs))
        mlm_loss_sum = utils.item(sum(log.get("mlm_loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        wass_loss_sum = utils.item(sum(log.get("wass_loss", 0) for log in logging_outputs))
        wass_loss_embed_sum = utils.item(sum(log.get("wass_loss_embed", 0) for log in logging_outputs))
        wass_loss_st_sum = utils.item(sum(log.get("wass_loss_st", 0) for log in logging_outputs))
        wass_loss_st_ctc_sum = utils.item(sum(log.get("wass_loss_st_ctc", 0) for log in logging_outputs))
        wass_loss_mt_sum = utils.item(sum(log.get("wass_loss_mt", 0) for log in logging_outputs))
        dtw_loss_sum = utils.item(sum(log.get("dtw_loss", 0) for log in logging_outputs))
        cross_attn_loss = utils.item(sum(log.get("cross_attn_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        # wass_pos_cost = sum(log.get("wass_pos_cost", 0) for log in logging_outputs)
        # ctc_weight = logging_outputs[0].get("ctc_weight", 0.0)
        reduced_speech_output = utils.item(
            sum(log.get("reduced_speech_output", 0.0) for log in logging_outputs)
            ) / len(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # metrics.log_scalar("ctc_weight", ctc_weight, 0, round=3)
        if ce_loss_speech_sum != 0.0:
            metrics.log_scalar(
                "ce_loss_speech", ce_loss_speech_sum / sample_size / math.log(2), sample_size, round=3
            )
        if ce_loss_text_sum != 0.0:
            metrics.log_scalar(
                "ce_loss_text", ce_loss_text_sum / sample_size / math.log(2), sample_size, round=3
            )
        if ctc_loss_sum != 0.0:
            metrics.log_scalar(
                "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if mlm_loss_sum != 0.0:
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if wass_loss_sum != 0:
            metrics.log_scalar(
                "wass_loss", wass_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            # metrics.log_scalar("wass_pos_cost", wass_pos_cost / len(logging_outputs), weight=0, round=0)
        if wass_loss_embed_sum != 0:
            metrics.log_scalar(
                "wass_loss_embed", wass_loss_embed_sum / sample_size / math.log(2), sample_size, round=3
            )
        if wass_loss_st_sum != 0:
            metrics.log_scalar(
                "wass_loss_st", wass_loss_st_sum / sample_size / math.log(2), sample_size, round=3
            )
        if wass_loss_st_ctc_sum != 0:
            metrics.log_scalar(
                "wass_loss_st_ctc", wass_loss_st_ctc_sum / sample_size / math.log(2), sample_size, round=3
            )
        if wass_loss_mt_sum != 0:
            metrics.log_scalar(
                "wass_loss_mt", wass_loss_mt_sum / sample_size / math.log(2), sample_size, round=3
            )
        if dtw_loss_sum != 0:
            metrics.log_scalar(
                "dtw_loss", dtw_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if cross_attn_loss != 0:
            metrics.log_scalar(
                "cross_attn_loss", cross_attn_loss / sample_size / math.log(2), sample_size, round=3
            )
        metrics.log_scalar("reduced_speech_output", reduced_speech_output)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
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

        total = utils.item(sum(log.get("total_speech", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total_speech", total)
            n_correct = utils.item(
                sum(log.get("n_correct_speech", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct_speech", n_correct)
            metrics.log_derived(
                "accuracy_speech",
                lambda meters: round(
                    meters["n_correct_speech"].sum * 100.0 / meters["total_speech"].sum, 3
                )
                if meters["total_speech"].sum > 0
                else float("nan"),
            )
        
        total = utils.item(sum(log.get("total_text", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total_text", total)
            n_correct = utils.item(
                sum(log.get("n_correct_text", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct_text", n_correct)
            metrics.log_derived(
                "accuracy_text",
                lambda meters: round(
                    meters["n_correct_text"].sum * 100.0 / meters["total_text"].sum, 3
                )
                if meters["total_text"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
     