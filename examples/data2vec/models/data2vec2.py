# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
from pathlib import Path
import numpy as np

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import checkpoint_utils
from fairseq import modules
from fairseq.modules import EMAModule, EMAModuleConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.roberta.model import RobertaLMHead

from examples.data2vec.data.modality import Modality

from examples.data2vec.models.modalities.base import (
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_annealed_rate,
)
from examples.data2vec.models.modalities.modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from examples.data2vec.models.modalities.audio import (
    D2vAudioConfig,
    AudioEncoder,
)
from examples.data2vec.models.modalities.images import (
    D2vImageConfig,
    ImageEncoder,
)
from examples.data2vec.models.modalities.text import (
    D2vTextConfig,
    TextEncoder,
)
from examples.pantagruel.models.modules import MHAPooling

logger = logging.getLogger(__name__)


@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()
    image: D2vImageConfig = D2vImageConfig()
    text: D2vTextConfig = D2vTextConfig()


@dataclass
class Data2VecMultiConfig(FairseqDataclass):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    clone_batch: int = 1

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")
    max_update: int = II("optimization.max_update")

    skip_ema: bool = False

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    mlm_loss: float = 0
    mlm_num_layers: int = 12
    mlm_impl: Optional[str] = None

    contrastive_loss: float = 0
    num_freeze_contrastive_updates: int = 0
    use_map_head_for_speech: Optional[bool] = False
    num_map_heads: Optional[int] = 1
    use_linear_head_for_text: Optional[bool]= False

    std_coeff: float = 0.0
    cov_coeff: float = 0.0

    decoder_group: bool = False
    mlm_group: bool = False

    pretrained_model_path: Optional[str] = None

    mlm_decay_steps: int = 0
    mlm_start_ratio: float = 0


@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel):
    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> ModalitySpecificEncoder:
        if cfg.type == Modality.AUDIO:
            enc_cls = AudioEncoder
        elif cfg.type == Modality.IMAGE:
            enc_cls = ImageEncoder
        elif cfg.type == Modality.TEXT:
            enc_cls = TextEncoder
            if hasattr(task, "text_task"):
                task = task.text_task
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities
        self.task = task

        self.padding_idx = None

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            enc = self.make_modality_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )
            self.modality_encoders[mod.name] = enc

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        if self.cfg.mae_init:
            self.apply(self._init_weights)
        else:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()

        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay)
            self.shared_decoder = (
                Decoder1d(cfg.shared_decoder, cfg.embed_dim)
                if self.cfg.shared_decoder is not None
                else None
            )
            if self.shared_decoder is not None:
                self.shared_decoder.apply(self._init_weights)

            self.recon_proj = None
            if cfg.recon_loss > 0:
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
            self.mlm_head = None
            self.mlm_num_layers = 0
            self.mlm_impl = "use_decoder_output" if not getattr(cfg, "mlm_impl", None) else "original_bert" # use forward to decoder as default implementation for backward compatibility
            logger.info(f"self.mlm_impl: {self.mlm_impl}")
            if cfg.mlm_loss > 0:
                self.mlm_num_layers = getattr(cfg, "mlm_num_layers", 12)
                # text modality
                self.padding_idx = task.dictionary.index("<pad>")
                assert self.padding_idx != task.dictionary.unk()
                logger.info(f"padding idx: {self.padding_idx}, vocab size: {len(task.dictionary)}")
                self.mlm_head = RobertaLMHead(
                    cfg.embed_dim, len(task.dictionary), "gelu"
                )
                if getattr(cfg, "mlm_group", False):
                    for p in self.mlm_head.parameters():
                        p.param_group = "mlm_head"

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        self.mlm_decay_steps = getattr(cfg, "mlm_decay_steps", 0)
        self.mlm_start_ratio = getattr(cfg, "mlm_start_ratio", 0.0)
        logger.info(f"mlm_decay_steps={self.mlm_decay_steps}, mlm_start_ratio={self.mlm_start_ratio}")

        self.contr_loss_weight = getattr(cfg, "contrastive_loss", 0.0)
        self.contr_logit_scale, self.contr_logit_bias = 1.0, 1.0
        self.num_freeze_contrastive_updates = getattr(cfg, "num_freeze_contrastive_updates", 0)
        if self.contr_loss_weight > 0.0:
            self.contr_logit_scale = nn.Parameter(torch.randn(1))
            self.contr_logit_bias = nn.Parameter(torch.randn(1))
            self.aux_heads = nn.ModuleDict()
            if getattr(cfg, "use_map_head_for_speech", False):
                self.aux_heads["AUDIO"] = MHAPooling(
                    cfg.embed_dim, getattr(cfg, "num_map_heads", 1)
                )
            if getattr(cfg, "use_linear_head_for_text", False):
                self.aux_heads["TEXT"] = nn.Linear(
                    cfg.embed_dim, cfg.embed_dim
                )

        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def make_ema_teacher(self, ema_decay):
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,
            ema_fp32=True,
            log_norms=self.cfg.log_norms,
            add_missing_params=False,
        )

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    def make_target_model(self):
        logger.info("making target model")

        ema_cfg = copy.deepcopy(self.cfg)
        if hasattr(ema_cfg, "use_map_head_for_speech"):
            ema_cfg.use_map_head_for_speech = False
        if hasattr(ema_cfg, "use_linear_head_for_text"):
            ema_cfg.use_linear_head_for_text = False

        model_copy = Data2VecMultiModel(
            ema_cfg, self.modalities, skip_ema=True, task=self.task
        )
        logger.info(f"model_copy: {model_copy}")

        if ema_cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            # for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
            #     p_t.data.copy_(p_s.data)
            excluded_modules = ["aux_heads"]
            for name, p_s in self.named_parameters():
                if any(excluded in name for excluded in excluded_modules):
                    continue
                if name in dict(model_copy.named_parameters()):
                    p_t = dict(model_copy.named_parameters())[name]
                    p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]
            )
        else:
            modalities = task.supported_modalities
        model = cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

        pretraining_path = getattr(cfg, "pretrained_model_path", None)
        logger.info(f'pretraining_path: {pretraining_path}')
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                state = torch.load(pretraining_path, map_location=torch.device("cpu"))
                pretrained_state_dict = state["model"]
                strict = True if getattr(cfg, "mlm_num_layers", 12) == cfg.depth else False
                model.load_state_dict(pretrained_state_dict, strict=strict)
                logger.info(f"loaded pretrained encoder from: {pretraining_path} with strict={strict}")
                ema_pretrained = {}
                for k, v in pretrained_state_dict["_ema"].items():
                    ema_pretrained[k.replace("blocks.", "")] = v
                model.ema.model.load_state_dict(ema_pretrained, strict=True)
                logger.info(f"loaded pretrained encoder to EMA: {pretraining_path}")
        return model

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
        source_mlm=None,
        target_mlm=None,
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        feature_extractor = self.modality_encoders[mode]

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        x_mlm = None
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if self.mlm_impl == "use_decoder_output" and i <= self.mlm_num_layers - 1:
                    x_mlm = x.clone()
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        xs = []

        if self.shared_decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                self.shared_decoder,
                encoder_mask,
            )
            xs.append(dx)
        if feature_extractor.decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                feature_extractor.decoder,
                encoder_mask,
            )
            xs.append(dx)
            orig_x = x

        x_masked_for_mlm = None
        if self.cfg.mlm_loss > 0 and self.mlm_impl == "use_decoder_output":
            if self.mlm_num_layers < self.cfg.depth:
                x_masked_for_mlm = self.forward_decoder(
                    x_mlm,
                    feature_extractor,
                    feature_extractor.decoder,
                    encoder_mask,
                )
            else:
                x_masked_for_mlm = dx.clone()

        assert len(xs) > 0

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        tm = self.ema.model

        with torch.no_grad():
            tm.eval()

            if self.cfg.ema_encoder_only:
                assert target is None
                ema_input = extractor_out["local_features"]
                ema_input = feature_extractor.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                if feature_extractor.modality_cfg.ema_local_encoder:
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    ema_input = tm.modality_encoders[mode](
                        inp,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )
                else:
                    assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_feature_enc = tm.modality_encoders[mode]
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input.get("alibi_bias", None)
            ema_alibi_scale = ema_input.get("alibi_scale", None)
            ema_input = ema_input["x"]

            y = []
            ema_x = []
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):
                ab = ema_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                ema_input, lr = blk(
                    ema_input,
                    padding_mask=ema_padding_mask,
                    alibi_bias=ab,
                )
                y.append(lr[:, extra_tokens:])
                ema_x.append(ema_input[:, extra_tokens:])

        y = self.make_targets(y, self.average_top_k_layers)
        orig_targets = y

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        y = y[masked_b]

        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
        else:
            xs = [x.reshape(-1, x.size(-1)) for x in xs]

        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        if self.cfg.cls_loss > 0:
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]
            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )

        if self.cfg.recon_loss > 0:
            with torch.no_grad():
                target = feature_extractor.patchify(source)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(self.cfg.clone_batch, 0)

                if masked_b is not None:
                    target = target[masked_b]

            recon = xs[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss
            )

        if self.cfg.mlm_loss > 0 and not features_only:
            # target_mlm is different for different implementation (depend on the skip_masking param in masked_lm task)
            # For use_decoder_output -> target_mlm is the original target sequence, 
            # used to eliminate the padded positions only. Masking is based on data2vec masking configuration
            # For original_bert -> target_mlm is tensor of padding_idx (not masked) and 1 (masked positions), masking strategy is ind
            if self.mlm_impl == "use_decoder_output":
                target_mlm = target_mlm.repeat_interleave(self.cfg.clone_batch, 0)
                valid_mask = masked_b & (target_mlm.ne(self.padding_idx))
                x_masked_for_mlm = x_masked_for_mlm[valid_mask]
                # mlm_logits = self.mlm_head(x_full, masked_tokens=valid_mask)
                mlm_logits = self.mlm_head(x_masked_for_mlm, masked_tokens=None)
                mlm_targets = target_mlm[valid_mask]
            elif self.mlm_impl == "original_bert":
                # do another forward like original BERT implementation
                mlm_extractor_out = feature_extractor(
                    source_mlm, None, False, False, 1, None, None
                )
                x_mlm = mlm_extractor_out["x"]
                encoder_mask_mlm = mlm_extractor_out["encoder_mask"]
                masked_padding_mask_mlm = mlm_extractor_out["padding_mask"]
                masked_alibi_bias_mlm = mlm_extractor_out.get("alibi_bias", None)
                alibi_scale_mlm = mlm_extractor_out.get("alibi_scale", None)
                for i, blk in enumerate(self.blocks):
                    if i <= self.mlm_num_layers - 1:
                        if (
                            not self.training
                            or self.cfg.layerdrop == 0
                            or (np.random.random() > self.cfg.layerdrop)
                        ):
                            ab = masked_alibi_bias_mlm
                            if ab is not None and alibi_scale_mlm is not None:
                                scale = (
                                    alibi_scale_mlm[i]
                                    if alibi_scale_mlm.size(0) > 1
                                    else alibi_scale_mlm.squeeze(0)
                                )
                                ab = ab * scale.type_as(ab)

                            x_mlm, _ = blk(
                                x_mlm,
                                padding_mask=masked_padding_mask_mlm,
                                alibi_bias=ab,
                            )

                masked_tokens = target_mlm.ne(self.padding_idx)
                # logger.info(f"masking percent: {int(masked_tokens.sum().item())} / {int(target_mlm.numel())}")
                masked_tokens = torch.where(
                    masked_tokens.any(), masked_tokens, masked_tokens.new([True])
                )
                mlm_logits = self.mlm_head(x_mlm, masked_tokens=masked_tokens)
                mlm_targets = target_mlm[masked_tokens]

            mlm_loss = modules.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )
            if self.mlm_decay_steps > 0:
                mlm_weight = max(
                    self.cfg.mlm_loss, # lambda_min
                    self.mlm_start_ratio - (self.mlm_start_ratio - self.cfg.mlm_loss) * (
                        self.num_updates / self.mlm_decay_steps
                    )
                )
            else:
                mlm_weight = self.cfg.mlm_loss

            result["losses"]["mlm"] = mlm_loss * mlm_weight

        if self.contr_loss_weight > 0.0:
            ft = self.num_freeze_contrastive_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                cls_target = orig_targets.mean(dim=1)
                if self.cfg.clone_batch > 1:
                    cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
                # cls_pred = x[:,0]
                cls_pred = self.get_sentence_level_pred(
                    x, mode, padding_mask=padding_mask
                )
                result["losses"]["contrastive"] = self.contrastive_loss(cls_pred, cls_target)

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                reg_loss = self.d2v_loss(x, y)
                n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss
                if getattr(self.cfg, "std_coeff", 0.0) > 0.0 or getattr(self.cfg, "cov_coeff", 0.0) > 0.0:
                    var_cov_loss = self.var_cov_loss(x, y)
                    result["losses"][n] += var_cov_loss

        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info) # full x
        x = decoder(*x)

        return x

    def get_sentence_level_pred(self, x, mode, padding_mask=None):
        if mode == "TEXT":
            x_pred = self.aux_heads["TEXT"](x[:, 0]) # CLS first representation
        elif mode == "AUDIO":
            x_pred = self.aux_heads["AUDIO"](x, padding_mask=padding_mask)
        return x_pred

    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss

    def contrastive_loss(self, pred, target):
        x_pred = F.normalize(pred, p=2, dim=-1)
        y_target = F.normalize(target, p=2, dim=-1)

        def block_diagonal_ones_vec(n, block_size, device):
            # Number of blocks
            num_blocks = n // block_size
            # Create a single block of ones
            block = torch.ones(block_size, block_size)
            # Repeat blocks along diagonal using kron (Kronecker product)
            A = torch.kron(torch.eye(num_blocks), block)
            return A.to(device=device)

        def get_labels(on_same_device):
            M = x_pred.size()[0]
            per_gpu_bsz = M // self.cfg.clone_batch # M = bsz * clone
            if on_same_device:
                # if clone_batch = 1
                # eye = torch.eye(per_gpu_bsz, device=x_pred.device)
                # labels = 2 * eye - torch.ones(per_gpu_bsz, device=x_pred.device)
                eye = block_diagonal_ones_vec(M, self.cfg.clone_batch, x_pred.device)
                labels = 2 * eye - torch.ones(M, device=x_pred.device) # 1 for pos, -1 for neg
            else:
                labels = -1
            return labels
        
        def compute_device_loss(zs, zt, on_same_device, temperature=0.1):
            logits = torch.mm(zs.float(), zt.T) / temperature # (B, D) @ (D, B) -> (B, B)
            logits = torch.clamp(logits, min=-50, max=50)
            logits = logits * self.contr_logit_scale.exp() + self.contr_logit_bias

            labels = get_labels(on_same_device=on_same_device)
            loss = F.logsigmoid(labels * logits)
            return -loss.sum()

        def gather_tensors_with_padding(tensor, world_size):
            """Gather tensors of different batch sizes across ranks."""
            local_bsz = tensor.shape[0]

            # gather the batch size across all devices
            local_shape = torch.tensor([local_bsz], device=tensor.device)
            gathered_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
            dist.all_gather(gathered_shapes, local_shape)

            gathered_shapes = torch.stack(gathered_shapes).cpu()
            max_size = gathered_shapes.max().item()

            # pad tensor to the max size
            padded_tensor = torch.zeros((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            padded_tensor[:local_bsz] = tensor

            # gather padded tensors
            gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, padded_tensor)

            # remove padding after gathering
            gathered_tensors = [g[:s.item()] for g, s in zip(gathered_tensors, gathered_shapes)]

            return gathered_tensors

        # Gather text embeddings from all devices
        world_size = dist.get_world_size()
        y_target_gathered = gather_tensors_with_padding(y_target, world_size)
        rank = dist.get_rank()

        loss_total = 0
        for i in range(world_size):
            per_device_loss = compute_device_loss(
                x_pred, y_target_gathered[i],
                on_same_device=(i==rank)
            )
            loss_total += per_device_loss

        return loss_total

    # see https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    def var_cov_loss(self, x, y):
        """
        compute the variance and covariance loss in VICReg
        x: BxTxD, y: BxTxD
        """
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        # logging.info(f"x: {x.size()}, y: {y.size()}")
        M, D = x.size()
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (M - 1)
        cov_y = (y.T @ y) / (M - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)
        
        var_cov_loss = self.cfg.std_coeff * std_loss + self.cfg.cov_coeff * cov_loss
        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(M)

        var_cov_loss = var_cov_loss * scale
        return var_cov_loss

    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        self.ema = None
        self.cfg.clone_batch = 1
        self.recon_proj = None

        if not keep_decoder:
            self.shared_decoder = None

        modality = modality.lower() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if modality is not None and k.lower() != modality:
                del self.modality_encoders[k]
            else:
                self.modality_encoders[k].remove_pretraining_modules(
                    keep_decoder=keep_decoder
                )
                if not keep_decoder:
                    self.modality_encoders[k].decoder = None