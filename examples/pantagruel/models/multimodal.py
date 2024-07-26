# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
import numpy as np

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from examples.data2vec.data.modality import Modality

from examples.data2vec.models.modalities.base import (
    MaskSeed,
    get_annealed_rate,
    D2vModalityConfig,
)
from examples.pantagruel.models.modalities.base_type import (
    PantagruelModalitySpecificEncoder
)
from examples.data2vec.models.modalities.modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from examples.data2vec.models.modalities.audio import (
    D2vAudioConfig,
)
from examples.pantagruel.models.modalities.audio_type import (
    AudioTypeEncoder,
)
from examples.data2vec.models.modalities.images import (
    D2vImageConfig,
    ImageEncoder,
)
from examples.pantagruel.models.modalities.text_type import (
    TextTypeEncoder,
    PantagruelD2vTextConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class PantagruelD2vModalitiesConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()
    image: D2vImageConfig = D2vImageConfig()
    text: PantagruelD2vTextConfig = PantagruelD2vTextConfig()


@dataclass
class PantagruelData2VecMultiConfig(FairseqDataclass):

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

    modalities: PantagruelD2vModalitiesConfig = PantagruelD2vModalitiesConfig()

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

    skip_ema: bool = False

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False

    use_token_type_embeddings: bool = False
    adversarial_loss: float = field(
        default=0.0,
        metadata={"help": "Adversarial weight in loss function"},
    )
    num_discriminator_layers: int = field(
        default=-1,
        metadata={"help": "number of discriminator layers"},
    )
    num_discriminator_steps: int = field(
        default=-1,
        metadata={"help": "number of discriminator steps"},
    )
    dummy_factor: float = 0.0
    skip_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "skip_mode"
        },
    )



class LinearDiscriminator(nn.Module):
    """Adapted from https://github.com/facebookresearch/UnsupervisedMT/blob/main/NMT/src/model/discriminator.py
    """
    def __init__(self, 
            input_dim, 
            num_outputs=2, 
            layers=3, 
            hidden_dim=1024, 
            dropout=0.1):
        """
        Discriminator initialization.
        """
        super(LinearDiscriminator, self).__init__()
        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers = []
        for i in range(self.layers + 1):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.hidden_dim
            output_dim = self.hidden_dim if i < self.layers else self.num_outputs

            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.layers:
                layers.append(nn.LeakyReLU(0.01))
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)



@register_model("pantagruel_multi", dataclass=PantagruelData2VecMultiConfig)
class PantagruelMultiModel(BaseFairseqModel):
    def make_modality_type_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
        token_type_embeddings,
    ) -> PantagruelModalitySpecificEncoder:
        if cfg.type == Modality.AUDIO:
            enc_cls = AudioTypeEncoder
        elif cfg.type == Modality.IMAGE:
            enc_cls = ImageEncoder
        elif cfg.type == Modality.TEXT:
            enc_cls = TextTypeEncoder
            if hasattr(task, "text_task") and self.skip_mode is None:
                task = task.text_task
        else:
            raise Exception(f"unsupported modality {cfg.type}")
        logging.info(f"make_modality::task:{task}")
        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
            token_type_embeddings,
        )

    def __init__(self, cfg: PantagruelData2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()

        self.cfg = cfg
        self.modalities = modalities
        self.task = task
        self.mask_idx = self.task.mask_idx # not used yet

        self.dummy_factor = getattr(cfg, "dummy_factor", 0.0)
        self.skip_mode = getattr(cfg, "skip_mode", None)

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

        token_type_embeddings = None
        if cfg.use_token_type_embeddings:
            token_type_embeddings = nn.Embedding(len(self.modalities), cfg.embed_dim)
            nn.init.xavier_normal_(token_type_embeddings.weight)

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            enc = self.make_modality_type_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
                token_type_embeddings,
            )
            self.modality_encoders[mod.name] = enc

        self.discriminator = None
        self.adversarial_loss = getattr(cfg, "adversarial_loss", 0.0)
        self.num_discriminator_steps = getattr(cfg, "num_discriminator_steps", -1)
        self.step_counter = 0
        if self.adversarial_loss > 0:
            self.discriminator = LinearDiscriminator(
                input_dim=cfg.embed_dim, 
                num_outputs=len(self.modalities), 
                hidden_dim=cfg.embed_dim * 2,
                layers=cfg.num_discriminator_layers,
            )
        
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

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

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

        model_copy = PantagruelMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=self.task
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
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
            try:
                assert k in state_dict
                self.ema.restore(state_dict[k], True)
                del state_dict[k]
            except:
                pass
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: PantagruelData2VecMultiConfig, task=None):
        logging.info(f"build_model::task: {task}")
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
            modalities = (task.supported_modalities 
                if cfg.supported_modality is None
                else [cfg.supported_modality]
            )
        if cfg.skip_mode is not None:
            if "TEXT" in cfg.skip_mode:
                modalities.append(Modality.TEXT)
            if "AUDIO" in cfg.skip_mode:
                modalities.append(Modality.AUDIO)       
        logger.info(f"modalities::: {modalities}")

        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)
        
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
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        feature_extractor = self.modality_encoders[mode]
        remaining_extractor_names = [m.name for m in self.modalities if m.name != mode]
        
        token_type_ids = None
        remaining_token_type_ids = {}

        for it, im in enumerate(self.modalities):
            if im.name == mode:
                token_type_ids = torch.ones((source.size()[0]), dtype=torch.int64, device=source.device) * it
            else:
                remaining_token_type_ids[im.name] = torch.ones((source.size()[0]), dtype=torch.int64, device=source.device) * it

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        extractor_out = feature_extractor(
            source, # B x T
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
            token_type_ids=token_type_ids,
        )

        x = extractor_out["x"] # B x T x C
        B, _, _ = x.size()

        x_dummies, encoder_mask_dummies = None, None
        if len(remaining_extractor_names) > 0:
            # modality: TEXT, source dtype: torch.int64
            # modality: AUDIO, source dtype: torch.float16
            dummy_source_text = torch.randint(
                self.task.vocab_size - 1, 
                (B, self.task.tokens_per_sample), 
                dtype=torch.int64, 
                device=source.device
            )
            dummy_source_audio = torch.randn(
                (B, self.task.max_sample_size), 
                dtype=torch.float16, 
                device=source.device
            )
            # Whether to perform masking for dummy inputs?
            x_dummies, encoder_mask_dummies = [], []
            for name in remaining_extractor_names:
                dummy = dummy_source_audio if name == "AUDIO" else dummy_source_text
                dummy_outs = self.modality_encoders[name](
                    dummy, None, False, False, token_type_ids=remaining_token_type_ids[name]
                )
                x_dummies.append(dummy_outs["x"]) # B x T x C
                encoder_mask_dummies.append(dummy_outs["encoder_mask"])
                x += self.dummy_factor * dummy_outs["x"].mean(dim=1).unsqueeze(1)
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        x_feat = x

        layer_results = []
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
        if len(remaining_extractor_names) > 0:
            for name, x_dummy, encoder_mask_dummy in zip(
                remaining_extractor_names, x_dummies, encoder_mask_dummies
            ):
                remaining_extractor = self.modality_encoders[name]
                dummy_out = self.forward_decoder(
                            x_dummy,
                            remaining_extractor,
                            remaining_extractor.decoder,
                            encoder_mask_dummy,
                            )
                dx += self.dummy_factor * dummy_out.mean(dim=1).unsqueeze(1)
                xs[-1] = dx

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

        if self.adversarial_loss > 0:
            self.step_counter += 1
            predictions = self.discriminator(x_feat.mean(dim=1)) # B x D
            for im, m in enumerate(self.modalities):
                if m.name == mode:
                    targets = torch.zeros(predictions.size()[0]).fill_(im).long().to(device=device)
                else:
                    fake_targets = torch.zeros(predictions.size()[0]).fill_(im).long().to(device=device)

            if self.step_counter % self.num_discriminator_steps == 0:
                result["losses"]["discriminator"] = self.adversarial_loss * F.cross_entropy(predictions, targets)
            else:
                result["losses"]["generator"] = self.adversarial_loss * F.cross_entropy(predictions, fake_targets)

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                reg_loss = self.d2v_loss(x, y)
                n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss

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
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x

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