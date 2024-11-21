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

from examples.pantagruel.data.modality import Modality

from examples.data2vec.models.modalities.base import (
    MaskSeed,
    get_annealed_rate,
    D2vModalityConfig,
)
from examples.pantagruel.models.modalities.base_encoder import (
    PantagruelModalitySpecificEncoder,
    PantagruelDualModalityConfig,
)
from examples.data2vec.models.modalities.modules import (
    D2vDecoderConfig,
    Decoder1d,
)

from examples.pantagruel.models.modalities.audio_type import (
    AudioTypeEncoder,
    PantagruelD2vAudioConfig,
)
from examples.data2vec.models.modalities.images import (
    D2vImageConfig,
    ImageEncoder,
)
from examples.pantagruel.models.modalities.text_type import (
    TextTypeEncoder,
    PantagruelD2vTextConfig,
)
from fairseq.modules import PositionalEmbedding
from examples.data2vec.models.modalities.base import MaskInfo
from .modules import AltBlockWithModalityExpert


logger = logging.getLogger(__name__)


@dataclass
class PantagruelD2vModalitiesConfig(FairseqDataclass):
    audio: PantagruelD2vAudioConfig = PantagruelD2vAudioConfig()
    image: D2vImageConfig = D2vImageConfig()
    text: PantagruelD2vTextConfig = PantagruelD2vTextConfig()
    audio_text: PantagruelDualModalityConfig = PantagruelDualModalityConfig()


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

    d2v_loss: float = 1

    decoder_group: bool = False

    use_token_type_embeddings: bool = False
    
    dummy_factor: float = 0.0
    skip_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "skip_mode"
        },
    )
    use_modality_experts_at_ffn: bool = False
    use_modality_experts_at_mha: bool = False
    modality_expert_rank: int = 0
    freeze_project_features: bool = False

    contrastive_loss_with_feature_decoder: bool = False
    feature_decoder_embed_dim: int = 384
    feature_decoder: Optional[D2vDecoderConfig] = D2vDecoderConfig()

    start_step_train_aux_loss: int = field(
        default=0,
        metadata={"help": "number of training steps to start training auxiliary loss"}
    )
    layer_norm_before_encoder: bool = False
    ema_modality_combiner: bool = False


class FeatureProjector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels*2, bias=True),
            nn.SyncBatchNorm(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels*2, in_channels//3, bias=False),
        )
    
    def forward(self, x):
        x = self.ffn(x)
        return x


class FeatureHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels*6, bias=True),
            nn.SyncBatchNorm(in_channels*6),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels*6, in_channels, bias=False),
        )
    
    def forward(self, x):
        x = self.ffn(x)
        return x


class FeatureDecoder(nn.Module):
    def __init__(self, cfg: PantagruelData2VecMultiConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.decoder = Decoder1d(cfg.feature_decoder, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.predictor = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)

    def forward(self, x, feature_extractor, maskinfo):
        x = self.embed(x)
        x = feature_extractor.decoder_input(x, maskinfo)
        x = self.decoder(*x)
        x = self.norm(x)
        x = self.predictor(x)
        return x


@register_model("pantagruel_model", dataclass=PantagruelData2VecMultiConfig)
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
        self.mask_idx = self.task.mask_idx

        self.dummy_factor = getattr(cfg, "dummy_factor", 0.0)
        self.skip_mode = getattr(cfg, "skip_mode", None)

        self.use_modality_experts_at_ffn = getattr(cfg, "use_modality_experts_at_ffn", False)
        self.use_modality_experts_at_mha = getattr(cfg, "use_modality_experts_at_mha", False)

        self.freeze_project_features = getattr(cfg, "freeze_project_features", False)

        self.contrastive_loss_after_encoder = getattr(cfg, "contrastive_loss_after_encoder", False)
        self.contrastive_loss_with_feature_decoder = getattr(cfg, "contrastive_loss_with_feature_decoder", False)

        self.start_step_train_aux_loss = getattr(cfg, "start_step_train_aux_loss", 0)

        self.num_updates = 0

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlockWithModalityExpert(
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
                dummy_factor=self.dummy_factor,
                modality_expert_rank=getattr(cfg, "modality_expert_rank", 0),
                modality_experts_at_ffn=(
                    self.modalities if self.use_modality_experts_at_ffn else None
                ),
                modality_experts_at_mha=(
                    self.modalities if self.use_modality_experts_at_mha else None
                ),
            )

        token_type_embeddings = None
        if cfg.use_token_type_embeddings:
            token_type_embeddings = nn.Embedding(len(self.modalities), cfg.embed_dim)
            nn.init.kaiming_normal_(token_type_embeddings.weight)

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            if "_" not in mod.name: # only build local encoder for single-modality
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
                if self.freeze_project_features:
                    logger.info(f'Freezeing {enc.__class__.__name__}: '
                                f'{enc.project_features.__class__.__name__}')
                    for _, m in enumerate(enc.project_features):
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight)
                            nn.init.normal_(m.bias, mean=0.0, std=0.02)
                        m.requires_grad_(False)
                self.modality_encoders[mod.name] = enc
            else:
                logger.info(f"{mod} -> use modality-specific encoders")

        # use layernorm before Transformer encoder
        self.norm_before_enc = None
        if cfg.layer_norm_before_encoder:
            self.norm_before_enc = nn.ModuleDict(
                {mod.name: make_layer_norm(cfg.embed_dim) for mod in self.modalities if "_" not in mod.name}
            ) 

        # modules to handle aligned data
        aligned_modality = [mod for mod in self.modalities if "_" in mod.name]
        align_mod = aligned_modality[0].name if len(aligned_modality) > 0 else None
        logger.info(f'aligned modality: {align_mod}')
        
        self.modality_combiner = None if not align_mod else nn.ModuleDict()
        if self.modality_combiner is not None:
            self.modality_combiner["embed"] = nn.ParameterDict(
                {_m: nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) 
                 for _m in align_mod.split("_")}
            )
            self.modality_combiner["fusion"] = make_block(drop_path=0) 
            self.modality_combiner["norm"] = nn.LayerNorm(cfg.embed_dim)
            self.modality_combiner["decoder_embed"] = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=True)
            self.modality_combiner["decoder_pos_embed"] = nn.ParameterDict(
                {_m: PositionalEmbedding(1512, cfg.embed_dim, 0)
                 for _m in align_mod.split("_")
                }
            )
        
        # Contrastive loss
        self.feature_decoder = None
        self.feature_proj, self.feature_head = None, None
        if self.contrastive_loss_with_feature_decoder:
            self.feature_decoder = FeatureDecoder(cfg)
            self.feature_proj = FeatureProjector(cfg.embed_dim)
            self.feature_head = FeatureHead(cfg.embed_dim//3)
            if self.num_updates < self.start_step_train_aux_loss:
                logger.info('Freezing feature_decoder, feature_proj, and feature_head...')
                self._freeze_modules(self.feature_decoder)
                self._freeze_modules(self.feature_proj)
                self._freeze_modules(self.feature_head)
            
        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = nn.ModuleDict(
                {mod.name: make_layer_norm(cfg.embed_dim) for mod in self.modalities if "_" not in mod.name}
            )

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

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        self.num_updates = 0
    
    def _freeze_modules(self, modules):
        for param in modules.parameters():
            param.requires_grad = False

    def _unfreeze_modules(self, modules):
        if all(not param.requires_grad for param in modules.parameters()):
            for param in modules.parameters():
                param.requires_grad = True

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
            model_copy.feature_decoder = None
            model_copy.feature_head = None

        if self.contrastive_loss_after_encoder:
            model_copy.predictor_mlp = None

        if not self.cfg.ema_modality_combiner and model_copy.modality_combiner:
            model_copy.modality_combiner = None

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
        state = super().state_dict(destination=destination, 
                                    prefix=prefix, 
                                    keep_vars=keep_vars)

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
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                    Modality.AUDIO_TEXT,
                ]
            )
        else:
            modalities = (task.supported_modalities 
                if cfg.supported_modality is None
                else [cfg.supported_modality]
            )
        
        # random training for the modalities provided in skip_mode for sanity check
        if cfg.skip_mode is not None:
            if "AUDIO" in cfg.skip_mode:
                modalities.append(Modality.AUDIO)
            if "IMAGE" in cfg.skip_mode:
                modalities.append(Modality.IMAGE)
            if "TEXT" in cfg.skip_mode:
                modalities.append(Modality.TEXT)

        logger.info(f"modalities supported by model: {modalities}")

        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)
    
    def _get_feature_extractors(self, mode):
        mode_list = mode.split("_")
        unimodal = True if len(mode_list) == 1 else False
        extractor = self.modality_encoders[mode] if unimodal else None
        remaining_extractor_names = [
            m.name for m in self.modalities 
            if m.name != mode and 
            m.name not in mode_list and
            m.name in self.modality_encoders.keys()
        ]
        return extractor, remaining_extractor_names
    
    def _get_values_from_extractor_out(self, extractor_out, key):
        return {
            _m: extractor_out[_m].get(key, None) for _m in extractor_out.keys()
        }

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
        source_aug=None,
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        # get relevant feature extractor
        feature_extractor, remaining_extractor_names = self._get_feature_extractors(mode)
        device = source.device if isinstance(source, torch.Tensor) else source["audio"].device
        do_multimodal = (self.modality_combiner is not None and "_" in mode)
        if do_multimodal: # use combiner of unimodal encoders
            assert not feature_extractor

        if isinstance(target, torch.Tensor) or target is None:
            target = {mode.lower(): target}
        if do_multimodal:
            target = {_m.lower(): None for _m in mode.split("_")}
        
        token_type_ids = None
        remaining_token_type_ids = {}
        for it, im in enumerate(self.modalities):
            if im.name == mode:
                token_type_ids = torch.ones((1), dtype=torch.int64, device=device) * it
            else:
                remaining_token_type_ids[im.name] = torch.ones((1), dtype=torch.int64, device=device) * it

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        B, _ = source.size() if isinstance(source, torch.Tensor) else source["audio"].size()
        extractor_out = {}
        if feature_extractor is not None: # unimodal input
            extractor_out[mode.lower()] = feature_extractor(
                source, # B x T
                padding_mask,
                mask,
                remove_masked=not features_only or force_remove_masked,
                clone_batch=self.cfg.clone_batch if not features_only else 1,
                mask_seeds=mask_seeds,
                precomputed_mask=precomputed_mask,
                token_type_ids=token_type_ids,
            )
            source = {mode.lower(): source}
            padding_mask = {mode.lower(): padding_mask}
        else:
            # for multimodal inputs: forward to each modality-specific encoders
            for _m, _s in source.items():
                _s = self.modality_encoders[_m.upper()].local_features(_s)
                _s = _s + self.modality_combiner["embed"][_m.upper()]
                extractor_out[_m] = self.modality_encoders[_m.upper()].contextualized_features(
                    _s,
                    padding_mask[_m],
                    mask,
                    remove_masked=not features_only or force_remove_masked,
                    clone_batch=self.cfg.clone_batch if not features_only else 1,
                    mask_seeds=mask_seeds,
                    precomputed_mask=precomputed_mask[_m] if _m in precomputed_mask else None,
                )

        x = self._get_values_from_extractor_out(extractor_out, "x") # M x T x C (M=B*clone_batch)
        # logger.info(f"[[{mode}]] after extractor: {[(k, v.size()) for k, v in x.items()]}")

        x_dummies, encoder_mask_dummies = {}, {}
        if len(remaining_extractor_names) > 0:
            # modality: TEXT, source dtype: torch.int64
            # modality: AUDIO, source dtype: torch.float16
            dummy_source_text = torch.randint(
                self.task.vocab_size - 1, 
                (1, self.task.tokens_per_sample), #(B, self.task.tokens_per_sample)
                dtype=torch.int64, 
                device=device
            )
            dummy_source_audio = torch.randn(
                (1, self.task.max_sample_size), #(B, self.task.max_sample_size), 
                dtype=torch.float16, 
                device=device
            )
            # forward dummy inputs
            for name in remaining_extractor_names:
                dummy = dummy_source_audio if name == "AUDIO" else dummy_source_text
                dummy_outs = self.modality_encoders[name](
                    dummy, None, False, False, token_type_ids=remaining_token_type_ids[name]
                )
                _x_dummy = dummy_outs["x"].repeat_interleave(B * self.cfg.clone_batch,0) #1xTxC -> M x T x C
                x_dummies[name.lower()] = _x_dummy
                encoder_mask_dummies[name.lower()] = dummy_outs["encoder_mask"]
                for _m, _x in x.items():
                    x[_m] = _x + self.dummy_factor * _x_dummy.mean(dim=1).unsqueeze(1)
        
        encoder_mask = self._get_values_from_extractor_out(extractor_out, "encoder_mask")
        masked_padding_mask = self._get_values_from_extractor_out(extractor_out, "padding_mask")
        masked_alibi_bias = self._get_values_from_extractor_out(extractor_out, "alibi_bias")
        alibi_scale = self._get_values_from_extractor_out(extractor_out, "alibi_scale")

        if self.dropout_input is not None:
            x = {_m: self.dropout_input(_x) for _m, _x in x.items()}

        if self.norm_before_enc:
            x = {_m: self.norm_before_enc[_m.upper()](_x) for _m, _x in x.items()}

        layer_results = {_m: [] for _m in x.keys()}
        for _m, _x in x.items():
            for i, blk in enumerate(self.blocks):
                if (
                    not self.training
                    or self.cfg.layerdrop == 0
                    or (np.random.random() > self.cfg.layerdrop)
                ):
                    ab = masked_alibi_bias[_m]
                    if ab is not None and alibi_scale[_m] is not None:
                        scale = (
                            alibi_scale[_m][i]
                            if alibi_scale[_m].size(0) > 1
                            else alibi_scale[_m].squeeze(0)
                        )
                        ab = ab * scale.type_as(ab)

                    _x, lr = blk(
                        _x,
                        padding_mask=masked_padding_mask[_m],
                        alibi_bias=ab,
                    )
                    if features_only:
                        layer_results[_m].append(lr)
            x[_m] = _x
        # logger.info(f"[[{mode}]] after Transformer: {[(k, v.size()) for k, v in x.items()]}")
        
        if self.norm:
            x = {_m: self.norm[_m.upper()](_x) for _m, _x in x.items()}

        if features_only:
            if remove_extra_tokens:
                x = {
                    _m: _x[:, feature_extractor.modality_cfg.num_extra_tokens :] 
                    for _m, _x in x.items()
                }
                masked_padding_mask = {
                    _m: masked_padding_mask[_m][
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ] if not masked_padding_mask[_m] else None for _m in x.keys()
                }

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            } # each value is a dict[mod: torch.Tensor] where mod is unimodal

        # concat for multimodal input
        x_combined, _padding_mask_combined = None, None
        if do_multimodal:
            x_combined = torch.cat(
                [x[_m.lower()] for _m in mode.split("_")], dim=1
            )
            if all(masked_padding_mask[_m.lower()] is not None for _m in mode.split("_")):
                _padding_mask_combined = torch.cat(
                    [masked_padding_mask[_m.lower()] for _m in mode.split("_")],
                    dim=1
                )
                # logger.info(f"_padding_mask_combined: {_padding_mask_combined.size()}")
            x_combined, _ = self.modality_combiner["fusion"](
                x_combined,
                padding_mask=_padding_mask_combined,
            )
            x_combined = self.modality_combiner["norm"](x_combined)
            x_combined = {mode.lower(): self.modality_combiner["decoder_embed"](x_combined)}

        # forward to decoder
        x_decoder = x if not x_combined else x_combined
        xs = {_m: [] for _m in x_decoder.keys()}
        for _m, _x in x_decoder.items():
            if "_" in _m:
                dx, encoder_mask_merged = self.forward_shared_decoder(
                    _x, x, encoder_mask,
                )
                xs[_m].append(dx)
                encoder_mask_merged = {_m: encoder_mask_merged}
            else:
                if feature_extractor.decoder is not None:
                    dx = self.forward_decoder(
                        _x,
                        self.modality_encoders[_m.upper()],
                        self.modality_encoders[_m.upper()].decoder,
                        encoder_mask[_m],
                    )
                    xs[_m].append(dx)
            if len(remaining_extractor_names) > 0:
                for _r in remaining_extractor_names:
                    remaining_extractor = self.modality_encoders[_r]
                    if remaining_extractor.decoder is not None:
                        dummy_out = self.forward_decoder(
                                    x_dummies[_r.lower()],
                                    remaining_extractor,
                                    remaining_extractor.decoder,
                                    encoder_mask_dummies[_r.lower()],
                                    )
                        xs[_m][-1] = dx + self.dummy_factor * dummy_out.mean(dim=1).unsqueeze(1)

        assert all(len(_xs) > 0 for _xs in xs.values())

        # # compute inputs for contrastive loss
        # proj_s = None
        # if self.feature_decoder is not None:
        #     x_s = self.feature_decoder(x, feature_extractor, encoder_mask)
        #     proj_s = self.feature_proj(torch.mean(x_s, dim=1))
        #     proj_s = self.feature_head(proj_s)

        # forward to teacher
        p = next(self.ema.model.parameters())
        device = x[mode.split("_")[0].lower()].device
        dtype = x[mode.split("_")[0].lower()].dtype
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

        ema_padding_mask = {}
        ema_alibi_bias, ema_alibi_scale = {}, {}
        ema_input = {}
        y, ema_x = {}, {}
        with torch.no_grad():
            tm.eval()

            for _m, _extractor_out in extractor_out.items():
                ema_blocks = tm.blocks
                assert target[_m] is None
                ema_input[_m] = _extractor_out["local_features"]
                ema_feature_enc = tm.modality_encoders[_m.upper()]
                ema_input[_m] = ema_feature_enc.contextualized_features(
                    ema_input[_m].to(dtype=ema_dtype),
                    padding_mask[_m],
                    mask=False,
                    remove_masked=False,
                )
                if self.norm_before_enc:
                    ema_input[_m]["x"] = tm.norm_before_enc[_m.upper()](ema_input[_m]["x"])

                ema_padding_mask[_m] = ema_input[_m]["padding_mask"]
                ema_alibi_bias[_m] = ema_input[_m].get("alibi_bias", None)
                ema_alibi_scale[_m] = ema_input[_m].get("alibi_scale", None)
                ema_input[_m] = ema_input[_m]["x"]

                y[_m] = []
                ema_x[_m] = []
                _ema_input_m = ema_input[_m]
                for i, blk in enumerate(ema_blocks):
                    ab = ema_alibi_bias[_m]
                    if ab is not None and alibi_scale[_m] is not None:
                        scale = (
                            ema_alibi_scale[_m][i]
                            if ema_alibi_scale[_m].size(0) > 1
                            else ema_alibi_scale[_m].squeeze(0)
                        )
                        ab = ab * scale.type_as(ab)

                    _ema_input_m, lr = blk(
                        _ema_input_m,
                        padding_mask=ema_padding_mask[_m],
                        alibi_bias=ab,
                        mode=(
                            mode if self.use_modality_experts_at_ffn or self.use_modality_experts_at_mha else None
                        ),
                    )
                    y[_m].append(lr[:, :])
                    ema_x[_m].append(_ema_input_m[:, :])
                ema_input[_m] = _ema_input_m
            # add concatenation
            _ema_x_combined, lr_combined = None, None
            if tm.modality_combiner and "_" in mode:
                _ema_input_combined = torch.cat(
                        [ema_input[_m.lower()] for _m in mode.split("_")], dim=1
                        )
                _ema_padding_mask_combined, lr_combined = None, None
                if all(ema_padding_mask[_m.lower()] is not None for _m in mode.split("_")):
                    _ema_padding_mask_combined = torch.cat(
                        [ema_padding_mask[_m.lower()] for _m in mode.split("_")],
                        dim=1
                    )
                _ema_x_combined, lr_combined = tm.modality_combiner["fusion"](
                    _ema_input_combined, padding_mask=_ema_padding_mask_combined)
                _ema_x_combined = tm.modality_combiner["norm"](_ema_x_combined)
                _ema_x_combined = {mode.lower(): tm.modality_combiner["decoder_embed"](_ema_x_combined)}
                i = 0
                for _m, _e in ema_input.items():
                    # logger.info(f'[[{_m}]]: {_e.size()}, lr_combined: {lr_combined.size()}')
                    _len_m = _e.size()[1]
                    y[_m].append(
                        lr_combined[:, :_len_m, :] if i==0 else lr_combined[:, _len_prev:]
                    )
                    _len_prev = _len_m
                    i += 1

        # if proj_t is not None:
        #     proj_t = tm.feature_proj(torch.mean(proj_t, dim=1))

        y = {
            _m: self.make_targets(y[_m], self.average_top_k_layers) for _m in y.keys()
        }
        if self.cfg.clone_batch > 1:
            y = {
                _m: y[_m].repeat_interleave(self.cfg.clone_batch, 0) for _m in y.keys()
            }
        
        masked = {_m: encoder_mask[_m].mask.unsqueeze(-1) for _m in y.keys()}
        masked_b = {_m: encoder_mask[_m].mask.bool() for _m in y.keys()}
        y = {_m: y[_m][masked_b[_m]] for _m in y.keys()}
        if do_multimodal:
            masked = {mode.lower(): torch.cat([masked[_m] for _m in y.keys()], dim=1)}
            masked_b = {mode.lower(): torch.cat([masked_b[_m] for _m in y.keys()], dim=1)}
            y = {mode.lower(): torch.cat([y[_m] for _m in y.keys()], dim=0)}
        # logger.info(f"[[{mode}]] make_targets: {[(k, v.size()) for k, v in y.items()]}")
        # logger.info(f"[[{mode}]] masked: {[(k, v.size()) for k, v in masked.items()]}")
        # logger.info(f"[[{mode}]] masked_b: {[(k, v.size()) for k, v in masked_b.items()]}")
        # logger.info(f"[[{mode}]] y_masked: {[(k, v.size()) for k, v in y.items()]}")

        for _m, _xsm in xs.items():
            # logger.info(f"{_m}: len={len(_xsm)}, {[_x.size() for _x in _xsm]}")
            if _xsm[0].size(1) == masked_b[_m].size(1):
                xs[_m] = [_x[masked_b[_m]] for _x in _xsm]
            else:
                xs[_m] = [_x.reshape(-1, _x.size(-1)) for _x in _xsm]

        sample_size = {
            _m: masked[_m].sum().long() for _m in xs.keys()
        }
        sample_size = sum(sample_size.values()) / len(sample_size)
        
        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        if self.cfg.d2v_loss > 0:
            for _m, _xsm in xs.items():
                for i, x in enumerate(_xsm):
                    reg_loss = self.d2v_loss(x, y[_m]) # x: TxD, y: TxD, reg_loss: TxD
                    n = f"{_m}_regression_{i}" if len(_xsm) > 1 else f"{_m}_regression"
                    result["losses"][n] = reg_loss * self.cfg.d2v_loss

        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            for _m, _xsm in xs.items():
                suffix = suffix if len(suffix) == 0 else f"{suffix}_{_m}"
                _encoder_mask = encoder_mask[_m] if not do_multimodal else encoder_mask_merged[_m]
                if _encoder_mask is not None:
                    result[f"masked_pct_{_m}"] = 1 - (
                        _encoder_mask.ids_keep.size(1) / _encoder_mask.ids_restore.size(1)
                    )
                for i, x in enumerate(_xsm):
                    n = f"pred_var{suffix}_{i}" if len(_xsm) > 1 else f"pred_var{suffix}"
                    result[n] = self.compute_var(x.float())
                if self.ema is not None:
                    for k, v in self.ema.logs.items():
                        result[k] = v

                y[_m] = y[_m].float()
                result[f"target_var{suffix}"] = self.compute_var(y[_m])

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
    
    def forward_shared_decoder(self, x_combined, xs_mod, mask_infos, mask_noise_std=0.01):
        # get the full concat sequence  
        x_combined_full = []
        x_fulls = {}
        masks, ids_restores, ids_keeps = [], [], []
        i = 0
        for _m, mask_info in mask_infos.items():
            x = xs_mod[_m] # encoded visible tokens
            B, L, D = x.size()
            # logger.info(f'x_combined: {x_combined.size()} type {type(x_combined)}')
            # logger.info(f'[{_m}] x: {x.size()}')
            num_masked = mask_info.ids_restore.shape[1] - L
            # logger.info(f"{_m}: num masked {num_masked}")
            # logger.info(f'mask_info.ids_restore: {mask_info.ids_restore.size()}')
            mask_tokens = x.new_empty(B, num_masked, D).normal_(0, mask_noise_std)
            # logger.info(f'mask_tokens: {mask_tokens.size()}')
            x_full = torch.cat(
                [x_combined[:, :L, :], mask_tokens], dim=1
            ) if i == 0 else torch.cat(
                [x_combined[:, L_prev:, :], mask_tokens], dim=1
            )
            L_prev = L
            # logger.info(f'x_full: {x_full.size()}')
            x_full = torch.gather(x_full, dim=1, index=mask_info.ids_restore)
            # logger.info(f'x_full reorganized: {x_full.size()}')
            x_combined_full.append(x_full)
            x_fulls[_m] = x_full
            
            # logger.info(f"[{_m}]: x_unmasked:{mask_info.x_unmasked.size()}, mask:{mask_info.mask.size()}, ids_restore:{mask_info.ids_restore.size()}, ids_keep:{mask_info.ids_keep.size()}")
            masks.append(mask_info.mask)
            ids_restore, ids_keep = mask_info.ids_restore, mask_info.ids_keep
            ids_restore = ids_restore + L_prev if i == 1 else ids_restore
            ids_restores.append(ids_restore)
            ids_keep = ids_keep + L_prev if i == 1 else ids_keep
            ids_keeps.append(ids_keep)
            i += 1

        x_combined_full = torch.cat(x_combined_full, dim=1)
        decoder_pos = torch.cat(
            [self.modality_combiner["decoder_pos_embed"][
                _m.upper()](torch.ones(x_fulls[_m].size()[:2], device=x_fulls[_m].device)) for _m in x_fulls.keys()],
            dim=1,
        )
        # logger.info(f'x_combined_full: {x_combined_full.size()}, decoder_pos: {decoder_pos.size()}')
        x_combined_full = x_combined_full + decoder_pos
        mask_info_merged = MaskInfo(
            x_unmasked=x_combined,
            mask=torch.cat(masks, dim=1),
            ids_restore=torch.cat(ids_restores, dim=1),
            ids_keep=torch.cat(ids_keeps, dim=1),
        )
        x = self.shared_decoder(x=x_combined_full, mask_info=mask_info_merged)
        return x, mask_info_merged
    
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

    def merge_modality_experts(self, modality=None):
        if self.use_modality_experts_at_mha:
            for i, blk in enumerate(self.blocks):
                logging.info(f'block {i} before merged: {torch.norm(blk.attn.qkv.weight.data)}')
                blk.attn.qkv.weight.data += blk.attn.modality_experts_qkv[modality.upper()].B.weight @ blk.attn.modality_experts_qkv[modality.upper()].A.weight
                logging.info(f'block {i} after merged: {torch.norm(blk.attn.qkv.weight.data)}')
                self.blocks[i].attn.modality_experts_qkv = None