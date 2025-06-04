# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy

from dataclasses import dataclass, field
from functools import partial

import logging
import math
import numpy as np
from omegaconf import II, MISSING

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.file_io import PathManager

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, FairseqDecoder

from examples.pantagruel.data.modality import Modality
from examples.data2vec.data.modality import Modality as Data2vecModality

from examples.data2vec.models.modalities.base import (
    MaskSeed,
    get_annealed_rate,
    D2vModalityConfig,
)
from examples.pantagruel.models.modalities.base_encoder import (
    PantagruelModalitySpecificEncoder,
    PantagruelFusionEncoder,
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
from examples.pantagruel.models.modules import AltBlockWithModalityExpert, MHAPooling
from examples.pantagruel.models.utils import load_all_pretrained_modules_to_model
from examples.pantagruel.models.modalities.mimi_audio_encoder import MimiAudioEncoder

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
        default=0, 
        metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": ("Scale the reconstruction loss by this constant. "
                "If None, then scales by 1/sqrt(dim).")
        }
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

    average_top_k_layers: str = field(
        default="{'audio': 8, 'text': 12}",
        metadata={"help": "how many layers to average for each modality"},
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
    skip_ema: bool = False
    max_update: int = II("optimization.max_update")
    
    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    modalities: PantagruelD2vModalitiesConfig = PantagruelD2vModalitiesConfig()
    supported_modality: Optional[Modality] = None

    mae_init: bool = False
    seed: int = II("common.seed")

    d2v_loss: float = 1.0

    decoder_group: bool = False
    shared_decoder: Optional[D2vDecoderConfig] = None

    use_token_type_embeddings: bool = False
    dummy_factor: float = 0.0
    skip_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "skip_mode"
        },
    )

    do_shallow_fusion: bool = True
    compute_cross_targets: Optional[bool] = False
    compute_cross_preds_for_text: Optional[bool] = False

    use_ctc_module: bool = False
    num_freeze_ctc_updates: int = 0

    extract_encoder_outs: bool = False
    num_freeze_ot_updates: int = 0

    std_coeff: float = 0.0
    cov_coeff: float = 0.0
    d2v_text_coeff: float = 1.0

    pretrained_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to load pretrained weights"}
    )
    skip_pretrained_modules: Optional[str] = field(
        default="",
        metadata={"help": "modules to skip when pre-training, in form of dict[module_name:list(skip_modules)]. use 'none' to not skip any modules, 'all' to skip all modules. e.g. {'modality_encoders': ['none'], 'backbone': ['all']}"} 
    )
    pretrained_path_overlay: Optional[str] = field(
        default=None,
        metadata={"help": "path to load pretrained weights"}
    )
    skip_pretrained_modules_overlay: Optional[str] = field(
        default="",
        metadata={"help": "modules to skip when pre-training, in form of dict[module_name:list(skip_modules)]"}
    )
    # for freezeing all the local encoders and decoders for a number of steps
    # local_grad_mult: is applied to the respective local encoder after num_steps_freeze_local_encoder
    num_steps_freeze_local_encoders: int = 0
    num_steps_freeze_local_decoders: int = 0

    moex_args_ffn: Optional[str] = field(
        default=None,
        metadata={"help": "configuration of the modality experts at FFN module"},
    )
    moex_args_mha: Optional[str] = field(
        default=None,
        metadata={"help": "configuration of the modality experts at MHA module"},
    )
    freeze_backbone: Optional[bool] = False
    freeze_decoder: Optional[bool] = False

    use_map_head_for_speech: Optional[bool] = False
    num_map_heads: Optional[int] = 1
    use_linear_head_for_text: Optional[bool]= False
    num_freeze_sigloss_updates: int = 0

    use_mimi_for_audio: bool = False


class CTCDecoder(nn.Module):
    def __init__(self, dictionary, embed_dim, dropout_rate=0.0, bias=True):
        super().__init__()
        self.blank_idx = 0 # default to be <s>
        self.pad_idx = dictionary.pad()
        self.eos_idx = dictionary.eos()
        self.dropout_module = nn.Dropout(dropout_rate)
        self.ctc_proj = nn.Linear(embed_dim, len(dictionary), bias=bias)
        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")

    def forward(self, x):
        x = self.ctc_proj(self.dropout_module(x)) # BxLxD -> BxLxV
        return x.transpose(0, 1)


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
        use_mimi_for_audio=False,
    ) -> PantagruelModalitySpecificEncoder:
        if cfg.type == Modality.AUDIO:
            if not use_mimi_for_audio:
                enc_cls = AudioTypeEncoder
            else:
                enc_cls = MimiAudioEncoder
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

    def __init__(
        self, cfg: PantagruelData2VecMultiConfig, modalities, skip_ema=False, task=None
    ):
        super().__init__()

        self.cfg = cfg
        self.modalities = modalities
        self.task = task
        self.mask_idx = self.task.mask_idx # not used yet

        self.dummy_factor = getattr(cfg, "dummy_factor", 0.0)
        self.skip_mode = getattr(cfg, "skip_mode", None)
        self.d2v_text_coeff = getattr(cfg, "d2v_text_coeff", 1.0)

        self.do_shallow_fusion = getattr(cfg, "do_shallow_fusion", True)
        self.compute_cross_targets = getattr(cfg, "compute_cross_targets", False)
        self.compute_cross_preds_for_text = getattr(cfg, "compute_cross_preds_for_text", False)

        self.use_ctc_module = getattr(cfg, "use_ctc_module", False)
        self.num_freeze_ctc_updates = getattr(cfg, "num_freeze_ctc_updates", 0)

        self.extract_encoder_outs = getattr(cfg, "extract_encoder_outs", False)
        self.num_freeze_ot_updates = getattr(cfg, "num_freeze_ot_updates", 0)

        self.num_steps_freeze_local_encoders = getattr(cfg, "num_steps_freeze_local_encoders", 0)
        self.num_steps_freeze_local_decoders = getattr(cfg, "num_steps_freeze_local_decoders", 0)

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
            )

        def make_block_moex(drop_path, dim=None, heads=None):
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
                moex_args_ffn=getattr(cfg, "moex_args_ffn", None),
                moex_args_mha=getattr(cfg, "moex_args_mha", None),
                freeze_backbone=getattr(cfg, "freeze_backbone", False),
            )

        token_type_embeddings = None
        self.uni_modalities = [m for m in self.modalities if "_" not in m.name]
        logger.info(f"[Pretraining] self.uni_modalities: {self.uni_modalities}")
        if cfg.use_token_type_embeddings:
            token_type_embeddings = nn.Embedding(
                len(self.uni_modalities), cfg.embed_dim
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        # modalities: use uppercase for modules and lowercase for variables
        for mod in self.modalities:
            if "_" not in mod.name:
                self.alibi_biases[mod.name] = {}
                mod_cfg = getattr(cfg.modalities, mod.name.lower())
                enc = self.make_modality_type_encoder(
                    mod_cfg,
                    cfg.embed_dim,
                    make_block,
                    make_layer_norm,
                    cfg.layer_norm_first,
                    self.alibi_biases[mod.name],
                    task,
                    token_type_embeddings,
                    use_mimi_for_audio=getattr(cfg, "use_mimi_for_audio", False),
                )
                self.modality_encoders[mod.name] = enc
                if getattr(cfg, "freeze_decoder", False):
                    logger.info("freezing freeze_decoder in modality encoders...")
                    for param in enc.decoder.parameters():
                        param.requires_grad = False
            else:
                if not self.do_shallow_fusion:
                    self.modality_encoders[mod.name] = PantagruelFusionEncoder.build_dual_encoders_from_unimodal(
                        getattr(cfg.modalities, mod.name.lower()),
                        cfg.embed_dim,
                        self.modality_encoders,
                        make_block,
                        make_layer_norm,
                        cfg.layer_norm_first,
                        self.alibi_biases,
                        token_type_embeddings,
                    )

        self.average_top_k_layers = eval(cfg.average_top_k_layers)
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block_moex(dpr[i]) for i in range(cfg.depth)])

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

        # build auxiliary components used for the supervised losses during pre-training
        self.aux_heads = None
        use_map_head_for_speech = getattr(cfg, "use_map_head_for_speech", False)
        use_linear_head_for_text = getattr(cfg, "use_linear_head_for_text", False)
        self.num_freeze_sigloss_updates = getattr(cfg, "num_freeze_sigloss_updates", 0)
        if use_map_head_for_speech or use_linear_head_for_text:
            self.aux_heads = nn.ModuleDict()
            if use_map_head_for_speech:
                self.aux_heads["AUDIO"] = MHAPooling(
                    cfg.embed_dim, getattr(cfg, "num_map_heads", 1)
                )
            if use_linear_head_for_text:
                self.aux_heads["TEXT"] = nn.Linear(
                    cfg.embed_dim, cfg.embed_dim
                )

        self.ctc_module = None
        if self.use_ctc_module:
            self.ctc_module = CTCDecoder(
                self.task.source_dictionary,
                self.cfg.embed_dim,
            )

        # init using pretrained models
        def parse_skip_modules(cfg_attr):
            skip = getattr(cfg, cfg_attr, "")
            if skip:
                skip = eval(skip)
                for k, v in skip.items():
                    if v[0].lower() == "none":
                        skip[k] = []
            return skip

        def load_pretrained_if_available(
            path_attr, skip_attr, modules,
        ):
            path = getattr(cfg, path_attr, None)
            logger.info(f"{path_attr}: {path}")
            if path:
                skip_modules = parse_skip_modules(skip_attr)
                logger.info(f"{skip_attr}: {skip_modules}")
                load_modules(
                    modules, path, skip_modules, 
                )

        def load_modules(
            modules, pretrained_path, skip_modules,
        ):
            for name, module in modules.items():
                load_all_pretrained_modules_to_model(
                    module, pretrained_path, skip_modules[name],
                )
        
        # Collect modules to load
        modules_to_load_pretrained = {
            'modality_encoders': self.modality_encoders, 'backbone': self.blocks
        }
        # if self.ctc_module:
        #     modules_to_load_pretrained['ctc_module'] = self.ctc_module

        # Load initial and overlay pretrained weights if applicable
        load_pretrained_if_available(
            "pretrained_path", "skip_pretrained_modules", 
            modules_to_load_pretrained,
        )
        load_pretrained_if_available(
            "pretrained_path_overlay", "skip_pretrained_modules_overlay", 
            modules_to_load_pretrained,
        )
        # freeze the pre-trained modules if specified
        self._update_status_local_encoders = False
        self._update_status_local_decoders = False
        if self.num_steps_freeze_local_encoders > 0 or self.num_steps_freeze_local_decoders > 0:
            for name, module in self.modality_encoders.items():
                for param_name, param in module.named_parameters():
                    is_decoder_param = "decoder" in param_name

                    if (
                        is_decoder_param and 
                        self.num_steps_freeze_local_decoders > 0
                    ):
                        logger.info(f"Freezing {param_name}: {param.shape}")
                        param.requires_grad = False

                    elif (
                        not is_decoder_param and 
                        self.num_steps_freeze_local_encoders > 0
                    ):
                        logger.info(f"Freezing {param_name}: {param.shape}")
                        param.requires_grad = False

        self.ema = None
        if not skip_ema:
            logger.info("Initializing EMA teacher model")
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

        ema_cfg = copy.deepcopy(self.cfg)
        if hasattr(ema_cfg, "use_map_head_for_speech"):
            ema_cfg.use_map_head_for_speech = False
        if hasattr(ema_cfg, "use_linear_head_for_text"):
            ema_cfg.use_linear_head_for_text = False
        if hasattr(ema_cfg, "use_ctc_module"):
            ema_cfg.use_ctc_module = False

        model_copy = PantagruelMultiModel(
            ema_cfg, self.modalities, skip_ema=True, task=self.task
        )

        if ema_cfg.ema_encoder_only:
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
    
    def _get_feature_extractor(self, mode):
        extractor = self.modality_encoders
        remaining_extractor_names = []
        # TODO: do_shallow_fusion=False
        if mode in self.modality_encoders:
            extractor = self.modality_encoders[mode]
            remaining_extractor_names = [
                m.name for m in self.modalities 
                if m.name != mode and m.name in self.modality_encoders.keys()
            ]
        return extractor, remaining_extractor_names

    def forward_ctc(self, extractor_out, padding_mask):
        ctc_out = {}
        ft = self.num_freeze_ctc_updates <= self.num_updates
        audio_extractor = self.modality_encoders["AUDIO"]
        
        with torch.no_grad() if not ft else contextlib.ExitStack():
            _speech_enc_out = audio_extractor.contextualized_features(
                extractor_out["local_features"]["audio"],
                padding_mask,
                mask=False,
                remove_masked=False,
            )
            _x_speech = _speech_enc_out["x"] # BxLxD
            
            # Forward through Transformer blocks
            for i, blk in enumerate(self.blocks):
                _alibi_bias = _speech_enc_out.get("alibi_bias", None)
                _alibi_scale = _speech_enc_out.get("alibi_scale", None)
                
                if _alibi_bias is not None and _alibi_scale is not None:
                    scale = (
                        _alibi_scale[i]
                        if _alibi_scale.size(0) > 1
                        else _alibi_scale.squeeze(0)
                    )
                    _alibi_bias = _alibi_bias * scale.type_as(_alibi_bias)

                _x_speech, _ = blk(
                    _x_speech,
                    padding_mask=_speech_enc_out["padding_mask"],
                    alibi_bias=_alibi_bias,
                    mode="AUDIO"
                )
            ctc_out["x"] = self.ctc_module(_x_speech)
            ctc_out["padding_mask"] = _speech_enc_out["padding_mask"]
            ctc_out["is_frozen"] = not ft
            ctc_out["_x_speech"] = _x_speech

        return ctc_out

    def forward_audio_map(self, extractor_out, padding_mask, ctc_out):
        ft = self.num_freeze_sigloss_updates <= self.num_updates

        if ctc_out and ft != ctc_out["is_frozen"]:
            _x_speech = ctc_out["_x_speech"]
            padding_mask = ctc_out["padding_mask"]
        else:
            # forward
            audio_extractor = self.modality_encoders["AUDIO"]
            with torch.no_grad() if not ft else contextlib.ExitStack():
                _speech_enc_out = audio_extractor.contextualized_features(
                    extractor_out["local_features"]["audio"],
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                _x_speech = _speech_enc_out["x"] # BxLxD
                
                # Forward through Transformer blocks
                for i, blk in enumerate(self.blocks):
                    _alibi_bias = _speech_enc_out.get("alibi_bias", None)
                    _alibi_scale = _speech_enc_out.get("alibi_scale", None)
                    
                    if _alibi_bias is not None and _alibi_scale is not None:
                        scale = (
                            _alibi_scale[i]
                            if _alibi_scale.size(0) > 1
                            else _alibi_scale.squeeze(0)
                        )
                        _alibi_bias = _alibi_bias * scale.type_as(_alibi_bias)

                    _x_speech, _ = blk(
                        _x_speech,
                        padding_mask=_speech_enc_out["padding_mask"],
                        alibi_bias=_alibi_bias,
                        mode="AUDIO"
                    )
                padding_mask = _speech_enc_out["padding_mask"]
        
        # forward to MAP head
        with torch.no_grad() if not ft else contextlib.ExitStack():
            _aux_head_audio = self.aux_heads["AUDIO"](
                        _x_speech, padding_mask=padding_mask
                    ) # B x D

        return _aux_head_audio

    def forward_dual_encoder(
        self, extractor_out, source, mode, ctc_out
    ):
        dual_encoder_outs = {}
        ft = self.num_freeze_ot_updates <= self.num_updates
        _modes = mode.split("_")
        if ctc_out and ft != ctc_out["is_frozen"]:
            dual_encoder_outs["audio"] = ctc_out["_x_speech"]
            _modes = ["TEXT"]

        with torch.no_grad() if not ft else contextlib.ExitStack():
            for _mod in _modes:
                _extractor = self.modality_encoders[_mod]
                _extractor_out_mod = _extractor.contextualized_features(
                    extractor_out["local_features"][_mod.lower()],
                    source[_mod.lower()]["padding_mask"],
                    mask=False,
                    remove_masked=False,
                )
                _x_mod = _extractor_out_mod["x"]
                for i, blk in enumerate(self.blocks):
                    _alibi_bias = _extractor_out_mod.get("alibi_bias", None)
                    _alibi_scale = _extractor_out_mod.get("alibi_scale", None)
                    if _alibi_bias is not None and _alibi_scale is not None:
                        scale = (
                            _alibi_scale[i]
                            if _alibi_scale.size(0) > 1
                            else _alibi_scale.squeeze(0)
                        )
                        _alibi_bias = _alibi_bias * scale.type_as(_alibi_bias)

                    _x_mod, _ = blk(
                        _x_mod,
                        padding_mask=_extractor_out_mod["padding_mask"],
                        alibi_bias=_alibi_bias,
                        mode=_mod
                    )
                dual_encoder_outs[_mod.lower()] = _x_mod
        dual_encoder_outs["is_frozen"] = not ft
        return dual_encoder_outs

    def forward_text_head(self, extractor_out, padding_mask, dual_encoder_outs):
        ft = self.num_freeze_sigloss_updates <= self.num_updates

        if dual_encoder_outs and ft != dual_encoder_outs["is_frozen"]:
            _x_text = dual_encoder_outs["text"]
        else:
            with torch.no_grad() if not ft else contextlib.ExitStack():
                text_extractor = self.modality_encoders["TEXT"]
                _text_extractor_out = text_extractor.contextualized_features(
                    extractor_out["local_features"]["text"],
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                _x_text = _text_extractor_out["x"]
                for i, blk in enumerate(self.blocks):
                    _alibi_bias = _text_extractor_out.get("alibi_bias", None)
                    _alibi_scale = _text_extractor_out.get("alibi_scale", None)
                    if _alibi_bias is not None and _alibi_scale is not None:
                        scale = (
                            _alibi_scale[i]
                            if _alibi_scale.size(0) > 1
                            else _alibi_scale.squeeze(0)
                        )
                        _alibi_bias = _alibi_bias * scale.type_as(_alibi_bias)

                    _x_text, _ = blk(
                        _x_text,
                        padding_mask=_text_extractor_out["padding_mask"],
                        alibi_bias=_alibi_bias,
                        mode="TEXT"
                    )
        # forward to MAP head
        with torch.no_grad() if not ft else contextlib.ExitStack():
            _aux_head_text = self.aux_heads["TEXT"](_x_text[:,0,:]) # B x D

        return _aux_head_text

    def update_freeze_status(self):
        """
        Enable gradients for encoder/decoder parameters in modality_encoders
        based on the current number of updates and configured freeze durations.
        """
        for name, module in self.modality_encoders.items():
            for param_name, param in module.named_parameters():
                is_decoder_param = "decoder" in param_name

                if is_decoder_param and self.num_updates >= self.num_steps_freeze_local_decoders:
                    logger.info(f"Unfreezing {param_name}: {param.shape}")
                    param.requires_grad = True
                    self._update_status_local_decoders = True
                elif not is_decoder_param and self.num_updates >= self.num_steps_freeze_local_encoders:
                    logger.info(f"Unfreezing {param_name}: {param.shape}")
                    param.requires_grad = True
                    self._update_status_local_encoders = True

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

        if isinstance(mode, Modality) or isinstance(mode, Data2vecModality):
            mode = mode.name

        should_unfreeze_encoders = (
            self.num_steps_freeze_local_encoders > 0 and 
            self.num_updates >= self.num_steps_freeze_local_encoders and
            not self._update_status_local_encoders
        )
        should_unfreeze_decoder = (
            self.num_steps_freeze_local_decoders > 0 and 
            self.num_updates >= self.num_steps_freeze_local_decoders and
            not self._update_status_local_decoders
        )

        if should_unfreeze_encoders or should_unfreeze_decoder:
            self.update_freeze_status()

        extractor, remaining_extractor_names = self._get_feature_extractor(mode)
        device = source.device if isinstance(source, torch.Tensor) else source["audio"]["source"].device

        token_type_ids = {}
        for it, im in enumerate(self.uni_modalities):
            token_type_ids[im.name.lower()] = torch.ones((1), dtype=torch.int64, device=device) * it

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        B, _ = source.size() if isinstance(source, torch.Tensor) else source["audio"]["source"].size()
        extractor_out = {
                "x": {},
                "local_features": {},
                "encoder_mask": {},
                "alibi_bias": {}, "alibi_scale": {},
                "encoder_mask": {},
                "padding_mask": {}
            }
        current_modes = [mode.lower()] if "_" not in mode else list(source.keys())
        for _mod in current_modes:
            _extractor_out_mod = self.modality_encoders[_mod.upper()](
                (
                    source if isinstance(source, torch.Tensor) 
                    else source[_mod]["source"]
                ),
                (
                    padding_mask if isinstance(source, torch.Tensor) 
                    else source[_mod]["padding_mask"]
                ),
                mask,
                remove_masked=not features_only or force_remove_masked,
                clone_batch=self.cfg.clone_batch if not features_only else 1,
                mask_seeds=mask_seeds,
                precomputed_mask=(
                    precomputed_mask if isinstance(source, torch.Tensor)
                    else source[_mod]["precomputed_mask"]
                ),
                token_type_ids=token_type_ids[_mod],
            ) # BxLxD
            for k, v in extractor_out.items():
                extractor_out[k][_mod] = _extractor_out_mod[k]

        x = extractor_out["x"] # {mod: M x T x C (M = B*clone_batch)}
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out["alibi_bias"]
        alibi_scale = extractor_out["alibi_scale"]

        x_dummies, encoder_mask_dummies = None, None
        if len(remaining_extractor_names) > 0:
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
            x_dummies, encoder_mask_dummies = [], []
            for name in remaining_extractor_names:
                dummy = dummy_source_audio if name == "AUDIO" else dummy_source_text
                dummy_outs = self.modality_encoders[name](
                    dummy, None, False, False, token_type_ids=token_type_ids[name.lower()]
                )
                _x_dummy = dummy_outs["x"].repeat_interleave(B * self.cfg.clone_batch,0) #1xTxC -> M x T x C
                x_dummies.append(_x_dummy)
                encoder_mask_dummies.append(dummy_outs["encoder_mask"])
                for _mod, _x in x.items():
                    x[_mod] = _x + self.dummy_factor * _x_dummy.mean(dim=1).unsqueeze(1)

        if self.dropout_input is not None:
            for _mod, _x in x.items():
                x[_mod] = self.dropout_input(_x)
        
        layer_results = {_mod: [] for _mod in current_modes}
        for _mod in current_modes:
            _x = x[_mod]
            for i, blk in enumerate(self.blocks):
                if (
                    not self.training
                    or self.cfg.layerdrop == 0
                    or (np.random.random() > self.cfg.layerdrop)
                ):
                        ab = masked_alibi_bias[_mod]
                        if ab is not None and alibi_scale[_mod] is not None:
                            scale = (
                                alibi_scale[_mod][i]
                                if alibi_scale[_mod].size(0) > 1
                                else alibi_scale[_mod].squeeze(0)
                            )
                            ab = ab * scale.type_as(ab)

                        _x, lr = blk(
                            _x,
                            padding_mask=masked_padding_mask[_mod],
                            alibi_bias=ab,
                            mode=_mod.upper(),
                        )
                        if features_only:
                            layer_results[_mod].append(lr)
            x[_mod] = _x

        if self.norm:
            for _mod, _x in x.items():
                x[_mod] = self.norm(_x)

        ctc_out, dual_encoder_outs = {}, {}
        aux_heads_out = {}
        if mode == "AUDIO_TEXT":
            if self.ctc_module is not None:
                ctc_out = self.forward_ctc(
                    extractor_out, source["audio"]["padding_mask"]
                )

            if self.extract_encoder_outs:
                dual_encoder_outs = self.forward_dual_encoder(
                    extractor_out, source, mode, ctc_out
                )

            if self.aux_heads is not None:
                aux_heads_out["audio"] = self.forward_audio_map(
                    extractor_out, source["audio"]["padding_mask"], ctc_out
                )
                aux_heads_out["text"] = self.forward_text_head(
                    extractor_out,  source["text"]["padding_mask"], dual_encoder_outs
                )
                aux_heads_out["is_frozen"] = (self.num_updates <= self.num_freeze_sigloss_updates)

        if features_only:
            if remove_extra_tokens:
                for _mod, _x in x.items():
                    _extractor = self.modality_encoders[_mod.upper()]
                    x[_mod] = _x[:, _extractor.modality_cfg.num_extra_tokens :]
                    if masked_padding_mask[_mod] is not None:
                        masked_padding_mask[_mod] = masked_padding_mask[_mod][
                            :, _extractor.modality_cfg.num_extra_tokens :
                        ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        xs = {_mod: [] for _mod in current_modes}
        for _mod in current_modes:
            _extractor = self.modality_encoders[_mod.upper()]
            if self.shared_decoder is not None:
                dx = self.forward_decoder(
                    x[_mod],
                    _extractor,
                    self.shared_decoder,
                    encoder_mask[_mod],
                )
                xs[_mod].append(dx)
            if _extractor.decoder is not None:
                dx = self.forward_decoder(
                    x[_mod],
                    _extractor,
                    _extractor.decoder,
                    encoder_mask[_mod]
                )
                xs[_mod].append(dx)

        if len(remaining_extractor_names) > 0:
            for name, x_dummy, encoder_mask_dummy in zip(
                remaining_extractor_names, x_dummies, encoder_mask_dummies
            ):
                remaining_extractor = self.modality_encoders[name]
                if remaining_extractor.decoder is not None:
                    dummy_out = self.forward_decoder(
                                x_dummy,
                                remaining_extractor,
                                remaining_extractor.decoder,
                                encoder_mask_dummy,
                                )
                    dx += self.dummy_factor * dummy_out.mean(dim=1).unsqueeze(1)
                    xs[_mod][-1] = dx

        assert all(len(xs[_mod]) > 0 for _mod in current_modes)

        p = next(self.ema.model.parameters())
        device = x[current_modes[0]].device
        dtype = x[current_modes[0]].dtype
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
                ema_input = {
                    "x": {}, "padding_mask": {}, "alibi_bias": {}, "alibi_scale": {}
                }
                _ema_input = extractor_out["local_features"]
                for _mod in current_modes:
                    _extractor = self.modality_encoders[_mod.upper()]
                    _ema_input_mod = _extractor.contextualized_features(
                        _ema_input[_mod].to(dtype=ema_dtype),
                        extractor_out["padding_mask"][_mod],
                        mask=False,
                        remove_masked=False,
                    )
                    for k in ema_input.keys():
                        ema_input[k][_mod] = _ema_input_mod[k]
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                ema_input = {
                    "x": {}, "padding_mask": {}, "alibi_bias": {}, "alibi_scale": {}
                }
                for _mod in current_modes:
                    _extractor = self.modality_encoders[_mod.upper()]
                    _source = (
                        source if isinstance(source, torch.Tensor) 
                        else source[_mod]["source"]
                    )
                    _padding_mask = (
                        padding_mask if isinstance(source, torch.Tensor) 
                        else source[_mod]["padding_mask"]
                    )
                    if _extractor.modality_cfg.ema_local_encoder:
                        inp = (
                            target.to(dtype=ema_dtype)
                            if target is not None
                            else _source.to(dtype=ema_dtype)
                        )
                        _ema_input = tm.modality_encoders[_mod.upper()](
                            inp.to(dtype=torch.int64) if _mod=="text" else inp,
                            _padding_mask,
                            mask=False,
                            remove_masked=False,
                        )
                    else:
                        assert target is None
                        _ema_input = extractor_out["local_features"][_mod]
                        ema_feature_enc = tm.modality_encoders[_mod.upper()]
                        _ema_input = ema_feature_enc.contextualized_features(
                            _ema_input.to(dtype=ema_dtype),
                            _padding_mask,
                            mask=False,
                            remove_masked=False,
                        )
                    for k in ema_input.keys():
                        ema_input[k][_mod] = _ema_input[k]

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input["alibi_bias"]
            ema_alibi_scale = ema_input["alibi_scale"]
            ema_input = ema_input["x"]

            y = {_mod: [] for _mod in current_modes}
            extra_tokens = {_mod: [] for _mod in current_modes}
            for _mod in current_modes:
                _extractor = self.modality_encoders[_mod.upper()]
                extra_tokens[_mod] = _extractor.modality_cfg.num_extra_tokens
                _ema_input_mod = ema_input[_mod]
                for i, blk in enumerate(ema_blocks):
                    ab = ema_alibi_bias[_mod]
                    if ab is not None and alibi_scale[_mod] is not None:
                        scale = (
                            ema_alibi_scale[_mod][i]
                            if ema_alibi_scale[_mod].size(0) > 1
                            else ema_alibi_scale[_mod].squeeze(0)
                        )
                        ab = ab * scale.type_as(ab)

                    _ema_input_mod, lr = blk(
                        _ema_input_mod,
                        padding_mask=ema_padding_mask[_mod],
                        alibi_bias=ab,
                        mode=_mod.upper(),
                    )
                    y[_mod].append(lr[:, extra_tokens[_mod] : ])

                y[_mod] = self.make_targets(y[_mod], self.average_top_k_layers[_mod])
            
            if len(current_modes) == 2 and self.compute_cross_targets:
                # represent one modality using the other modality
                y_new = {}
                for i in range(2):
                    query = y[current_modes[i]]
                    key = value = y[current_modes[1-i]]
                    attn_scores = F.softmax(
                        query @ key.transpose(1, 2), dim=-1, dtype=torch.float32
                    )
                    y_new[current_modes[i]] = attn_scores @ value
                y = y_new

        if self.cfg.clone_batch > 1:
            for _mod in current_modes:
                y[_mod] = y[_mod].repeat_interleave(self.cfg.clone_batch, 0)

        if len(current_modes) == 2 and self.compute_cross_preds_for_text:
            assert all(len(xs[_mod])==1 for _mod in current_modes)
            _audio = xs["audio"][0].clone()
            xs["text"][0] = F.softmax(torch.matmul(
                xs["text"][0], _audio.transpose(1, 2)
            ), dim=-1) @ _audio

        masked, masked_b, sample_sizes = {}, {}, {}
        for _mod in current_modes:
            masked[_mod] = encoder_mask[_mod].mask.unsqueeze(-1)
            masked_b[_mod] = encoder_mask[_mod].mask.bool()
            y[_mod] = y[_mod][masked_b[_mod]]

            if xs[_mod][0].size(1) == masked_b[_mod].size(1):
                xs[_mod] = [x[masked_b[_mod]] for x in xs[_mod]]
            else:
                xs[_mod] = [x.reshape(-1, x.size(-1)) for x in xs[_mod]]

            sample_sizes[_mod] = masked[_mod].sum().long()
        
        result = {
            "losses": {},
            "sample_size": sample_sizes,
            "ctc_out": ctc_out,
            "dual_encoders_out": dual_encoder_outs,
            "aux_heads_out": aux_heads_out,
        }

        if self.cfg.d2v_loss > 0:
            for _mod in current_modes:
                for i, x in enumerate(xs[_mod]):
                    reg_loss = self.d2v_loss(x, y[_mod]) # x: TxD, y: TxD, reg_loss: TxD
                    n = (
                        f"{_mod.upper()}_regression_{i}" 
                        if len(xs[_mod]) > 1 
                        else f"{_mod.upper()}_regression"
                    )
                    d2v_loss_scale = self.cfg.d2v_loss
                    if _mod.upper() == "TEXT":
                        d2v_loss_scale *= self.d2v_text_coeff
                    result["losses"][n] = reg_loss * d2v_loss_scale
                    if getattr(self.cfg, "std_coeff", 0.0) > 0.0 or getattr(self.cfg, "cov_coeff", 0.0) > 0.0:
                        var_cov_loss = self.var_cov_loss(x, y[_mod])
                        result["losses"][n] += var_cov_loss

        with torch.no_grad():
            for _mod in current_modes:
                if encoder_mask[_mod] is not None:
                    result[f"masked_pct_{_mod.upper()}"] = 1 - (
                        encoder_mask[_mod].ids_keep.size(1) / 
                        encoder_mask[_mod].ids_restore.size(1)
                        )

                result[f"target_var_{_mod.upper()}"] = self.compute_var(y[_mod].float())
                for i, x in enumerate(xs[_mod]):
                    result[f"pred_var_{_mod.upper()}"] = self.compute_var(x.float())

            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            if self.num_updates > 5000:
                for k in result.keys():
                    if k.startswith("target_var") and result[k] < self.cfg.min_target_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                        )
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
        if self.ctc_module is not None:
            self.ctc_module = None

        modality = modality.upper() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if not keep_decoder:
                self.modality_encoders[k].decoder = None

        if modality:
            self.keep_modules_by_name(modality)

    def keep_modules_by_name(self, kept_modality):
        modalities_to_remove = [mod for mod in list(self.modality_encoders.keys()) if mod != kept_modality.upper()]
        logger.info(f"modalities_to_remove: {modalities_to_remove}")

        for _mod_to_remove in modalities_to_remove:
            to_delete = []
            for name, module in list(self.named_modules()):
                if _mod_to_remove in name:
                    to_delete.append(name)

            for name in to_delete:
                parts = name.split(".")
                parent_name = ".".join(parts[:-1])  # Parent module
                child_name = parts[-1]  # Module to delete

                if parent_name:
                    parent_module = dict(self.named_modules()).get(parent_name, None)
                    if (
                        parent_module and hasattr(parent_module, "_modules") and child_name in parent_module._modules
                    ):
                        del parent_module._modules[child_name]
                else:  # If it's a top-level module
                    if child_name in self._modules:
                        del self._modules[child_name]

    def merge_modality_experts(self, modality=None):
        if self.use_modality_experts_at_mha:
            for i, blk in enumerate(self.blocks):
                logging.info(f'block {i} before merged: {torch.norm(blk.attn.qkv.weight.data)}')
                blk.attn.qkv.weight.data += blk.attn.modality_experts_qkv[modality.upper()].B.weight @ blk.attn.modality_experts_qkv[modality.upper()].A.weight
                logging.info(f'block {i} after merged: {torch.norm(blk.attn.qkv.weight.data)}')
                self.blocks[i].attn.modality_experts_qkv = None