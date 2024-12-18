# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import numpy as np
from typing import Optional, Callable, Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)

from examples.data2vec.models.modalities.modules import BlockEncoder, Decoder1d
from examples.data2vec.models.modalities.base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias as get_alibi_bias_fn,
    _learned_alibi_bias,
    MaskSeed, MaskInfo,
)
from examples.pantagruel.data.modality import Modality

logger = logging.getLogger(__name__)


@dataclass
class PantagruelDualModalityConfig(D2vModalityConfig):
    type: Modality = Modality.AUDIO_TEXT
    dropout: float = 0.1
    conv_pos_width: int = field(
        default=95,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": "depth of positional encoder network"},
    )
    conv_pos_pre_ln: bool = False
    use_project_features: bool = False
    mlp_spec: str = field(
        default="{'num_layers': 1, 'mlp_dim': 128}",
        metadata={
            "help": "specs of project feature layers: {'num_layers': 0, 'mlp_dim': 128}"
        },
    )


class PantagruelModalitySpecificEncoder(ModalitySpecificEncoder):
    def __init__(
        self, 
        modality_cfg: D2vModalityConfig, 
        embed_dim: int, 
        local_encoder: nn.Module, 
        project_features: nn.Module, 
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
        get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
        token_type_embeddings: Optional[nn.Module],
    ):
        super().__init__(modality_cfg, embed_dim, local_encoder, project_features, fixed_positional_encoder, relative_positional_encoder, context_encoder, decoder, get_alibi_bias)

        self.token_type_embeddings = token_type_embeddings

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
        token_type_ids=None,
    ):
        x = self.local_features(features) # B x L x D
        if self.token_type_embeddings is not None and token_type_ids is not None:
            # self.token_type_embeddings(token_type_ids): 1 x D
            x = x + self.token_type_embeddings(token_type_ids)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )
    

class PantagruelFusionEncoder(ModalitySpecificEncoder):
    def __init__(
        self,
        modality_cfg: PantagruelDualModalityConfig,
        embed_dim: int,
        local_encoder: nn.ModuleDict,
        project_features: Optional[nn.ModuleDict],
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: Optional[nn.Module],
        decoder: Optional[nn.Module],
        get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
        token_type_embeddings: Optional[nn.Module],
    ):
        super().__init__(modality_cfg, embed_dim, local_encoder, project_features, fixed_positional_encoder, relative_positional_encoder, context_encoder, decoder, get_alibi_bias)

        self.token_type_embeddings = token_type_embeddings
        self.project_features_multi = nn.Identity()
        if getattr(modality_cfg, "use_project_features", False):
            mlp_spec = eval(getattr(modality_cfg, "mlp_spec", None))
            assert mlp_spec is not None
            project_features = self._build_mlp(
                num_layers=mlp_spec['num_layers'],
                input_dim=embed_dim,
                mlp_dim=mlp_spec['mlp_dim'],
                output_dim=embed_dim
            )

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.LayerNorm(dim1))
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.ReLU(inplace=False))
        return nn.Sequential(*mlp)

    @classmethod
    def build_dual_encoders_from_unimodal(
        cls, 
        modality_cfg: PantagruelDualModalityConfig, 
        embed_dim: int,
        unimodal_encoders: nn.ModuleDict,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        token_type_embeddings: Optional[nn.Module],
    ):
        dual_local_encoders = nn.ModuleDict()
        for mod in modality_cfg.type.name.split("_"):
            if mod in unimodal_encoders:
                assert isinstance(unimodal_encoders[mod], PantagruelModalitySpecificEncoder)
                dual_local_encoders[mod] = unimodal_encoders[mod]
            else:
                raise ValueError(f"Modality '{mod}' not found in provided modality_encoders")

        local_encoder = nn.ModuleDict(
            {mod: dual_local_encoders[mod].local_encoder for mod in dual_local_encoders}
        )
        project_features = nn.ModuleDict(
            {mod: dual_local_encoders[mod].project_features for mod in dual_local_encoders}
        )
        fixed_positional_encoder = None

        # build CNN positional encoder
        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)

        relative_positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=modality_cfg.conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )
        if modality_cfg.conv_pos_pre_ln:
            positional_encoder = nn.Sequential(LayerNorm(embed_dim), positional_encoder)

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else None
        )

        alibi_bias_fn = partial(get_alibi_bias_fn, alibi_biases=alibi_biases)

        get_alibi_bias = alibi_bias_fn if modality_cfg.use_alibi_encoder else None

        return cls(
            modality_cfg, embed_dim, local_encoder, project_features,
            fixed_positional_encoder, relative_positional_encoder,
            context_encoder, decoder, get_alibi_bias,
            token_type_embeddings,
        )

    def local_features(self, features):
        fused_x = []
        for mod_name, mod_inputs in features.items():
            x = self.local_encoder[mod_name.upper()](mod_inputs["source"])
            x = self.project_features[mod_name.upper()](x) # BxLxD
            fused_x.append(x)
        return torch.cat(fused_x, dim=1) # Bx(S+T)xD

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
        token_type_ids=None,
    ):
        x = self.local_features(features) # B x L x D
        if self.token_type_embeddings is not None and token_type_ids is not None:
            # self.token_type_embeddings(token_type_ids): 1 x D
            x = x + self.token_type_embeddings(token_type_ids)
        x = self.project_features_multi(x)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )