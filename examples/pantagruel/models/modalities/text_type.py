# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.tasks import FairseqTask
from .base_encoder import D2vModalityConfig, PantagruelModalitySpecificEncoder

from examples.data2vec.models.modalities.text import (
    TextEncoder,
)
from examples.pantagruel.data.modality import Modality


@dataclass
class PantagruelD2vTextConfig(D2vModalityConfig):
    type: Modality = Modality.TEXT
    max_source_positions: int = 512
    learned_pos: bool = True
    dropout: float = 0.1  # used for both local_encoder and contextualized encoder. tied with global transformer in data2vec_text
    no_scale_embedding: bool = True
    layernorm_embedding: bool = True
    no_token_positional_embeddings: bool = False
    use_project_features: bool = False
    use_relative_positional_encoder: bool = False
    disable_embed_positions: bool = True # while use relative positional encoder only
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
    mlp_spec: str = field(
        default="{'num_layers': 1, 'mlp_dim': 128}",
        metadata={
            "help": "specs of project feature layers: {'num_layers': 0, 'mlp_dim': 128}"
        },
    )

class TextTypeEncoder(PantagruelModalitySpecificEncoder):
    def __init__(
        self,
        modality_cfg: PantagruelD2vTextConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
        token_type_embeddings: Optional[nn.Module],
    ):
        text_encoder = TextEncoder(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            make_block=make_block,
            norm_layer=norm_layer,
            layer_norm_first=layer_norm_first,
            alibi_biases=alibi_biases,
            task=task,
            rotary_emb=None,
        )
        project_features = nn.Identity()
        if getattr(modality_cfg, "use_project_features", False):
            mlp_spec = eval(getattr(modality_cfg, "mlp_spec", None))
            assert mlp_spec is not None
            project_features = self._build_mlp(
                num_layers=mlp_spec['num_layers'],
                input_dim=embed_dim,
                mlp_dim=mlp_spec['mlp_dim'],
                output_dim=embed_dim
            )
        positional_encoder = None
        if getattr(modality_cfg, "use_relative_positional_encoder", False):
            k = max(1, modality_cfg.conv_pos_width // modality_cfg.conv_pos_depth)
            positional_encoder = nn.Sequential(
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
                    nn.LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(modality_cfg.conv_pos_depth)
            ],
            TransposeLast(),
        )
            if getattr(modality_cfg, "disable_embed_positions", True):
                text_encoder.local_encoder.embed_positions = None

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=text_encoder.local_encoder,
            project_features=project_features,
            fixed_positional_encoder=text_encoder.fixed_positional_encoder,
            relative_positional_encoder=positional_encoder,
            context_encoder=text_encoder.context_encoder,
            decoder=text_encoder.decoder,
            get_alibi_bias=text_encoder.get_alibi_bias,
            token_type_embeddings=token_type_embeddings,
            rotary_emb=None,
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
    
    def reset_parameters(self):
        super().reset_parameters()

    def convert_padding_mask(self, x, padding_mask):
        if padding_mask is None or padding_mask.size(1) == x.size(1):
            return padding_mask

        diff = self.downsample - padding_mask.size(1) % self.downsample
        if 0 < diff < self.downsample:
            padding_mask = F.pad(padding_mask, (0, diff), value=True)

        padding_mask = padding_mask.view(padding_mask.size(0), -1, self.downsample)
        padding_mask = padding_mask.all(-1)
        if padding_mask.size(1) > x.size(1):
            padding_mask = padding_mask[:, : x.size(1)]

        assert x.size(1) == padding_mask.size(
            1
        ), f"{x.size(1), padding_mask.size(1), diff, self.downsample}"

        return padding_mask
