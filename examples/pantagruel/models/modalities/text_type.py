# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.modules import (
    PositionalEmbedding,
    FairseqDropout,
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.tasks import FairseqTask
from .base_type import D2vModalityConfig, PantagruelModalitySpecificEncoder
from examples.data2vec.models.modalities.base import (
    get_alibi_bias,
)
from examples.data2vec.models.modalities.text import (
    D2vTextConfig,
    TextEncoder,
)
from examples.data2vec.models.modalities.modules import BlockEncoder, Decoder1d
from examples.data2vec.data.modality import Modality
from examples.data2vec.models.modalities.text import (
    D2vTextConfig,
)

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
        logging.info(f"TextEncoder::task: {task}")
        text_encoder = TextEncoder(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            make_block=make_block,
            norm_layer=norm_layer,
            layer_norm_first=layer_norm_first,
            alibi_biases=alibi_biases,
            task=task,
        )
        project_features = nn.Identity()
        if getattr(modality_cfg, "use_project_features", False):
            project_features = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim*3),
                nn.ReLU(),
                nn.Linear(embed_dim*3, embed_dim)
            )
        positional_encoder = None
        if getattr(modality_cfg, "use_relative_positional_encoder", False):
            k = max(3, modality_cfg.conv_pos_width // modality_cfg.conv_pos_depth)
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
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(modality_cfg.conv_pos_depth)
            ],
            TransposeLast(),
        )


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
        )
    
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
