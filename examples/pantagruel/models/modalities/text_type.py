# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.modules import PositionalEmbedding, FairseqDropout, LayerNorm
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


class TextTypeEncoder(PantagruelModalitySpecificEncoder):
    def __init__(
        self,
        modality_cfg: D2vTextConfig,
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

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=text_encoder.local_encoder,
            project_features=text_encoder.project_features,
            fixed_positional_encoder=text_encoder.fixed_positional_encoder,
            relative_positional_encoder=text_encoder.relative_positional_encoder,
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
