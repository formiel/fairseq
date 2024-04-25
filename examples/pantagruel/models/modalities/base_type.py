# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from omegaconf import MISSING, II
from typing import Optional, Callable
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules import GradMultiply
from fairseq.utils import index_put
from examples.data2vec.data.modality import Modality
from examples.data2vec.models.modalities.modules import D2vDecoderConfig
from examples.data2vec.models.modalities.base import (
    MaskSeed, MaskInfo,
    D2vModalityConfig,
    ModalitySpecificEncoder,
)

logger = logging.getLogger(__name__)


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
        x = self.local_features(features)
        # logger.info(f'x: {x.size()}, norm: {torch.linalg.matrix_norm(x)}')
        if self.token_type_embeddings is not None:
            # logger.info(f'token_type_ids: {token_type_ids} size {self.token_type_embeddings(token_type_ids).size()}')
            x += self.token_type_embeddings(token_type_ids).unsqueeze(1)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

