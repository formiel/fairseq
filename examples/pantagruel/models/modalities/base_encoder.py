# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch
import torch.nn as nn

from typing import Optional, Callable

from examples.data2vec.models.modalities.base import (
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
        x = self.local_features(features) # B x L x D
        if self.token_type_embeddings is not None:
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
    

class PantagruelDualModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        dual_modality_names: str,
        modality_encoders: nn.ModuleDict,
    ):
        super().__init__()
        self.dual_modality_encoders = nn.ModuleDict()
        for mod in dual_modality_names.split("_"):
            if mod in modality_encoders:
                self.dual_modality_encoders[mod] = modality_encoders[mod]
            else:
                raise ValueError(f"Modality '{mod}' not found in provided modality_encoders")
            
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
        pass