from dataclasses import dataclass
import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import vector_norm

logger = logging.getLogger(__name__)


@dataclass
class RPQConfig:
    input_feature_dim: int = 80 # Dimension of input.
    codebook_vocab: int = 16 # Number of codes
    codebook_dim: int = 8192 # Codebook dimension i.e. embedding size
    encoder_hidden_size: int = 768 # Number of encoder output dimensions


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, config: RPQConfig):
        super().__init__()
        self.random_projection = nn.Linear(
            config.input_feature_dim, config.codebook_dim, bias=False
        )
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.code_book = nn.Parameter(
            torch.randn(config.codebook_vocab, config.codebook_dim)
        )

        self.random_projection.weight.requires_grad = False
        self.code_book.requires_grad = False

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook indexes.
        input_values: T x M x C
        """
        targets = self.random_projection(input_values) # T x M x C -> T x M x D
        vector_distances = vector_norm(targets.unsqueeze(1) - self.code_book.unsqueeze(1), dim=-1)
        labels = torch.argmin(vector_distances, dim=1) # T x B

        return labels