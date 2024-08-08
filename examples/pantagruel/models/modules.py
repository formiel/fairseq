
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from examples.data2vec.models.modalities.modules import (
    AltBlock,
)

class ModalityExpert(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

class AltBlockWithModalityExpert(AltBlock):
    def __init__(
        self, dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop=0,
        attn_drop=0,
        mlp_drop=0,
        post_mlp_drop=0,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        cosine_attention=False,
        modalities=None,
        dummy_factor=0.0,

    ):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, mlp_drop, post_mlp_drop, drop_path, act_layer, norm_layer, layer_norm_first, ffn_targets, cosine_attention)

        self.dummy_factor = dummy_factor

        self.modalities = modalities
        self.modality_experts = None
        if self.modalities is not None:
            self.modality_experts = nn.ModuleDict()
            for mod in self.modalities:
                self.modality_experts[mod.name] = ModalityExpert(dim, dim, dim//2)
    
    def forward(self, x, padding_mask=None, alibi_bias=None, mode=None):
        if self.modality_experts is None:
            return super().forward(x, padding_mask, alibi_bias)
        else:
            remaining_extractor_names = [m.name for m in self.modalities if m.name != mode 
                                     and m.name in self.modality_experts.keys()]

            if self.layer_norm_first:
                x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
                x = self.norm2(x)
                x_modality = x
                r = x = self.mlp(x)
                x += self.modality_experts[mode](x_modality)
                for name in remaining_extractor_names:
                    x += self.dummy_factor * self.modality_experts[name](x_modality)
                t = x
                x = r + self.drop_path(self.post_mlp_dropout(x))
                if not self.ffn_targets:
                    t = x
            else:
                x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
                r = x = self.norm1(x)
                x_modality = x
                x = self.mlp(x)
                x += self.modality_experts[mode](x_modality)
                for name in remaining_extractor_names:
                    x += self.dummy_factor * self.modality_experts[name](x_modality)
                t = x
                x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
                if not self.ffn_targets:
                    t = x

            return x, t