# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import wraps
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from fairseq.modules import (
    LayerNorm,
    SamePad,
    SamePad2d,
    TransposeLast,
)

logger = logging.getLogger(__name__)


@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 5
    decoder_layers: int = 5
    input_dropout: float = 0.1

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


class FixedPositionalEncoder(nn.Module):
    def __init__(self, pos_embed):
        super().__init__()
        self.positions = pos_embed

    def forward(self, x, padding_mask):
        return self.positions


class TextFeatPositionalEncoder(nn.Module):
    """
    Original encoder expects (B, T) long input. This module wraps it to take
    local_encoder output which are (B, T, D) float tensors
    """

    def __init__(self, pos_encoder):
        super().__init__()
        self.pos_encoder = pos_encoder

    def forward(self, x, padding_mask):
        # assume padded token embeddings are 0s
        # TODO: consider using padding_mask as input
        return self.pos_encoder(x[..., 0])


class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x


class DecoderBase(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig):
        super().__init__()

        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if (
            residual is None
            or not self.decoder_cfg.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret


class Decoder1d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__(cfg)

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad(cfg.decoder_kernel),
                TransposeLast(),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        projs = []
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask_info):

        x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class Decoder2d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim, h_size, w_size):
        super().__init__(cfg)

        self.h_size = h_size
        self.w_size = w_size

        def make_block(in_dim):
            block = [
                nn.Conv2d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad2d(cfg.decoder_kernel),
                TransposeLast(tranpose_dim=-3),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def forward(self, x, mask_info):
        B, T, C = x.shape

        x = x.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.reshape(B, -1, T).transpose(1, 2)
        x = self.proj(x)
        return x


class TransformerDecoder(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig, input_dim, encoder):
        super().__init__()

        self.decoder_cfg = cfg

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.encoder = encoder

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, mask_info):
        x = self.input_proj(x)
        x = self.encoder(x, None, None, 1)
        x = self.proj(x)
        return x


class AltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        cosine_attention=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(
        self, x, padding_mask=None, alibi_bias=None, position_embeddings=None
    ):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(
                self.norm1(x), padding_mask, alibi_bias, 
                position_embeddings=position_embeddings
            ))
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(self.attn(
                x, padding_mask, alibi_bias, 
                position_embeddings=position_embeddings
            ))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x

        return x, t


class AltAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop = proj_drop

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(
        self, x, padding_mask=None, alibi_bias=None, position_embeddings=None, fast=True
    ):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if position_embeddings is not None:
            cos, sin = position_embeddings # (M, unmasked_L, head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos, sin) # (M, heads, unmasked_L, head_dim)

        if not fast:
            if self.cosine_attention:
                # cosine attention
                attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
                ).exp()
                attn = attn * logit_scale
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1) # B x C//H x L x L

            if alibi_bias is not None:
                attn = attn.type_as(alibi_bias)
                attn[:, : alibi_bias.size(1)] += alibi_bias

            if padding_mask is not None and padding_mask.any():
                attn = attn.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )

            attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
            # attn = self.attn_drop(attn)
            attn = F.dropout(attn, p=self.attn_drop if self.training else 0.0)
            x = (attn @ v).transpose(1, 2)
        else:
            # Using pytorch 2's sdpa
            assert not self.cosine_attention, "Not support cosine attention yet"
            # Integrate padding_mask and alibi_bias
            if padding_mask is not None and padding_mask.any():
                if alibi_bias is not None:
                    padding_mask = alibi_bias.masked_fill(
                            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        ).to(dtype=dtype)
                else:
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(
                        torch.bool).to(dtype=dtype)
            else:
                if alibi_bias is not None:
                    padding_mask = alibi_bias.to(dtype=dtype)
                else:
                    padding_mask = None
            x = F.scaled_dot_product_attention(q, k, v, 
                                attn_mask=padding_mask, 
                                dropout_p=self.attn_drop if self.training else 0.0,
                                scale=self.scale).transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        x = F.dropout(x, p=self.proj_drop if self.training else 0.0)
        return x


class EncDecAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        kv_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(q_dim, q_dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop = proj_drop

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(self, q, kv, padding_mask=None, alibi_bias=None, fast=True):
        B, N, C = q.shape

        q = (
            self.q_proj(q)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B x H x L x D
        kv = (
            self.kv_proj(kv)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # kv x B x H x L x D
        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if not fast:
            if self.cosine_attention:
                # cosine attention
                attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
                ).exp()
                attn = attn * logit_scale
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)

            if alibi_bias is not None:
                attn = attn.type_as(alibi_bias)
                attn[:, : alibi_bias.size(1)] += alibi_bias

            if padding_mask is not None and padding_mask.any():
                attn = attn.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )

            attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
            # attn = self.attn_drop(attn)
            attn = F.dropout(attn, p=self.attn_drop if self.training else 0.0)
            x = (attn @ v).transpose(1, 2)  #
        else:
            assert not self.cosine_attention, "Not support cosine attention yet"
            # Integrate padding_mask and alibi_bias
            if padding_mask is not None and padding_mask.any():
                if alibi_bias is not None:
                    padding_mask = alibi_bias.masked_fill(
                            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        ).to(dtype=dtype)
                else:
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(
                        torch.bool).to(dtype=dtype)
            else:
                if alibi_bias is not None:
                    padding_mask = alibi_bias.to(dtype=dtype)
                else:
                    padding_mask = None
            x = F.scaled_dot_product_attention(q, k, v, 
                                attn_mask=padding_mask, 
                                dropout_p=self.attn_drop if self.training else 0.0,
                                scale=self.scale).transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        x = F.dropout(x, p=self.proj_drop if self.training else 0.0)
        return x


class EncDecBlock(nn.Module):
    def __init__(
        self,
        q_dim,
        kv_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        cosine_attention=False,
        first_residual=True,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(q_dim)
        self.attn = EncDecAttention(
            q_dim,
            kv_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=q_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)
        self.first_residual = first_residual

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        r = q if self.first_residual else 0
        if self.layer_norm_first:
            x = r + self.drop_path(
                self.attn(self.norm1(q), kv, padding_mask, alibi_bias)
            )
            r = x = self.mlp(self.norm2(x))
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = r + self.drop_path(self.attn(q, kv, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x


class EncDecTransformerDecoder(nn.Module):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.blocks = nn.Sequential(
            *[
                EncDecBlock(
                    q_dim=cfg.decoder_dim,
                    kv_dim=input_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    mlp_drop=0.0,
                    post_mlp_drop=0.0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    layer_norm_first=False,
                    cosine_attention=False,
                    first_residual=i > 0,
                )
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, kv):
        x = self.input_proj(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x, kv)

        x = self.proj(x)
        return x


def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable):
            The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
    """

    def longrope_frequency_update(self, position_ids, device):
        """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
        seq_len = torch.max(position_ids) + 1
        if hasattr(self.config, "original_max_position_embeddings"):
            original_max_position_embeddings = self.config.original_max_position_embeddings
        else:
            original_max_position_embeddings = self.config.max_position_embeddings
        if seq_len > original_max_position_embeddings:
            if not hasattr(self, "long_inv_freq"):
                self.long_inv_freq, _ = self.rope_init_fn(
                    self.config, device, seq_len=original_max_position_embeddings + 1
                )
            self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

    def dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        elif self.rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper


# modified from HuggingFace
class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _compute_default_rope_parameters(
    config,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.embed_dim // config.num_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)