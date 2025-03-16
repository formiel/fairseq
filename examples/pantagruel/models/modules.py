from functools import partial
import importlib

import logging
import math

from timm.models.vision_transformer import DropPath

import torch
import torch.version
import torch.nn as nn
import torch.nn.functional as F


def is_flash_attn_2_available():
    return importlib.util.find_spec("flash_attn") is not None

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


logger = logging.getLogger(__name__)


# copied from timm.layers.helpers
def _ntuple(n):
    import collections.abc
    from itertools import repeat
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class ModalityExpert(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, rank: int, 
        alpha: float, ln: bool,
    ):
        super().__init__()
        self.scaling = alpha / rank

        self.moex_A = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        nn.init.kaiming_uniform_(self.moex_A.weight, a=math.sqrt(5))

        self.moex_B = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        nn.init.zeros_(self.moex_B.weight)
        
        self.moex_ln = None
        if ln:
            self.moex_ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        # W = self.moex_A @ self.moex_B
        # x =  x @ W
        x = self.scaling * self.moex_B(self.moex_A(x))
        if self.moex_ln:
            x = self.moex_ln(x)
        return x


# modified from timm.models.vision_transformer.MLP
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            moex_args=None,
            dummy_factor=0.0,
            freeze_backbone=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        def make_moex_modules(in_dim, out_dim, moex_args):
            return nn.ModuleDict(
                    {
                        _mod: ModalityExpert(
                            in_dim=in_dim, out_dim=out_dim,
                            rank=moex_args["rank"], alpha=moex_args["alpha"],
                            ln=moex_args["use_ln"]
                        ) for _mod in moex_args["modalities"]
                    }
                )
        
        self.dummy_factor = dummy_factor
        self.modalities = moex_args.get("modalities", None) if moex_args else None
        self.moex_fc1, self.moex_fc2 = None, None
        if moex_args:
            # creating moex modules for each modality
            self.moex_fc1 = make_moex_modules(in_features, hidden_features, moex_args)
            self.moex_fc2 = make_moex_modules(hidden_features, out_features, moex_args)
            if freeze_backbone:
                logger.info("freezing the backbone: MLP layer")
                # do not freeze layernorm
                for param in self.fc1.parameters():
                    param.requires_grad = False
                for param in self.fc2.parameters():
                    param.requires_grad = False

    def get_remaining_experts(self, mode):
        remaining_modes = []
        if mode:
            remaining_modes = list(set(self.modalities) - set([mode]))
        return remaining_modes

    def apply_experts(self, x, mode, experts_modules, remaining_modes):
        x_in = x
        x = experts_modules[mode](x)
        for _mod in remaining_modes:
            if _mod in experts_modules:
                x += self.dummy_factor * experts_modules[_mod](x_in.mean(dim=1)).unsqueeze(1)
        return x

    def forward(self, x, mode=None):
        remaining_modes = self.get_remaining_experts(mode)

        # First linear layer and modality expert processing (if any)
        x_in = x
        x = self.fc1(x)
        if mode and self.moex_fc1:
            x = x + self.apply_experts(x_in, mode, self.moex_fc1, remaining_modes)

        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)

        # Second linear layer and modality expert processing (if any)
        x_in = x
        x = self.fc2(x)
        if mode and self.moex_fc2:
            x = x + self.apply_experts(x_in, mode, self.moex_fc2, remaining_modes)

        x = self.drop2(x)
        return x

class AltBlockWithModalityExpert(nn.Module):
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
        dummy_factor=0.0,
        moex_args_ffn=None,
        moex_args_mha=None,
        freeze_backbone=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.moex_args_ffn = eval(moex_args_ffn) if moex_args_ffn else None
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
            moex_args=self.moex_args_ffn,
            dummy_factor=dummy_factor,
            freeze_backbone=freeze_backbone,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

        self.dummy_factor = dummy_factor

        self.moex_args_mha = eval(moex_args_mha) if moex_args_mha else None
        self.attn = AltAttentionWithExperts(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
            moex_args=self.moex_args_mha,
            dummy_factor=dummy_factor,
            freeze_backbone=freeze_backbone,
        )
    
    def forward(self, x, padding_mask=None, alibi_bias=None, mode=None):
        if self.layer_norm_first:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x), padding_mask, alibi_bias, 
                    mode=mode if self.moex_args_mha else None
                )
            )
            r = x = self.mlp(self.norm2(x), mode=mode if self.moex_args_ffn else None)
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(
                self.attn(x, padding_mask, alibi_bias, 
                mode=mode if self.moex_args_mha else None,
                )
            )
            r = x = self.norm1(x)
            x = self.mlp(x, mode=mode if self.moex_args_ffn else None)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x

        return x, t
        

class AltAttentionWithExperts(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
        moex_args=None,
        dummy_factor=0.0,
        freeze_backbone=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = proj_drop

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        # Modality-specific experts
        self.dummy_factor = dummy_factor
        self.modalities = moex_args.get("modalities", None) if moex_args else None
        self.moex_qkv, self.moex_proj = None, None

        def make_moex_modules(in_dim, out_dim, moex_args):
            return nn.ModuleDict(
                    {
                        _mod: ModalityExpert(
                            in_dim=in_dim, out_dim=out_dim,
                            rank=moex_args["rank"], alpha=moex_args["alpha"],
                            ln=moex_args["use_ln"]
                        ) for _mod in moex_args["modalities"]
                    }
                )

        if moex_args:
            self.moex_qkv = make_moex_modules(dim, dim*3, moex_args)
            self.moex_proj = make_moex_modules(dim, dim, moex_args)
            if freeze_backbone:
                logger.info("freezing the backbone: MHA module")
                for param in self.qkv.parameters():
                    param.requires_grad = False
                for param in self.proj.parameters():
                    param.requires_grad = False

    def get_remaining_experts(self, mode):
        remaining_modes = []
        if mode:
            remaining_modes = list(set(self.modalities) - set([mode]))
        return remaining_modes

    def apply_experts(self, x, mode, experts_modules, remaining_modes=None):
        x_in = x
        x = experts_modules[mode](x)
        for _mod in remaining_modes:
            if _mod in experts_modules:
                x += self.dummy_factor * experts_modules[_mod](x_in.mean(dim=1)).unsqueeze(1)
        return x

    def forward(self, x, padding_mask=None, alibi_bias=None, fast=True, mode=None):
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

        # modality experts
        remaining_modes = self.get_remaining_experts(mode)
        if self.moex_qkv:
            qkv_experts = self.apply_experts(
                x, mode, self.moex_qkv, remaining_modes=remaining_modes
            )
            qkv_experts = (
                qkv_experts.reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q = q + qkv_experts[0]
            k = k + qkv_experts[1]
            v = v + qkv_experts[2]

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
            attn = F.dropout(attn, p=self.attn_drop)
            x = (attn @ v).transpose(1, 2)
        else:
            # Using pytorch 2's sdpa for CUDA and FlashAttention2 for ROCm
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
            # logger.info(f"padding_mask:{padding_mask.size()}\n{padding_mask}")
            # if torch.version.cuda is not None: # Using pytorch 2's sdpa for CUDA
            x = F.scaled_dot_product_attention(q, k, v, 
                                attn_mask=padding_mask, 
                                dropout_p=self.attn_drop if self.training else 0.0,
                                scale=self.scale).transpose(1, 2)
            # elif torch.version.hip is not None:
            #     # logger.info(f"using FlashAttention2 backend...")
            #     # using FlashAttention2:
            #     assert is_flash_attn_2_available()
            #     q = q.permute(0, 2, 1, 3) # BxHxLxD -> BxLxHxD
            #     k = k.permute(0, 2, 1, 3)
            #     v = v.permute(0, 2, 1, 3)
            #     x = self._flash_attention_forward(q, k, v,
            #                                         attention_mask=padding_mask,
            #                                         query_length=N,
            #                                         dropout=self.attn_drop if self.training else 0.0)

        x = x.reshape(B, N, C)
        x_in = x
        x = self.proj(x)
        # x = self.proj_drop(x)
        if self.moex_proj:
            x = x + self.apply_experts(
                x_in, mode, self.moex_proj, remaining_modes=remaining_modes
            )
        x = F.dropout(x, p=self.proj_drop if self.training else 0.0)
        return x
    
    # based on transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, causal=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.size()[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output
    
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.size()

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # logger.info(f"seqlens_in_batch: {seqlens_in_batch}")
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
