import logging
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath, Mlp
import torch.version


def is_flash_attn_2_available():
    return importlib.util.find_spec("flash_attn") is not None

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


logger = logging.getLogger(__name__)


class ModalityExpert(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        # self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.A = nn.Linear(in_features=in_dim,
                                 out_features=rank,
                                 bias=False)
        self.A.weight.data.normal_(mean=0.0, std=std_dev)
        self.B = nn.Linear(in_features=rank,
                                 out_features=out_dim,
                                 bias=False)
        self.B.weight.data.fill_(0.0)
        self.alpha = alpha

    def forward(self, x):
        # x = self.alpha * (x @ self.A @ self.B)
        x = self.alpha * self.B(self.A(x))
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
        modality_expert_rank=0,
        modality_experts_at_ffn=None,
        modality_experts_at_mha=None,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        self.norm1 = norm_layer(dim)

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

        # Modality-specific modules
        self.modality_experts_at_ffn = modality_experts_at_ffn
        self.modality_experts_at_mha = modality_experts_at_mha
        if modality_experts_at_ffn is not None or modality_experts_at_mha is not None:
            assert modality_expert_rank > 0

        self.dummy_factor = dummy_factor
        self.modality_experts = None
        if self.modality_experts_at_ffn is not None:
            self.modality_experts = nn.ModuleDict()
            for mod in self.modality_experts_at_ffn:
                self.modality_experts[mod.name] = ModalityExpert(dim, dim, modality_expert_rank)

        self.attn = AltAttentionWithExperts(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
            modalities=modality_experts_at_mha,
            dummy_factor=dummy_factor,
            modality_expert_rank=modality_expert_rank,
        )
    
    def forward(self, x, padding_mask=None, alibi_bias=None, mode=None):

        if self.modality_experts is not None:
            remaining_experts = [
                m.name for m in self.modality_experts_at_ffn if (
                    m.name != mode and m.name in self.modality_experts.keys()
                )
            ]

        if self.layer_norm_first:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x), padding_mask, alibi_bias, mode=(
                        mode if self.modality_experts_at_mha is not None else None)
                )
            )
            x_modality = x = self.norm2(x)
            r = x = self.mlp(x)
            if self.modality_experts is not None:
                x += self.modality_experts[mode](x_modality)
                for name in remaining_experts:
                    x += self.dummy_factor * self.modality_experts[name](x_modality)
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(
                self.attn(x, padding_mask, alibi_bias, mode=(
                    mode if self.modality_experts_at_mha is not None else None)
                    )
            )
            r = x = self.norm1(x)
            x_modality = x
            x = self.mlp(x)
            if self.modality_experts is not None:
                x += self.modality_experts[mode](x_modality)
                for name in remaining_experts:
                    x += self.dummy_factor * self.modality_experts[name](x_modality)
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
        modalities=None,
        dummy_factor=0.0,
        modality_expert_rank=0,
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
        self.modalities = modalities
        self.modality_experts_qkv = None
        if self.modalities is not None:
            self.modality_experts_qkv = nn.ModuleDict()
            for mod in self.modalities:
                self.modality_experts_qkv[mod.name] = ModalityExpert(dim, dim*3, modality_expert_rank)

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
        if self.modality_experts_qkv is not None:
            qkv_experts = (
                self.modality_experts_qkv[mode](x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q += qkv_experts[0]
            k += qkv_experts[1]
            v += qkv_experts[2]

            remaining_experts = [
                m.name for m in self.modalities if (
                    m.name != mode and m.name in self.modality_experts_qkv.keys()
                )
            ]
            for name in remaining_experts:
                qkv_remainings = (
                    self.modality_experts_qkv[name](x)
                    .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q += self.dummy_factor * qkv_remainings[0]
                k += self.dummy_factor * qkv_remainings[1]
                v += self.dummy_factor * qkv_remainings[2]

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
        x = self.proj(x)
        # x = self.proj_drop(x)
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
