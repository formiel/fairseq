from examples.data2vec.models.modalities.modules import (
    AltAttention,
    EncDecAttention
)
import torch


def main():
    B, N, C = 32, 100, 768
    L = N
    H = 2
    D = C // H
    attn = AltAttention(dim=C, num_heads=H).to("cuda")
    x = torch.randn(B, N, C).to("cuda")
    paddding_mask = (torch.randn(B, L) > 0.001).to("cuda")
    alibi_bias = torch.randn(B, H, L, L).to("cuda")
    y1 = attn(x, paddding_mask, alibi_bias, fast=False)
    y2 = attn(x, paddding_mask, alibi_bias, fast=True)

    # print(f'y1\n{y1}')
    # print(f'y2\n{y2}')
    # print('*'*100)
    print("Checking AltAttention")
    print(f'y1={torch.linalg.matrix_norm(y1)}')
    print(f'y2={torch.linalg.matrix_norm(y2)}')
    print(f"max diff = {(y1 - y2).abs().max()}")


    print("*"*200)
    print("Checking EncDecAttention")
    B, N, C = 2, 10, 128
    N_kv, C_kv = 10, 128
    L = N
    H = 2
    D = C // H
    attn = EncDecAttention(q_dim=C, kv_dim=C_kv, num_heads=H).to("cuda")
    q = torch.randn(B, N, C).to("cuda")
    kv = torch.randn(B, N_kv, C_kv).to("cuda")
    paddding_mask = (torch.randn(B, L) > 0.001).to("cuda")
    alibi_bias = torch.randn(B, H, L, L).to("cuda")
    y1 = attn(q, kv, paddding_mask, alibi_bias, fast=False)
    y2 = attn(q, kv, paddding_mask, alibi_bias, fast=True)

    print(f'y1={torch.linalg.matrix_norm(y1)}')
    print(f'y2={torch.linalg.matrix_norm(y2)}')
    print(f"max diff = {(y1 - y2).abs().max()}")
    
    return


if __name__ == "__main__":
    main()