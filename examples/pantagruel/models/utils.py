import torch

def _to_bf16(x, forward=True):
    if forward:
        return x.to(torch.bfloat16)
    else:
        return x.to(torch.float16)

