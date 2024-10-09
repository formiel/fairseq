import torch

def get_random_crops(input_tensor, frame_length=None):
    """
    Sample two frames from a batch of encoded embeddings.
    """
    B, T, C = input_tensor.size()
    if not frame_length:
        frame_length = T // 3

    pos = torch.randint(0, T - frame_length + 1, size=(B,))
    crops1 = torch.stack(
        [input_tensor[i, pos[i].item():pos[i].item() + frame_length, :] for i in range(B)], dim=0
    )
    pos = torch.randint(0, T - frame_length + 1, size=(B,))
    crops2 = torch.stack(
        [input_tensor[i, pos[i].item():pos[i].item() + frame_length, :] for i in range(B)], dim=0
    )

    return [crops1, crops2]