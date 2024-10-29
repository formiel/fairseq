import torch

def get_random_crops(input_tensor, frame_length=200):
    """
    Sample two non-overlapping frames from a batch of encoded embeddings.
    """
    B, T, _ = input_tensor.size()
    frame_length = min(T//2, frame_length)

    # get the first crop
    pos1 = torch.randint(0, T//2 - frame_length + 1, size=(B,))
    crops1 = torch.stack(
        [input_tensor[i, pos1[i].item():pos1[i].item() + frame_length, :] for i in range(B)], dim=0
    )

    # get the second crop
    pos2 = pos1 + frame_length  # Start after crops1
    max_start_pos2 = T - frame_length  # Maximum starting position for crops2
    # Mask for positions that would go out of bounds
    invalid_pos2_mask = pos2 > max_start_pos2
    if invalid_pos2_mask.sum().item() > 0:
        pos2[invalid_pos2_mask] = torch.randint(T//2, T, size=(invalid_pos2_mask.sum(),))

    crops2 = torch.stack(
        [input_tensor[i, pos2[i].item():pos2[i].item() + frame_length, :] for i in range(B)], dim=0
    )

    return [crops1, crops2]


def create_negative_pairs(X, Y, nclone=1):
    """create negative pairs"""
    M, D = X.size()
    X_rep = X.unsqueeze(1).expand(M, M-nclone, D)

    mask = torch.eye(M // nclone, dtype=torch.bool)
    mask = mask.repeat_interleave(nclone, 1).repeat_interleave(nclone, 0)

    Y_negs = [Y[~mask[:,i]] for i in range(M)] # [list of M elements of (M-nclone, D)]
    Y_negs = torch.stack(Y_negs, dim=0) # (M, M-nclone, D)

    out = torch.cat((X_rep, Y_negs), dim=-1) # (M, M-nclone, 2D)

    return out.view(-1, 2*D)