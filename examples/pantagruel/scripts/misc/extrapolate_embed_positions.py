import argparse
import numpy as np
import torch


def interpolate_positional_embeddings(origE, scale_factor=2):
    """
    Linearly interpolates a positional embedding matrix E to a larger size,
    following the method described in Karypis et al. (2024).

    Parameters
    ----------
    E : numpy.ndarray
        Original embedding matrix of shape (Lt, d).
    scale_factor : int
        How much larger the new matrix should be. Must be an integer.
        Example: 2 = double size.

    Returns
    -------
    E_new : numpy.ndarray
        Interpolated embedding matrix of shape (Lt * scale_factor, d).
    """
    print(f"Original embedding shape: {origE.shape}")
    E = origE[1:-1] # removing first and last token
    print(f"After embedding shape: {E.shape}")
    Lt, d = E.shape
    Le = Lt * scale_factor
    beta = scale_factor
    
    E_new = np.zeros((Le, d), dtype=E.dtype)
    print(f"New embedding shape: {E_new.shape}")

    for i in range(Le):
        # Corresponds to floor(i / β)
        base_idx = i // beta
        
        # modulo part i % β
        r = i % beta
        
        # When base_idx == Lt - 1, use the last embedding for safety
        if base_idx == Lt - 1:
            E_new[i] = E[-1]
        else:
            w1 = (beta - r) / beta
            w2 = r / beta
            E_new[i] = w1 * E[base_idx] + w2 * E[base_idx + 1]

    # Add back the first and last token embeddings
    first_token = origE[0]
    last_token = origE[-1]
    E_new = np.vstack([first_token, E_new, last_token])
    print(f"Final new embedding shape (with first and last tokens): {E_new.shape}")

    return torch.from_numpy(E_new)


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate positional embeddings to a larger size."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input model containing the embed positions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the model",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=2,
        help="Factor by which to scale the positional embeddings.",
    )

    args = parser.parse_args()

    ckpt = torch.load(args.input_path, map_location="cpu")
    model = ckpt['model']
    # get positional embeddings
    pos_embed_key = 'modality_encoders.TEXT.local_encoder.embed_positions.weight'
    origE = model[pos_embed_key].numpy()

    # Interpolate embeddings
    E_new = interpolate_positional_embeddings(origE, scale_factor=args.scale_factor)

    ckpt['model'][pos_embed_key] = E_new

    # Save new model
    torch.save(ckpt, args.output_path)
    print(f"Saving model with extrapolated embed to {args.output_path}")


# Example usage
if __name__ == "__main__":
    main()

    # Lt = 6
    # d = 5
    # E = np.random.randn(Lt, d)
    # print(f"E: {E}")

    # E2 = interpolate_positional_embeddings(E, scale_factor=2)
    # print(f"E2: {E2}")
    # print("Original shape:", E.shape)
    # print("New shape:", E2.shape)