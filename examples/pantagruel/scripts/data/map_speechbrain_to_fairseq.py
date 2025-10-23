import torch


def init_fairseq_model(fairseq_model_class, cfg, ckpt_path=None):
    """Instantiate a fairseq model (eg wav2vec2) without weights, optionally load checkpoint."""
    model = fairseq_model_class(cfg)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
    return model
    

def map_sb_to_fairseq(sb_state_dict, fairseq_state):
    """Map SpeechBrain parameter names to fairseq model parameter names."""
    fs_state_dict = {}
    for sb_name, sb_param in sb_state_dict.items():
        sb_name = sb_name.replace("model.wav2vec2.", "")
        sb_name = sb_name.replace("masked_spec_embed", "mask_emb")
        if sb_name.startswith("feature_extractor.conv_layers"):
            i = sb_name.split(".")[2]
            sb_name = sb_name.replace(
                f"feature_extractor.conv_layers.{i}.conv.", f"feature_extractor.conv_layers.{i}.0."
            ).replace(
                f"feature_extractor.conv_layers.{i}.layer_norm", f"feature_extractor.conv_layers.{i}.2.1"
            )
        # sb line 30
        sb_name = sb_name.replace("feature_projection.layer_norm", "layer_norm")
        sb_name = sb_name.replace("feature_projection.projection", "post_extract_proj")
        sb_name = sb_name.replace("encoder.pos_conv_embed.conv", "encoder.pos_conv.0")
        if sb_name.startswith("encoder.layers."):
            i = sb_name.split(".")[2]
            sb_name = sb_name.replace(
                f"encoder.layers.{i}.attention", f"encoder.layers.{i}.self_attn"
            ).replace(
                f"encoder.layers.{i}.layer_norm", f"encoder.layers.{i}.self_attn_layer_norm"
            ).replace(
                f"encoder.layers.{i}.feed_forward.intermediate_dense", f"encoder.layers.{i}.fc1"
            ).replace(
                f"encoder.layers.{i}.feed_forward.output_dense", f"encoder.layers.{i}.fc2"
            )

        sb_name = sb_name.replace("model.quantizer.codevectors", "quantizer.vars")
        sb_name = sb_name.replace("model.project_hid", "final_proj")
        sb_name = sb_name.replace("model.project_q", "project_q")
        sb_name = sb_name.replace("model.quantizer.weight_proj", "quantizer.weight_proj")

        if sb_name in fairseq_state.keys():
            fs_state_dict[sb_name] = sb_param
        else:
            # either skip or log unmapped
            print(f"WARNING: sb param {sb_name} not in fairseq model.")
    
    return fs_state_dict


def save_fairseq_checkpoint(fs_model, save_path, args, optimizer_history):
    """Save a fairseq-compatible checkpoint."""
    ckpt = {
        "cfg": args,            # you can fill in model args if needed
        "model": fs_model.state_dict(),
        "optimizer_history": optimizer_history, # empty or as needed
        "extra_state": {}, 
        "version": 1
    }
    torch.save(ckpt, save_path)
    print(f"Saved fairseq checkpoint to {save_path}")


def main():
    ROOT_DIR = "/linkhome/rech/genlig01/ucy22cr/pantagruel/LeBenchmark"
    sp_path = f"{ROOT_DIR}/wav2vec2-FR-14K-large/wav2vec2.ckpt"
    fs_path = f"{ROOT_DIR}/wav2vec2-FR-7K-large/checkpoint_best.pt"

    sp_state = torch.load(sp_path, map_location="cpu", weights_only=True)
    fs_state = torch.load(fs_path, map_location="cpu")
    pretrained_args = fs_state["cfg"]

    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
    print('Initializing configuration...')
    cfg = Wav2Vec2Config(
        extractor_mode="layer_norm",
        layer_norm_first=True,
        quantize_targets=True,
        encoder_layers=24,
        encoder_embed_dim=1024,
        encoder_ffn_embed_dim=4096,
        encoder_attention_heads=16,
        conv_bias=True,
        latent_dim=256, final_dim=256, latent_groups=2, quantizer_factor=2
    )
    print('Initializing models...')
    fs_model = init_fairseq_model(Wav2Vec2Model, cfg)
    mapped_sd = map_sb_to_fairseq(sp_state, fs_state["model"])
    fs_model.load_state_dict(mapped_sd, strict=False)

    # override pretrained_args with modified cfg
    print(f'pretrained_args: {pretrained_args}\n{type(pretrained_args)}')
    print(f'cfg: {cfg}\n{type(cfg)}')

    save_path = f"{ROOT_DIR}/wav2vec2-FR-14K-large/fairseq_wav2vec2_from_sb.pt"
    save_fairseq_checkpoint(
        fs_model, save_path, pretrained_args, fs_state["optimizer_history"]
    )

if __name__ == "__main__":
    main()