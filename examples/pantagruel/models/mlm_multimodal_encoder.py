from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.pantagruel.models.modalities.random_projection_quantizer import RandomProjectionQuantizer

logger = logging.getLogger(__name__)


@dataclass
class MLMMultimodalEncoderConfig:
    num_heads: int = 8
    num_layers: int = 6
    embed_dim: int = 768
    tie_weights_embeddings: bool = True


class MLMMultimodalEncoder(nn.Module):
    def __init__(
        self, 
        task, 
        embed_dim: int, 
        config: MLMMultimodalEncoderConfig, 
        rpq_config=None,
        modalities=None,
        embedding_weights=None,
        downsampling_audio_ratio=None,
    ):
        super().__init__()

        self.downsampling_audio_ratio = downsampling_audio_ratio

        self.input_projs = nn.ModuleDict(
            {mod.name: nn.Linear(embed_dim, embed_dim) 
            for mod in modalities if "_" not in mod.name}
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=config.num_heads,
                dim_feedforward=embed_dim * 4,
                activation='relu',
                norm_first=True,
                batch_first=True,
            ) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU(inplace=False)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        self.padding_idx = task.source_dictionary.pad()

        out_dims = {
            "TEXT": len(task.source_dictionary),
            "AUDIO": rpq_config.codebook_dim,
        }
        self.prediction_heads = nn.ModuleDict(
            {mod.name: nn.Linear(config.embed_dim, out_dims[mod.name]) 
            for mod in modalities if "_" not in mod.name}
        )
        if embedding_weights is not None:
            logger.info("Setting prediction heads for TEXT to be the same as embedding weights provided")
            self.prediction_heads["TEXT"].weight = embedding_weights

        self.random_projection_quantizer = RandomProjectionQuantizer(rpq_config)

    def forward(self, x, source, target=None, masks=None, padding_mask=None):
        # concat the inputs from all modalities so that the output become B x (T1 + T2) x D
        sorted_mods = sorted(x.keys())
        x_concat = [x[key] for key in sorted_mods]
        x_concat = torch.cat(x_concat, dim=1)

        mask_concat = [masks[key] for key in sorted_mods] if masks is not None else None
        if mask_concat is not None:
            mask_concat = torch.cat(mask_concat, dim=1) # [B, T_concat]
        M, T_concat, D = x_concat.size()

        # create the combined padding mask for all tokens (masked + unmasked)
        combined_padding_mask = {}
        if len(sorted_mods) == 1:
            combined_padding_mask = (
                padding_mask[sorted_mods[0]] if isinstance(padding_mask, dict) 
                else padding_mask if isinstance(padding_mask, torch.Tensor) 
                else None
            )
        else:
            if not all([_pm is None for _pm in padding_mask.values()]):
                for _mod, _pm in padding_mask.items():
                    if _pm is None:
                        _pm = torch.zeros(
                            M, x[_mod].size(1), dtype=torch.bool, device=x_concat.device
                        )
                    combined_padding_mask[_mod] = _pm
                # Concatenate padding masks for all modalities
                combined_padding_mask = [combined_padding_mask[key] for key in sorted_mods]
                combined_padding_mask = torch.cat(combined_padding_mask, dim=1)
            else:
                combined_padding_mask = None

        # forward to the transformer encoder (both masked and unmasked tokens)
        for layer in self.layers:
            x_concat = layer(x_concat, src_key_padding_mask=combined_padding_mask) # [M, T_concat, D]
        
        # project masked tokens or all tokens if no masking
        ids_masked_mod, ids_masked_concat = None, None
        if mask_concat is not None:
            # get the indices of masked tokens for each modality
            ids_masked_mod_list, ids_masked_concat_list = [], []
            offset_for_mod_len = 0
            num_masked_by_mod = {}
            for i, _mod in enumerate(sorted_mods):
                min_num_masked = masks[_mod].sum(dim=-1).min()
                ids_unmasked_masked = masks[_mod].argsort(dim=-1)
                ids_masked_mod = ids_unmasked_masked[:, -min_num_masked:] # ids of masked tokens in each modality
                ids_masked_concat = ids_masked_mod + offset_for_mod_len  # ids of masked tokens in the concatenated sequence
                ids_masked_mod_list.append(ids_masked_mod)
                ids_masked_concat_list.append(ids_masked_concat)
                offset_for_mod_len += x[_mod].size(1)
                num_masked_by_mod[_mod] = min_num_masked.item()  # number of masked tokens for each modality
            
            ids_masked_concat = torch.cat(ids_masked_concat_list, dim=1)
            ids_masked_mod = ids_masked_mod_list
            # select only masked tokens
            x_concat = x_concat[torch.arange(M).unsqueeze(1), ids_masked_concat, :]  # [M, T_concat_(masked), D]

        x_concat = self.norm(x_concat)
        x_concat = self.act(x_concat) # [M, T_concat_(masked), D]

        # project the output to the prediction heads
        x_out = {}
        start_idx = 0
        for i, _mod in enumerate(sorted_mods):
            if self.prediction_heads is not None and _mod.upper() in self.prediction_heads:
                x_out[_mod] = self.prediction_heads[_mod.upper()](
                    x_concat[:, start_idx:start_idx + num_masked_by_mod[_mod], :]
                ) # [M, T_masked, V]
                start_idx += num_masked_by_mod[_mod]
            else:
                x_out[_mod] = x_concat[:, start_idx:start_idx + x[_mod].size(1), :]  # [M, T_all, D]
                start_idx += x[_mod].size(1)

        # get targets for each modality
        labels = None
        if target is not None and ids_masked_concat is not None:
            labels = {}
            # start_idx = 0
            for i, _mod in enumerate(sorted_mods):
                if _mod.upper() == "TEXT":
                    _label = self.get_label_text(
                        target if isinstance(target, torch.Tensor) else target[_mod],
                        ids_masked=ids_masked_mod[i],
                    )
                elif _mod.upper() == "AUDIO":
                    _label = self.get_label_audio(
                        target if isinstance(target, torch.Tensor) else target[_mod],
                        x[_mod].size(1),  # T_cnn
                        ids_masked=ids_masked_mod[i],
                    )
                # start_idx += num_masked_by_mod[i]
                labels[_mod] = _label
            
        return {
            "x": x_out,  # {mod: M x T_masked x D_out}
            "labels": labels if labels is not None else None,  # {mod: M x T_masked}
        }

    def get_label_text(
        self, target, ids_masked=None,
    ):
        if ids_masked is not None:
            target = target[torch.arange(target.size(0)).unsqueeze(1), ids_masked]
        return target

    def get_label_audio(
        self, target, T_cnn, ids_masked,
    ):
        M, C, T_mel = target.size()  # [B, C, T_mel]
        target = target.transpose(1, 2) # [B, T_mel, C]
        ids_masked_spec = self.map_masked_indices_to_interp_window(
            ids_masked, T_cnn, T_mel
        )
        target = target[torch.arange(target.size(0)).unsqueeze(1), ids_masked_spec]
        target = target.transpose(0, 1) # TxBxC

        label = self.random_projection_quantizer(target) # T_masked x B

        return label.transpose(0, 1)  # B x T_masked

    def map_masked_indices_to_interp_window(self, ids_masked, T_cnn, T_mel):
        """
        Map masked CNN indices to interpolated indices with surrounding windows.
        """
        # map CNN indices to interpolated positions
        t_interp = torch.round((ids_masked.float() / (T_cnn - 1)) * (T_mel - 1)).long() # MxT_masked
        # M, T_masked = t_interp.size()

        # # for each t', take a window around it (clamped to bounds)
        # t_minus = torch.clamp(t_interp - 1, min=0)
        # t_plus  = torch.clamp(t_interp + 1, max=T_mel - 1)
        # expanded = torch.stack([t_minus, t_interp, t_plus], dim=-1)  # (M, T_masked, 3)
        # expanded = expanded.view(M, -1)  # (M, T_masked * 3)

        # # Remove repeated values per row, keep order, cut to min length
        # # Use torch.unique_consecutive for speed (works because repeated values are consecutive due to clamping)
        # unique_rows = []
        # min_len = expanded.size(1)
        # for row in expanded:
        #     uniq = torch.unique_consecutive(row)
        #     min_len = min(min_len, uniq.size(0))
        #     unique_rows.append(uniq)
        # # Stack and cut to min_len
        # ids_masked_spec = torch.stack([row[:min_len] for row in unique_rows], dim=0)  # (M, min_len)
        return t_interp