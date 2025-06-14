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
    ):
        super().__init__()

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

    def forward(self, x, source, target=None, mask_info=None, padding_mask=None, clone_batch=1):
        # inputs: x: {mod: M x T x C (M = B*clone_batch)}, where x contains only visible (unmasked) tokens
        # encoder_mask (mask_info): {mod: MaskInfo(x_unmasked, mask, ids_restore, ids_keep)}, in which
        # - mask: the length of all (masked + visible) tokens, 
        # - ids_restore has the length of all tokens
        # - ids_keep contains ids of visible tokens
        # masked_padding_mask (padding_mask): {mod: B x T} or None, where T is the length of visible tokens

        # restore inputs to the original length
        restored_x, masks = self._restore_inputs(x, mask_info=mask_info)
        
        # concat the inputs from all modalities so that the output become M x (T_all_mod1 + T_all_mod2) x D
        sorted_mods = sorted(restored_x.keys())
        x_concat = [restored_x[key] for key in sorted_mods]
        x_concat = torch.cat(x_concat, dim=1)
        mask_concat = [masks[key] for key in sorted_mods] if masks is not None else None # [M, T_concat]
        if mask_concat is not None:
            mask_concat = torch.cat(mask_concat, dim=1)
        M, T_concat, D = x_concat.size()

        # create the combined padding mask
        combined_padding_mask = {}
        if not all([_pm is None for _pm in padding_mask.values()]):
            for _mod, _pm in padding_mask.items():
                _pm_full = torch.zeros(
                    M, restored_x[_mod].size(1), dtype=torch.bool, device=x_concat.device
                )
                if _pm is not None:
                    # _pm: padding mask for x[_mod], shape BxT where T is the length of visible tokens
                    # restored_x[_mod] has the length of all tokens T_all for _mod 
                    # -> use values from _pm where ids_keep is True
                    # -> for the rest of the tokens, set padding value to True if source[_mod] == self.padding_idx
                    _pm_full[torch.arange(M).unsqueeze(1), mask_info[_mod].ids_keep[..., 0]] = _pm
                    if isinstance(source, torch.Tensor):
                        _source = source.repeat_interleave(
                            clone_batch, dim=0
                        )
                    else:
                        # source is a dict with modality names as keys
                        _source = source[_mod]["source"]
                        _source = _source.repeat_interleave(
                            clone_batch, dim=0
                        )
                    _mask = mask_info[_mod].mask.to(torch.bool)
                    _pm_full[_mask] = (_source[_mask] == self.padding_idx)
                combined_padding_mask[_mod] = _pm_full
            # Concatenate padding masks for all modalities
            combined_padding_mask = [combined_padding_mask[key] for key in sorted_mods]
            combined_padding_mask = torch.cat(combined_padding_mask, dim=1)
        else:
            combined_padding_mask = None

        # forward to the transformer encoder (both masked and unmasked tokens)
        for layer in self.layers:
            x_concat = layer(x_concat, src_key_padding_mask=combined_padding_mask) # [M, T_concat, D]
        
        # project masked tokens or all tokens if no masking
        ids_masked = None
        if mask_concat is not None:
            min_num_masked = mask_concat.sum(dim=-1).min()
            ids_unmasked_masked = mask_concat.argsort(dim=-1)
            ids_masked = ids_unmasked_masked[:, -min_num_masked:]
            x_concat = x_concat[torch.arange(M).unsqueeze(1), ids_masked, :]

        x_concat = self.norm(x_concat)
        x_concat = self.act(x_concat) # [M, T_concat_(masked), D]

        # project the output to the prediction heads
        x_out = {}
        start_idx = 0
        num_masked_by_mod = torch.tensor(
            [restored_x[_mod].size(1) - x[_mod].size(1) for _mod in sorted_mods],
            device=x_concat.device
        )
        for i, _mod in enumerate(sorted_mods):
            if self.prediction_heads is not None and _mod.upper() in self.prediction_heads:
                x_out[_mod] = self.prediction_heads[_mod.upper()](
                    x_concat[:, start_idx:start_idx + num_masked_by_mod[i], :]
                ) # [M, T_masked, V]
                start_idx += num_masked_by_mod[i]
            else:
                x_out[_mod] = x_concat[:, start_idx:start_idx + restored_x[_mod].size(1), :]  # [M, T_all, D]
                start_idx += restored_x[_mod].size(1)

        # get targets for each modality
        labels = None
        if target is not None and ids_masked is not None:
            labels = {}
            start_idx = 0
            for i, _mod in enumerate(sorted_mods):
                if _mod.upper() == "TEXT":
                    _label = self.get_label_text(
                        target if isinstance(target, torch.Tensor) else target[_mod]["target"],
                        ids_masked=ids_masked[:, start_idx:start_idx + num_masked_by_mod[i]],
                        clone_batch=clone_batch,
                    )
                elif _mod.upper() == "AUDIO":
                    _label = self.get_label_audio(
                        target if isinstance(target, torch.Tensor) else target[_mod]["target"],
                        restored_x[_mod].size(1),  # T_cnn
                        ids_masked=ids_masked[:, start_idx:start_idx + num_masked_by_mod[i]],
                        clone_batch=clone_batch,
                    )
                start_idx += num_masked_by_mod[i]
                labels[_mod] = _label
                # logger.info(f"[{_mod.upper()}]: pred: {x_out[_mod].size()} labels: {labels[_mod].size()}")
            
        return {
            "x": x_out,  # {mod: M x T_masked x D_out}
            "labels": labels if labels is not None else None,  # {mod: M x T_masked}
        }

    def _restore_inputs(self, x, mask_info=None):
        restored_x = {}
        masks = {} if mask_info is not None else None
        for _mod, _x in x.items():
            # forward the input through the input projection layer
            _x = self.input_projs[_mod.upper()](_x)  # [M, T, D]
            if masks is not None:
                # get the ids of visible tokens and the lengths
                ids_keep = mask_info[_mod].ids_keep[..., 0]  # [M, T]
                M, T_all, D = mask_info[_mod].ids_restore.shape

                _x_full = self.mask_emb.expand(M, T_all, D).clone()
                _x_full[torch.arange(M).unsqueeze(1), ids_keep, :] = _x
                restored_x[_mod] = _x_full
                masks[_mod] = mask_info[_mod].mask
            else:
                restored_x[_mod] = _x

        return restored_x, masks

    def get_label_text(
        self, target, ids_masked=None, clone_batch=1
    ):
        if ids_masked is not None:
            target = target.repeat_interleave(clone_batch, dim=0)
            target = target[torch.arange(target.size(0)).unsqueeze(1), ids_masked]
        return target

    def get_label_audio(
        self, target, T_cnn, ids_masked, clone_batch=1
    ):
        target = target.repeat_interleave(clone_batch, dim=0)  # [B, C, T] -> [B*clone_batch, C, T]
        M, C, T_mel = target.size()  # [B*clone_batch, C, T_mel]
        target = target.transpose(1, 2) # [B*clone_batch, T_mel, C]

        ids_masked_spec = self.map_masked_indices_to_interp_window(
            ids_masked, T_cnn, T_mel
        )

        target = target[torch.arange(target.size(0)).unsqueeze(1), ids_masked_spec]
        target = target.transpose(0, 1) # TxMxC

        label = self.random_projection_quantizer(target) # T_masked x B*clone_batch

        return label.transpose(0, 1)  # B*clone_batch x T_masked

    def map_masked_indices_to_interp_window(self, ids_masked, T_cnn, T_mel):
        """
        Map masked CNN indices to interpolated indices with surrounding windows.
        """
        # map CNN indices to interpolated positions
        t_interp = torch.round((ids_masked.float() / (T_cnn - 1)) * (T_mel - 1)).long() # MxT_masked
        # M, _ = t_interp.size()

        # not try using surrounding positions yet
        # # for each t', take a window around it (clamped to bounds)
        # t_minus = torch.clamp(t_interp - 1, min=0)
        # t_plus  = torch.clamp(t_interp + 1, max=T_mel - 1)
        # expanded = torch.stack([t_minus, t_interp, t_plus], dim=-1)  # (B, T, 3)
        # ids_masked_spec = expanded.view(M, -1)  # (B, T * 3)
        return t_interp