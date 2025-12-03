# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import logging

import torch
import torch.nn as nn

from fairseq.tasks import FairseqTask
from fairseq.data.data_utils import compute_mask_indices
from .base_encoder import PantagruelModalitySpecificEncoder

from examples.data2vec.models.modalities.audio import (
    D2vAudioConfig,
    AudioEncoder,
)
from examples.data2vec.models.modalities.base import (
    MaskSeed, random_masking
)

from examples.pantagruel.data.modality import Modality


@dataclass
class PantagruelD2vAudioConfig(D2vAudioConfig):
    type: Modality = Modality.AUDIO
    local_grad_mult_mimi_encoder_transformer: float = 1.0

    use_mimi_downsample: bool = True
    local_grad_mult_mimi_downsample: float = 1.0

    init_mimi_pretrained: bool = True
    
    use_mimi_discrete_codebook: bool = False
    local_grad_mult_mimi_discrete_codebook: float = 0.0


class AudioTypeEncoder(PantagruelModalitySpecificEncoder):

    modality_cfg: D2vAudioConfig

    def __init__(
        self,
        modality_cfg: D2vAudioConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
        token_type_embeddings: Optional[nn.Module],
    ):

        audio_encoder = AudioEncoder(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            make_block=make_block,
            norm_layer=norm_layer,
            layer_norm_first=layer_norm_first,
            alibi_biases=alibi_biases,
            task=task,
            rotary_emb=None,
        )
        self.feature_enc_layers = eval(modality_cfg.feature_encoder_spec)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=audio_encoder.local_encoder,
            project_features=audio_encoder.project_features,
            fixed_positional_encoder=audio_encoder.fixed_positional_encoder,
            relative_positional_encoder=audio_encoder.relative_positional_encoder,
            context_encoder=audio_encoder.context_encoder,
            decoder=audio_encoder.decoder,
            get_alibi_bias=audio_encoder.get_alibi_bias,
            token_type_embeddings=token_type_embeddings,
            rotary_emb=None,
        )

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask
    
    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.project_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = cfg.mask_prob

            if (
                cfg.mask_prob_min is not None
                and cfg.mask_prob_min >= 0
                and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    try:
                        mask = compute_mask_indices(
                            (B, T),
                            padding_mask,
                            mask_prob,
                            cfg.mask_length,
                            min_masks=1,
                            require_same_masks=True,
                            mask_dropout=cfg.mask_dropout,
                            add_masks=cfg.add_masks,
                            seed=mask_seed.seed if mask_seed is not None else None,
                            epoch=mask_seed.update if mask_seed is not None else None,
                            indices=mask_seed.ids if mask_seed is not None else None,
                        )
                        mask = torch.from_numpy(mask).to(device=x.device)
                        if self.modality_cfg.inverse_mask:
                            mask = 1 - mask
                        mask_info = self.make_maskinfo(x, mask)
                    except:
                        mask_info = random_masking(x, mask_prob, mask_seed)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info