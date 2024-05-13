# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.tasks import FairseqTask
from .base_type import D2vModalityConfig, PantagruelModalitySpecificEncoder
from examples.data2vec.models.modalities.base import (
    get_alibi_bias,
)
from examples.data2vec.models.modalities.modules import BlockEncoder, Decoder1d
from examples.data2vec.models.modalities.audio import (
    D2vAudioConfig,
    AudioEncoder,
)
from examples.data2vec.data.modality import Modality


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