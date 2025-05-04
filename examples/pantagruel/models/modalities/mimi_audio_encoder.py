# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from functools import partial
import math
import numpy as np
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from fairseq.modules import GradMultiply
from fairseq.tasks import FairseqTask

from examples.data2vec.models.modalities.audio import (
    AudioEncoder,
    D2vAudioConfig,
)
from examples.data2vec.models.modalities.modules import Decoder1d
from examples.pantagruel.data.modality import Modality
from .base_encoder import PantagruelModalitySpecificEncoder
from .configuration_mimi import MimiConfig
from .modeling_mimi import (
    MimiConv1d,
    MimiEncoder,
    MimiModel,
    MimiTransformerModel,
    MimiSplitResidualVectorQuantizer,
)


logger = logging.getLogger(__name__)


class MimiAudioEncoder(PantagruelModalitySpecificEncoder):

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
        # for the audio encoder, we need to define the local_encoder and context_encoder from Mimi
        mimi_cfg = MimiConfig()
        mimi_cfg._attn_implementation = "sdpa"
        logger.info(
            f"Mimi Config:\t"
            f"  is_causal={mimi_cfg.is_causal}\t"
            f"  use_causal_conv={mimi_cfg.use_causal_conv}\t"
            f"  _attn_implementation={mimi_cfg._attn_implementation}"
        )

        # initialize Mimi CNN encoder and subsequent encoder_transformer
        mimi_cnn_encoder = MimiEncoder(mimi_cfg)
        mimi_encoder_transformer = MimiTransformerModel(mimi_cfg)

        # initialize CNN downsample module
        self.do_downsampling = False
        mimi_downsample = None
        if getattr(modality_cfg, "use_mimi_downsample", True):
            mimi_upsampling_ratios = [8, 6, 5, 4]
            hop_length = np.prod(mimi_upsampling_ratios)
            mimi_sr = 24000
            mimi_encodec_frame_rate = math.ceil(mimi_sr / hop_length)
            mimi_downsample = MimiConv1d(
                    mimi_cfg,
                    mimi_cfg.hidden_size,
                    mimi_cfg.hidden_size,
                    kernel_size=2 * int(mimi_encodec_frame_rate / mimi_cfg.frame_rate),
                    stride=2,
                    bias=False,
                    pad_mode="replicate",
                )
            self.do_downsampling = True
        
        # initialize quantizer if using discrete codebook
        self.num_quantizers = 0
        mimi_quantizer, mimi_upsample = None, None
        if getattr(modality_cfg, "use_mimi_discrete_codebook", False):
            self.num_quantizers = mimi_cfg.num_quantizers
            logger.info(f"Using Mimi discrete codebook with {self.num_quantizers} quantizers")
            mimi_quantizer = MimiSplitResidualVectorQuantizer(mimi_cfg)

        # projet Mimi output to the desired embed_dim
        if not mimi_quantizer:
            project_features = nn.Sequential(
                nn.LayerNorm(mimi_cfg.hidden_size),
                nn.Linear(mimi_cfg.hidden_size, embed_dim),
            )
        else:
            # create project features layer like the embedding layer 
            # where the input dimension is mimi_cfg.codebook_size and output 
            # being the embed_dim
            project_features = nn.Embedding(
                mimi_quantizer.codebook_size,
                embed_dim,
            )

        # setup the encoder using default Mimi architecture, keeping the 
        # key names as they are in the original Mimi model
        mimi_full_encoder = nn.ModuleDict(
            {
                "encoder": mimi_cnn_encoder,
                "encoder_transformer": mimi_encoder_transformer,
            }
        )
        if mimi_downsample is not None:
            mimi_full_encoder["downsample"] = mimi_downsample
        if mimi_quantizer is not None:
            mimi_full_encoder["quantizer"] = mimi_quantizer

        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else None
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=mimi_full_encoder,
            project_features=project_features,
            fixed_positional_encoder=None,
            relative_positional_encoder=None,
            context_encoder=None,
            decoder=decoder,
            get_alibi_bias=None,
            token_type_embeddings=token_type_embeddings,
        )

        if modality_cfg.init_mimi_pretrained:
            # load the pre-trained Mimi model
            mimi_pretrained_model = MimiModel.from_pretrained("kyutai/mimi")
            # set the weights of the local_encoder and context_encoder
            logger.info(f"Loading Mimi's CNN encoder weights")
            mimi_full_encoder["encoder"].load_state_dict(
                mimi_pretrained_model.encoder.state_dict()
            )
            logger.info(f"Loading Mimi's encoder_transformer weights")
            mimi_full_encoder["encoder_transformer"].load_state_dict(
                mimi_pretrained_model.encoder_transformer.state_dict()
            )
            if mimi_downsample is not None:
                logger.info(f"Loading Mimi's downsample weights")
                mimi_full_encoder["downsample"].load_state_dict(
                    mimi_pretrained_model.downsample.state_dict()
                )
            if mimi_quantizer is not None:
                logger.info(f"Loading Mimi's quantizer weights")
                mimi_full_encoder["quantizer"].load_state_dict(
                    mimi_pretrained_model.quantizer.state_dict()
                )
            logger.info(f"Loaded all relevant Mimi pretrained weights!")

        # log values of self.local_grad_mult
        self.grad_mult_encoder_transformer = modality_cfg.local_grad_mult_mimi_encoder_transformer
        self.grad_mult_downsample = modality_cfg.local_grad_mult_mimi_downsample
        self.grad_mult_quantizer = getattr(
            modality_cfg, "local_grad_mult_mimi_discrete_codebook", 0.0
        )
        logger.info(
            f"Mimi's local gradient multipliers:\t"
            f"- encoder: {self.local_grad_mult}, "
            f"- encoder_transformer: {self.grad_mult_encoder_transformer}, "
            f"- downsample: {self.grad_mult_downsample}, "
            f"- quantizer: {self.grad_mult_quantizer}"
        )

        # freeze parameters of mimi's components if grad_mult == 0
        def freeze_parameters(module, grad_mult):
            if grad_mult == 0:
                logger.info(f"Freezing {module.__class__.__name__}")
                for param in module.parameters():
                    param.requires_grad = False
        freeze_parameters(mimi_cnn_encoder, self.local_grad_mult)
        freeze_parameters(mimi_encoder_transformer, self.grad_mult_encoder_transformer)
        if mimi_downsample is not None:
            freeze_parameters(mimi_downsample, self.grad_mult_downsample)
        if mimi_quantizer is not None:
            freeze_parameters(mimi_quantizer, self.grad_mult_quantizer)

    def local_features(self, features):
        input_values = features.unsqueeze(1)  # B x L -> B x 1 x L

        # forward pass through the local encoder
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder["encoder"](input_values)
            else:
                x = GradMultiply.apply(self.local_encoder["encoder"](input_values), self.local_grad_mult)
        else:
            with torch.no_grad():
                x = self.local_encoder["encoder"](input_values)  
        # x: B x D x L

        # forward pass through the encoder_transformer
        if self.grad_mult_encoder_transformer > 0:
            if self.grad_mult_encoder_transformer == 1.0:
                x = self.local_encoder["encoder_transformer"](x.transpose(1, 2))[0]
            else:
                x = GradMultiply.apply(
                    self.local_encoder["encoder_transformer"](x.transpose(1, 2))[0],
                    self.grad_mult_encoder_transformer,
                )
        else:
            with torch.no_grad():
                x = self.local_encoder["encoder_transformer"](x.transpose(1, 2))[0]
        # x: B x L x D

        # forward pass through the downsample
        if self.do_downsampling:
            if self.grad_mult_downsample > 0:
                if self.grad_mult_downsample == 1.0:
                    x = self.local_encoder["downsample"](x.transpose(1, 2))
                else:
                    x = GradMultiply.apply(self.local_encoder["downsample"](x.transpose(1, 2)), self.grad_mult_downsample)
            else:
                with torch.no_grad():
                    x = self.local_encoder["downsample"](x.transpose(1, 2)) 
            # x: B x D x L

        if self.num_quantizers > 0:
            if self.grad_mult_quantizer > 0:
                if self.grad_mult_quantizer == 1.0:
                    x = self.local_encoder["quantizer"].encode(
                        x, num_quantizers=1
                    )
                else:
                    x = GradMultiply.apply(
                        self.local_encoder["quantizer"].encode(
                            x, num_quantizers=1
                        ),
                        self.grad_mult_quantizer,
                    )
            else:
                with torch.no_grad():
                    x = self.local_encoder["quantizer"].encode(
                        x, num_quantizers=1
                    )
            x = x.squeeze(0)  # num_codebooks x B x L -> B x L
            
            # x: num_codebooks x B x L
            # indices = torch.randint(0, self.num_quantizers, (x.shape[1],), device=x.device)
            # x = x[indices, torch.arange(x.shape[1])] # B x L

        if self.num_quantizers == 0 and self.do_downsampling:
            x = self.project_features(x.transpose(1, 2))
        else:
            x = self.project_features(x)

        return x # B x L x D'

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        mask_info = None
        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0) # M x L x D (M=B*clone_batch)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": None,
            "alibi_scale": None,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
        token_type_ids=None,
    ):
        x = self.local_features(features) # B x L x D
        res = self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )
        x = res["x"]

        if self.token_type_embeddings is not None and token_type_ids is not None:
            # self.token_type_embeddings(token_type_ids): 1 x D
            x = x + self.token_type_embeddings(token_type_ids)
        res["x"] = x

        return res
