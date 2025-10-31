import logging

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from omegaconf import II, MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils, tasks

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel, register_model, FairseqEncoderDecoderModel
)
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2Seq2SeqConfig, Wav2VecEncoder
)
from fairseq.models.transformer import Embedding
from fairseq.tasks import FairseqTask
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable

from examples.data2vec.data.modality import Modality


logger = logging.getLogger(__name__)


@dataclass
class Data2vec2STConfig(Wav2Vec2Seq2SeqConfig):
    seed: int = 1


@register_model("data2vec2_st", dataclass=Data2vec2STConfig)
class Data2vec2STModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Data2vec2STConfig, task: FairseqTask):
        """Build a new model instance."""

        tgt_dict = task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, cfg.decoder_embed_dim
        )
        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: Data2vec2STConfig):
        return Wav2VecEncoder(cfg)

    @classmethod
    def build_decoder(cls, cfg: Data2vec2STConfig, tgt_dict, embed_tokens):
        return TransformerDecoderScriptable(cfg, tgt_dict, embed_tokens)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, padding_mask=None
    ):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(
            source=src_tokens, padding_mask=padding_mask, features_only=True
        )
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out
