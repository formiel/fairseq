import logging

from dataclasses import dataclass, field
from typing import Any

from omegaconf import II, MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel, register_model, FairseqEncoderDecoderModel
)
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2Seq2SeqConfig, Wav2VecEncoder, TransformerDecoder
)
from fairseq.models.transformer import Embedding
from fairseq.tasks import FairseqTask

from examples.data2vec.data.modality import Modality


logger = logging.getLogger(__name__)


@dataclass
class Data2vec2STConfig(Wav2Vec2Seq2SeqConfig):
    toto: int = 1


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
        return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    
