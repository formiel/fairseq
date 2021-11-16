#!/usr/bin/env python3

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
)
from fairseq.models.speech_to_text import (
    S2TTransformerEncoder,
    TransformerDecoderScriptable,
)
from fairseq.models.transformer import TransformerEncoder
from torch import Tensor

from geomloss import SamplesLoss

logger = logging.getLogger(__name__)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    return Embedding(num_embeddings, embed_dim, padding_idx)


class CTCDecoder(FairseqDecoder):
    def __init__(self, dictionary, embed_dim, task, dropout_rate=0.0):
        super().__init__(dictionary)
        self.blank_idx = (
            dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = dictionary.pad()
        self.eos_idx = dictionary.eos()
        self.dropout_module = FairseqDropout(dropout_rate)
        self.proj = Linear(embed_dim, len(dictionary))
        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
    ):
        x = encoder_out["encoder_out"][0].transpose(0, 1) # B x T x D
        x = self.proj(self.dropout_module(x))
        return x.transpose(0, 1), {"attn": [], "inner_states": None}


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)


class TransformerLinearEncoder(FairseqEncoder):
    def __init__(self, model_args, dictionary, enc_emb):
        super().__init__(dictionary)

        self.main_encoder = TransformerEncoder(model_args, dictionary, enc_emb)
        self.dropout_module = FairseqDropout(model_args.dropout)
        self.proj = Linear(model_args.encoder_embed_dim, len(dictionary))

    def forward(self, src_tokens, src_lengths):
        ret = self.main_encoder(src_tokens, src_lengths)
        x = ret["encoder_out"][0]
        x = self.proj(self.dropout_module(x.transpose(0, 1))) # B x T x V
        ret["encoder_out"] = [x.transpose(0, 1)]
        return ret


class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        text_encoder,
        dictionary,
    ):
        super().__init__(dictionary)

        self.spch_encoder = spch_encoder
        self.text_encoder = text_encoder
        self.shrink_speech_output = getattr(args, "shrink_speech_output", False)
        self.zero_speech_output = getattr(args, "zero_speech_output", False)
        self.use_linear_after_encoder = getattr(args, "use_linear_after_encoder", False)
        self.compute_ot_plan = getattr(args, "compute_ot_plan", False)

    @classmethod
    def build_speech_encoder(cls, args):
        cfg = {
            "input_feat_per_channel": args.input_feat_per_channel,
            "input_channels": args.input_channels,
            "conv_kernel_sizes": args.conv_kernel_sizes,
            "conv_channels": args.conv_channels,
            "encoder_embed_dim": args.encoder_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.speech_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "layernorm_embedding": args.layernorm_embedding,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
            "encoder_freezing_updates": 0,
        }
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        spch_encoder = S2TTransformerEncoder(model_args)
        return spch_encoder

    @classmethod
    def build_text_encoder(cls, args, src_dictionary, spch_encoder):
        cfg = {
            "encoder_embed_dim": args.encoder_text_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.text_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "encoder_learned_pos": args.encoder_learned_pos,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "adaptive_input": args.adaptive_input,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
        }
        text_encoder = None
        if getattr(args, "no_text_encoder", False):
            return text_encoder

        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        enc_emb = build_embedding(src_dictionary, model_args.encoder_embed_dim)
        if not getattr(args, "use_linear_after_encoder", False):
            text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)
        else:
            text_encoder = TransformerLinearEncoder(model_args, src_dictionary, enc_emb)

        # Set shared layers
        if args.encoder_shared_layers > 0:
            start = 0 if args.encoder_shared_layers_order == 0 \
                    else -min(args.text_encoder_layers, args.speech_encoder_layers)
            end = args.encoder_shared_layers if start == 0 else 0
            for i in range(start, end):
                if isinstance(text_encoder, TransformerLinearEncoder):
                    text_encoder.main_encoder.layers[i] = spch_encoder.transformer_layers[i]
                else:
                    text_encoder.layers[i] = spch_encoder.transformer_layers[i]
        return text_encoder

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        src_txt_tokens=None,
        src_txt_lengths=None,
        **kwargs
    ):
        """
        Args:
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (speech) (B,)
            src_txt_tokens: padded tensor (B, T)
            src_txt_lengths: tensor of original lengths of input utterances (text) (B,)
        """
        # src_tokens only: inference
        # src_tokens, src_lengths: speech only training
        # src_tokens, src_txt_tokens, src_txt_lengths: siamese training
        
        if src_tokens is None and src_txt_tokens is None:
            raise ValueError(
                "src_tokens and src_txt_tokens cannot be None at the same time"
            )
        ret1 = self.spch_encoder(src_tokens, src_lengths)
        ret2 = None
        if src_txt_tokens is not None and self.text_encoder is not None:
            ret2 = self.text_encoder(src_txt_tokens, src_txt_lengths)

        # if self.compute_ot_plan:
        #     p = 2
        #     blur = 0.05
        #     OT_solver = SamplesLoss(loss="sinkhorn", p=p, blur=blur, debias=False, potentials=True)
        #     speech_out = ret1["encoder_out"][0] # N x B x D
        #     text_out = ret2["encoder_out"][0] # M x B x D
        #     B = speech_out.size()[1]
        #     out = [None] * B
        #     for i in range(B):
        #         x = speech_out[:, i, :].squeeze(1) # N x D
        #         y = text_out[:, i, :].squeeze(1)
        #         N, M, D = x.size()[0], y.size()[0], x.size()[1]
        #         x_weight = OT_solver.generate_weights(x).detach() # B x N
        #         y_weight = OT_solver.generate_weights(y).detach() # B x M
        #         F, G = OT_solver(x_weight, x, y_weight, y)  # Dual potentials
        #         F = F.detach()
        #         G = G.detach()

        #         a_i, x_i = x_weight.view(N, -1), x.view(N, -1, D)
        #         b_j, y_j = y_weight.view(-1, M), y.view(-1, M, D)
        #         F_i, G_j = F.view(N, -1), G.view(-1, M)
        #         C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
        #         eps = blur**p  # temperature epsilon
        #         P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan
        #         out[i] = torch.matmul(P_ij.T, x).transpose(0, 1)
        #     ret1["encoder_out"] = [torch.nn.utils.rnn.pad_sequence(out).transpose(0, 2)]
        #     ret1["encoder_padding_mask"] = ret2["encoder_padding_mask"]

        def merge_output(rst1, rst2):
            if rst2 is None:
                return rst1
            return (rst1, rst2)

        return merge_output(ret1, ret2)

    def reorder_encoder_out(self, encoder_out, new_order):
        assert self.training is False  # used for inference only
        return self.spch_encoder.reorder_encoder_out(encoder_out, new_order)


class MultiOutputDecoder(FairseqDecoder):
    def __init__(
        self, 
        dictionary,
        speech_decoder,
        ctc_module,
    ):
        super().__init__(dictionary)
        self.speech_decoder = speech_decoder
        self.ctc_module = ctc_module

    def forward(
        self, 
        prev_output_tokens, 
        encoder_out, 
        incremental_state=None, 
        **kwargs
    ):
        speech_dec_out = self.speech_decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **kwargs
        )
        ctc_out = self.ctc_module(
            prev_output_tokens,
            encoder_out=encoder_out,
            **kwargs
        )
        x = (speech_dec_out[0], ctc_out[0])
        extra = speech_dec_out[1]
        return x, extra


@register_model("siamese_st2t_transformer")
class SiameseST2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # encoder 1: S2TTransformerEncoder for speech
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        # standard Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-text-embed-dim",
            type=int,
            metavar="N",
            help="encoder text embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        # non-standard transformer parameters
        parser.add_argument(
            "--speech-encoder-layers",
            type=int,
            metavar="N",
            help="num speech encoder layers",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            metavar="N",
            help="num text encoder layers",
        )
        parser.add_argument(
            "--load-pretrain-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained encoder """,
        )
        parser.add_argument(
            "--load-pretrain-speech-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder-last",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-speech-decoder",
            type=str,
            metavar="EXPR",
            default="",
            help=""" path to the pretrained speech decoder """,
        )
        parser.add_argument(
            "--load-pretrain-decoder",
            type=str,
            metavar="EXPR",
            default="",
            help=""" path to the pretrained encoder """,
        )
        # additional parameters for Siamese encoders
        parser.add_argument(
            "--no-text-encoder",
            action="store_true",
            help="Do not use text encoder"
        )
        parser.add_argument(
            "--no-decoder",
            action="store_true",
            help="Do not use decoder"
        )
        parser.add_argument(
            "--use-ctc-module",
            action="store_true",
            help="Use CTC module (Linear + Softmax) after the encoder"
        )
        parser.add_argument(
            "--use-speech-decoder",
            action="store_true",
            help="Use speech decoder"
        )
        parser.add_argument(
            "--use-linear-after-encoder",
            action="store_true",
            help="Add Linear after the text encoder"
        )
        parser.add_argument(
            "--encoder-shared-layers",
            type=int,
            default=0,
            metavar="N",
            help="num shared encoder layers",
        )
        parser.add_argument(
            "--encoder-shared-layers-order",
            type=int,
            choices=[0, -1],
            metavar="N",
            help="0: shared top N layers from 0:N \
                -1: shared bottom N layers from -N:",
        )
        parser.add_argument(
            "--share-speech-text-encoder-embed",
            action="store_true",
            help="",
        )
        parser.add_argument(
            "--shrink-speech-output",
            action="store_true",
            help="Shrink speech encoder's output based on CTC module"
        )
        parser.add_argument(
            "--zero-speech-output",
            action="store_true",
            help="Zero out speech encoder's output based on CTC module"
        )
        parser.add_argument(
            "--share-text-encoder-ctc-decoder-input-output",
            action="store_true",
            help="share text encoder embed and ctc output layer",
        )
        parser.add_argument(
            "--compute-ot-plan",
            action="store_true",
            help="Comupute optimal transport plan"
        )  

    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder = SiameseSpeechTextEncoders.build_speech_encoder(args)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(
            args, task.src_dict, spch_encoder
        )
        encoder = SiameseSpeechTextEncoders(
            args,
            spch_encoder,
            text_encoder,
            task.src_dict,
        )

        if args.load_pretrain_speech_encoder != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrain_speech_encoder)
            ckpt_component_type = ["encoder.spch_encoder"] \
                if any([key.startswith("encoder.spch_encoder") for key in state["model"].keys()]) else ["encoder"]
            checkpoint_utils.load_pretrained_component_from_model_different_keys(
                spch_encoder, state, ckpt_component_types=ckpt_component_type)
            logging.info(f"Loaded pretrained speech encoder from {args.load_pretrain_speech_encoder}")

        if getattr(args, "load_pretrain_text_encoder_last", "") != "":
            # if share encoder, speech encoder parameters will be used.
            # It provides a chance to use pre-trained mt encoder instead
            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrain_text_encoder_last)
            # check if language pairs in state
            multi_dec = False
            lang_pair = None
            for key in state["model"].keys():
                multi_dec = True if len(key.split(".")[1].split("-")) == 2 else False
                lang_pair = key.split(".")[1]
                if multi_dec:
                    break    
            ckpt_component_type = [f"models.{lang_pair}.encoder", "models.encoder"] \
                if multi_dec else ["models.encoder"]
            checkpoint_utils.load_pretrained_component_from_model_different_keys(
                    text_encoder, state, ckpt_component_types=ckpt_component_type)
            logging.info(f"Loaded pretrained text encoder last from {args.load_pretrain_text_encoder_last}")

        if getattr(args, "load_pretrain_encoder", "") != "":
            checkpoint_utils.load_pretrained_component_from_model(
                encoder, args.load_pretrain_encoder)
            logging.info(f"Loaded pretrained encoder from {args.load_pretrain_encoder}")

        return encoder

    @classmethod
    def build_decoder(cls, args, task, encoder):
        dec_cfg = {
            "decoder_layerdrop": args.decoder_layerdrop,
            "share_decoder_input_output_embed": args.share_decoder_input_output_embed,
            "decoder_embed_dim": args.decoder_embed_dim,
            "max_target_positions": args.max_target_positions,
            "dropout": args.dropout,
            "encoder_learned_pos": args.encoder_learned_pos,
            "decoder_learned_pos": args.decoder_learned_pos,
            "layernorm_embedding": args.layernorm_embedding,
            "decoder_normalize_before": args.decoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "decoder_ffn_embed_dim": args.decoder_ffn_embed_dim,
            "decoder_layers": args.decoder_layers,
            "decoder_attention_heads": args.decoder_attention_heads,
            "decoder_output_dim": args.decoder_embed_dim,
            "no_scale_embedding": args.no_scale_embedding,
            "adaptive_input": args.adaptive_input,
            "quant_noise_pq": args.quant_noise_pq,
            "adaptive_softmax_cutoff": args.adaptive_softmax_cutoff,
            "tie_adaptive_weights": args.tie_adaptive_weights,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
        }
        if getattr(args, "no_decoder", False):
            return DummyDecoder(task.target_dictionary) # dummy decoder

        dec_cfg = namedtuple("args", dec_cfg.keys())(*dec_cfg.values())
        ctc_module = None
        if getattr(args, "use_ctc_module", False):
            ctc_module = CTCDecoder(
                task.source_dictionary,
                dec_cfg.decoder_embed_dim,
                task,
                dec_cfg.dropout,
            )
        if getattr(args, "share_text_encoder_ctc_decoder_input_output", False):
            assert ctc_module is not None and encoder.text_encoder is not None
            ctc_module.proj.weight = encoder.text_encoder.embed_tokens.weight

        speech_decoder = None
        if getattr(args, "use_speech_decoder", False):
            dec_emb = build_embedding(task.target_dictionary, args.decoder_embed_dim)
            speech_decoder = TransformerDecoderScriptable(
                dec_cfg, task.target_dictionary, dec_emb)

        if getattr(args, "share_speech_text_encoder_embed", False):
            assert speech_decoder is not None
            encoder.text_encoder.embed_tokens = speech_decoder.embed_tokens
            encoder.text_encoder.embed_positions = speech_decoder.embed_positions

        if getattr(args, "load_pretrain_speech_decoder", "") != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrain_speech_decoder)
            # check if language pairs in state
            multi_dec = False
            lang_pair = None
            for key in state["model"].keys():
                multi_dec = True if len(key.split(".")[1].split("-")) == 2 else False
                lang_pair = key.split(".")[1]
                if multi_dec:
                    break
            
            ckpt_component_type = [f"models.{lang_pair}.decoder", "models.decoder"] \
                if multi_dec else ["models.decoder"]
            checkpoint_utils.load_pretrained_component_from_model_different_keys(
                    speech_decoder, state, ckpt_component_types=ckpt_component_type)
            logging.info(f"Loaded pretrained decoder from {args.load_pretrain_speech_decoder}")

        if ctc_module is not None and speech_decoder is not None:
            decoder = MultiOutputDecoder(task.target_dictionary,
                                         speech_decoder,
                                         ctc_module)
        else:
            assert not (ctc_module is None and speech_decoder is None)
            decoder = ctc_module if ctc_module is not None else speech_decoder 

        return decoder

    @classmethod
    def build_model(cls, args, task):
        # torch.autograd.set_detect_anomaly(True)
        # make sure that all args are properly defaulted
        siamese_st2t_transformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, encoder)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None, idx=0):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample, idx=idx)
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        idx=0,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        assert not isinstance(self.decoder, DummyDecoder)
        # if isinstance(self.decoder, MultiOutputDecoder):
        #     net_output = (net_output[0][idx], net_output[1])
        
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens=None,
        use_encoder_outputs=False,
        src_txt_tokens=None,
        src_txt_lengths=None,
        **kwargs
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths,
            **kwargs
        )
        if isinstance(self.decoder, DummyDecoder):
            return None, encoder_out

        decoder_input = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=decoder_input,
            **kwargs
        )

        def zero_speech_output(speech_out, preds):
            """
            Zero elements in x (corresponding to repeated consecutive 
                                values or blank index in preds)
            Args:
                x: T x B x D
                preds: T x B
            """
            preds = preds.double()
            D = speech_out.size()[-1]        
            T, B = preds.size()
            # get indices to be removed (blank tokens) and merge repeated predictions into 1
            # construct a difference matrix having below format  
            # [[ 1,  0,  0,  0,  ...],
            #  [-1,  1,  0,  0,  ...],
            #  [ 0, -1,  1,  0,  ...],
            #  [ 0,  0, -1,  1,  ...],
            diff_matrix = (
                torch.triu(torch.tril(torch.ones(T, T)*-1), -1) + torch.eye(T) * 2
            ).to(preds.device).double() # T x T
            diff_preds = torch.matmul(diff_matrix, preds) # T x B
            blank_idx = self.decoder.blank_idx if isinstance(self.decoder, CTCDecoder) \
                else self.decoder.ctc_module.blank_idx
            m = ~(preds.eq(blank_idx) | diff_preds.eq(0)) # T x B
            reduced_t = torch.numel(m) - torch.sum(m)
            m = m.transpose(0, 1).unsqueeze(2).expand(-1, -1, D) # B x T x D
            speech_out = speech_out.transpose(0, 1) * m # B x T x D
            return speech_out.transpose(0, 1), reduced_t / (T*B)

        def pad_seq_given_lens_arrays(input, lengths, padding_value=0.0):
            """
            Reshape and pad an input tensor given lengths of each chunk in the tensor
            """
            cum_len = 0
            y = []
            for _, val in enumerate(lengths):
                y.append(input[cum_len : cum_len+val])
                cum_len += val
            return torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=padding_value)

        def shrink_speech_output(speech_out, preds):
            """
            Average elements in x correponsing to repeated consecutive values in preds 
            Args:
                x: T x B x D
                preds: T x B
            """
            # iterate through batch dimension
            T, B, D = speech_out.size()
            speech_out = speech_out.transpose(0, 1)
            preds = preds.transpose(0, 1)
            Y = []
            preds_after_merged = []
            reduced_t = 0
            for i in range(B):
                p, c = preds[i].unique_consecutive(return_counts=True, dim=0)
                # create a padded tensor of shape num_chunks x max_len_chunks x D
                padded = pad_seq_given_lens_arrays(speech_out[i], c) # N x S x D
                # sum over each chunk and divide by lengths
                out = torch.sum(padded, dim=1) / c.unsqueeze(-1).expand(-1, D)
                Y.append(out)
                preds_after_merged.append(p)
                reduced_t += torch.sum(c[~c.eq(1)]) - torch.numel(c[~c.eq(1)])
            
            Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True) # B x T x D
            preds_after_merged = torch.nn.utils.rnn.pad_sequence(
                            preds_after_merged, 
                            batch_first=True,
                            padding_value=self.decoder.pad_idx,
                            )
            # Get mask of elements which are blank
            non_blank_mask = ~preds_after_merged.eq(self.decoder.blank_idx)
            # if preds_after_merged are all blank then not reducing
            non_blank_mask = ~non_blank_mask if torch.all(~non_blank_mask) else non_blank_mask
            reduced_t += torch.sum(~non_blank_mask)
            # Get new lengths
            lengths = torch.sum(non_blank_mask, dim=-1)
            Y = Y.masked_select(non_blank_mask.unsqueeze(-1)).view(-1, D)
            Y = pad_seq_given_lens_arrays(Y, lengths)
            return Y.transpose(0, 1), reduced_t / (T*B)

        if not self.encoder.use_linear_after_encoder:
            speech_out = encoder_out[0]["encoder_out"][0] if isinstance(encoder_out, tuple) \
                else encoder_out["encoder_out"][0] # T x B x D
        else:
            assert isinstance(self.decoder, CTCDecoder)
            speech_out = decoder_out[0] # T x B x V
        x = speech_out
        
        # Shrink speech output
        if self.encoder.shrink_speech_output or self.encoder.zero_speech_output:
            assert isinstance(self.decoder, CTCDecoder) or isinstance(self.decoder, MultiOutputDecoder)
            ctc_out = decoder_out[0] if isinstance(self.decoder, CTCDecoder) else decoder_out[0][1]
            lprobs_ctc = F.log_softmax(ctc_out, dim=-1).contiguous() # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1).contiguous() # T x B

            if self.encoder.zero_speech_output:
                x, reduced_t = zero_speech_output(speech_out, preds)
            elif self.encoder.shrink_speech_output:
                x, reduced_t = shrink_speech_output(speech_out, preds)
            else:
                raise NotImplementedError

            decoder_out[-1]["reduced_speech_output"] = reduced_t

        if isinstance(encoder_out, tuple):
            encoder_out[0]["encoder_out"] = [x] # T x B x D or T x B x V
        else:
            encoder_out["encoder_out"] = [x] # T x B x D or T x B x V

        if use_encoder_outputs:
            return decoder_out, encoder_out

        return decoder_out 


@register_model_architecture(
    "siamese_st2t_transformer", "siamese_st2t_transformer_base"
)
def siamese_st2t_transformer_base(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_text_embed_dim = getattr(
        args, "encoder_text_embed_dim", args.encoder_embed_dim
    )
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_s")
def siamese_st2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_text_embed_dim = getattr(args, "encoder_text_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    siamese_st2t_transformer_base(args)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_m")
def siamese_st2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    siamese_st2t_transformer_base(args)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_m_post_norm")
def siamese_st2t_transformer_m_post_norm(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    siamese_st2t_transformer_base(args)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_mb")
def siamese_st2t_transformer_mb(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    siamese_st2t_transformer_base(args)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_mb_s")
def siamese_st2t_transformer_mb_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.15)
    siamese_st2t_transformer_base(args)


@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_l")
def siamese_st2t_transformer_l(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 18)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    siamese_st2t_transformer_base(args)

@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer_lp")
def siamese_st2t_transformer_lp(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 18)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    siamese_st2t_transformer_base(args)