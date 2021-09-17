#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import numpy as np
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from fairseq.data.audio.audio_utils import get_waveform
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

import fairseq

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "tgt_lang",
                    "src_text", "src_lang"]


class mTEDx(Dataset):
    """
    Create a Dataset for Multilingual TEDx.
    Each item is a tuple of the form: waveform, sample_rate, source utterance,
    target utterance, speaker_id, utterance_id
    """

    SPLITS    = ["train", "valid", "test"]
    LANGPAIRS = ["es-es","fr-fr","pt-pt","it-it","ru-ru","el-el","ar-ar","de-de",
                 "es-en","es-fr","es-pt","es-it","fr-en","fr-es","fr-pt",
                 "pt-en","pt-es","it-en","it-es","ru-en","el-en"]


    def __init__(self, root: str, lang: str, split: str,
                use_w2v_feats: bool = False,
                w2v_path: str = None,
                use_gpu: bool = True,
                normalize_signal: bool = False,
                w2v_ctc: bool = False,
                ) -> None:
        assert split in self.SPLITS and lang in self.LANGPAIRS
        _root = Path(root) / f"{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        self.use_w2v_feats = use_w2v_feats
        self.gpu = torch.cuda.current_device() if use_gpu else None
        self.normalize_signal = normalize_signal
        self.w2v_ctc = w2v_ctc
        if self.use_w2v_feats:
            assert os.path.isfile(w2v_path), f"{w2v_path} does not exist."
            print(f'Loading model from {w2v_path}...')
            w2v_model, _ , _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([w2v_path])
            w2v_model = w2v_model[0]
            w2v_model.eval()
            self.w2v_model = w2v_model.cuda(self.gpu) if self.gpu is not None else w2v_model

        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the Multilingual TEDx YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        src,tgt = lang.split("-")
        for _lang in [src, tgt]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            # wav_filename = wav_filename.replace(".wav", ".flac")
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src],
                        segment[tgt],
                        segment["speaker_id"],
                        tgt,
                        src,
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, tgt_lang, src_lang, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset) # 1 x T
        waveform = torch.from_numpy(waveform)
        feats = None
        if self.use_w2v_feats:
            waveform = waveform.cuda(self.gpu) if self.gpu is not None else waveform
            with torch.no_grad():
                if self.normalize_signal:
                    waveform = F.layer_norm(waveform, waveform.shape)
                if not self.w2v_ctc:
                    feats = self.w2v_model(waveform, mask=False, features_only=True)["x"] # 1 x T x D_w2v
                else:
                    feats = self.w2v_model.w2v_encoder.w2v_model.extract_features(
                                            source=waveform, padding_mask=None)["x"]
                feats = feats.squeeze(0).cpu().numpy()
        return waveform, feats, sr, src_utt, tgt_utt, spk_id, tgt_lang, src_lang, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    lang_pairs = mTEDx.LANGPAIRS
    # Get wanted language pairs
    if args.src is not None:
        _pairs = []
        src_lgs = args.src.split(",")
        all_srcs = [p.split("-")[0] for p in lang_pairs]
        all_tgts = [p.split("-")[-1] for p in lang_pairs]
        for i, s in enumerate(all_srcs):
            if s in src_lgs:
                _pairs.append(f"{s}-{all_tgts[i]}")
        lang_pairs = _pairs
    if args.tgt is not None:
        _pairs = []
        tgt_lgs = args.tgt.split(",")
        all_srcs = [p.split("-")[0] for p in lang_pairs]
        all_tgts = [p.split("-")[-1] for p in lang_pairs]
        for i, t in enumerate(all_tgts):
            if t in tgt_lgs:
                _pairs.append(f"{all_srcs[i]}-{t}")
        lang_pairs = _pairs

    for lang in lang_pairs:
        cur_root = root / f"{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        feature_root = cur_root / "fbank80" if not args.use_w2v_feats else cur_root / "w2v2_feats"
        feature_root.mkdir(exist_ok=True)
        if not args.get_manifest_only:
            for split in mTEDx.SPLITS:
                print(f"Fetching split {split}...")
                dataset = mTEDx(root.as_posix(), lang, split,
                                use_w2v_feats=args.use_w2v_feats,
                                w2v_path=args.w2v_path,
                                use_gpu=args.use_gpu,
                                normalize_signal=args.normalize_signal,
                                w2v_ctc=args.w2v_ctc)
                print("Extracting log mel filter bank or wav2vec features...")
                if not args.use_w2v_feats:
                    for waveform, _, sample_rate, _, _, _, _, _, utt_id in tqdm(dataset):
                        extract_fbank_features(
                            waveform, sample_rate, feature_root / f"{utt_id}.npy"
                        )
                else:
                    for _, features, sample_rate, _, _, _, _, _, utt_id in tqdm(dataset):
                        output_path = feature_root / f"{utt_id}.npy"
                        np.save(output_path.as_posix(), features)
        # Pack features into ZIP
        zip_path = cur_root / "fbank80.zip" if not args.use_w2v_feats else cur_root / "w2v2_feats.zip"
        if not args.get_manifest_only:
            print("ZIPing features...")
            create_zip(feature_root, zip_path)
            print("Fetching ZIP manifest...")
            zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in mTEDx.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = mTEDx(root.as_posix(), lang, split,
                            use_w2v_feats=args.use_w2v_feats,
                            w2v_path=args.w2v_path,
                            use_gpu=args.use_gpu,
                            normalize_signal=args.normalize_signal,
                            w2v_ctc=args.w2v_ctc)
            for wav, _, sr, src_utt, tgt_utt, speaker_id, tgt_lang, src_lang, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                if not args.get_manifest_only:
                    manifest["audio"].append(zip_manifest[utt_id])
                else:
                    manifest["audio"].append("")
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
                manifest["speaker"].append(speaker_id)
                manifest["tgt_lang"].append(tgt_lang)
                manifest["src_text"].append(src_utt)
                manifest["src_lang"].append(src_lang)
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, 
                                    is_train_split=is_train_split, 
                                    no_audio=args.get_manifest_only)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")

        if not args.get_manifest_only:
            # Generate vocab
            v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
            spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
            with NamedTemporaryFile(mode="w") as f:
                for t in train_text:
                    f.write(t + "\n")
                gen_vocab(
                    Path(f.name),
                    cur_root / spm_filename_prefix,
                    args.vocab_type,
                    args.vocab_size,
                )
            # Generate config YAML
            input_feat_per_channel = 80
            if args.use_w2v_feats:
                input_feat_per_channel = np.load(os.path.join(feature_root,
                                                    os.listdir(feature_root)[0])).shape[1]
            gen_config_yaml(
                cur_root,
                spm_filename_prefix + ".model",
                yaml_filename=f"config_{args.task}.yaml",
                specaugment_policy=None,
                input_feat_per_channel=input_feat_per_channel,
            )
        # Clean up
        shutil.rmtree(feature_root)


def process_joint(args):
    cur_root = Path(args.data_root)
    assert all((cur_root / f"{lang}").is_dir() for lang in mTEDx.LANGPAIRS), \
        "do not have downloaded data available for all languages"
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for lang in mTEDx.LANGPAIRS:
            tsv_path = cur_root / f"{lang}" / f"train_{args.task}.tsv"
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        special_symbols = None
        if args.joint:
            special_symbols = list(set([f'<lang:{lang.split("-")[1]}>' for lang in mTEDx.LANGPAIRS]))  #add tgt_lang tags to dict
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols
        )
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{args.task}.yaml",
        prepend_tgt_lang_tag=(args.joint),
    )
    # Make symbolic links to manifests
    for lang in mTEDx.LANGPAIRS:
        for split in mTEDx.SPLITS:
            src_path = cur_root / f"{lang}" / f"{split}_{args.task}.tsv"
            desc_path = cur_root / f"{split}_{lang}_{args.task}.tsv"
            if not desc_path.is_symlink():
                os.symlink(src_path, desc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--use-w2v-feats", action="store_true")
    parser.add_argument("--w2v-path", type=str)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--normalize-signal", action="store_true")
    parser.add_argument("--w2v-ctc", action="store_true")
    parser.add_argument("--src", type=str, default=None, 
                        help="Preprocess only language pairs including this\
                            language as a source. To include multiple languages\
                            seperated by comma.")
    parser.add_argument("--tgt", type=str, default=None, 
                        help="Preprocess only language pairs including this\
                            language as a target. To include multiple languages\
                            seperated by comma.")
    parser.add_argument("--get-manifest-only", action="store_true",
                        help="Extracting the manifest only.")
    args = parser.parse_args()

    if args.joint:
        process_joint(args)
    else:
        process(args)


if __name__ == "__main__":
    main()