# Pre-training for Speech Translation: CTC Meets Optimal Transport

This is the codebase for the paper [Pre-training for Speech Translation: CTC Meets Optimal Transport](https://arxiv.org/abs/2301.11716).

# Table of Contents
1. [Pre-trained models](#1-pre-trained-models)
2. [Data Preparation](#2-data-preparation)
3. [Model Training](#3-model-training)
4. [Decoding](#4-decoding)
5. [References](#5-references)

# 1. Pre-trained models
## 1.1. MT models
For MuST-C bilingual MT models, we used the ones provided in [Joint Speech Text Training example](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/ende-mustc.md) and trained our multilingual MT models (for both MuST-C and CoVoST-2) ourselves.
- [MuST-C En-De](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_mt.pt), [MuST-C En-Fr](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_fr/checkpoint_mt.pt), [MuST-C one-to-many](https://zenodo.org/record/7646534/files/mustc_all_mt_pho2spm.pt?download=1)

- [CoVoST one-to-many](https://zenodo.org/record/7646534/files/covost_en2xx_mt_pho2spm.pt?download=1), [CoVoST many-to-one](https://zenodo.org/record/7646534/files/covost_xx2en_mt_spm2spm.pt?download=1)

## 1.2. ASR models
`CE`, `CTC`, `CTC+CE`, and `CTC+OT` indicate the loss(es) used during the ASR pre-training stage. `CE`, `CTC`, and `OT`, respectively, stand for cross-entropy, connectionist temporal classification, and optimal transport.
- **MuST-C**
    * En-De: [CE](https://zenodo.org/record/7645707/files/mustc_ende_asr_ce.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_ende_asr_ctc.pt?download=1), [CTC+CE](https://zenodo.org/record/7645707/files/mustc_ende_asr_ctc_ce0.1.pt?download=1), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_ende_asr_ctc_ot0.1.pt?download=1)
    * En-Fr: [CE](https://zenodo.org/record/7645707/files/mustc_enfr_asr_ce.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_enfr_asr_ctc.pt?download=1), [CTC+CE](https://zenodo.org/record/7645707/files/mustc_enfr_asr_ctc_ce0.1.pt?download=1), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_enfr_asr_ctc_ot0.1.pt?download=1)
    * One-to-many: [CE](https://zenodo.org/record/7645707/files/mustc_all_asr_ce_archl.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_all_asr_ctc_archl.pt?download=1), [CTC+CE](), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_all_asr_ctc_ot0.1_archl.pt?download=1)
<!-- TODO: re-upload ASR model for CTC+CE -->

- **CoVoST-2**
    * One-to-many: [CE](https://zenodo.org/record/7646534/files/covost_en_asr_ce.pt?download=1), [CTC](https://zenodo.org/record/7646534/files/covost_en_asr_ctc.pt?download=1), [CTC+CE](https://zenodo.org/record/7646534/files/covost_en_asr_ctc_ce0.1.pt?download=1), [CTC+OT](https://zenodo.org/record/7646534/files/covost_en_asr_ctc_ot0.1.pt?download=1), [CTC+OT large](https://zenodo.org/record/7646534/files/covost_en_asr_ctc_ot0.1_archl.pt?download=1)
    * Many-to-one: [CE](https://zenodo.org/record/7646534/files/covost_m2o_asr_ce.pt?download=1), [CTC](https://zenodo.org/record/7646534/files/covost_m2o_asr_ctc.pt?download=1), [CTC+CE](https://zenodo.org/record/7646534/files/covost_m2o_asr_ctc_ce0.1.pt?download=1), [CTC+OT](https://zenodo.org/record/7646534/files/covost_m2o_asr_ctc_ot0.1.pt?download=1)

## 1.3. ST models
`CE`, `CTC`, `CTC+CE`, and `CTC+OT` indicate the corresponding pre-trained speech encoder (obtained from the above ASR pre-trained stage) used to initialize the speech encoder in the ST model. The speech decoder is initialized using the MT decoder corresponding to each translation direction.

- **MuST-C**
    * En-De: [CE](https://zenodo.org/record/7645707/files/mustc_ende_st_init_ce_with_mt.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_ende_st_init_ctc_with_mt.pt?download=1), [CTC+CE](https://zenodo.org/record/7645707/files/mustc_ende_st_init_ctc_ce0.1_with_mt.pt?download=1), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_ende_st_init_ctc_ot0.1_with_mt.pt?download=1)
    * En-Fr: [CE](https://zenodo.org/record/7645707/files/mustc_enfr_st_init_ce_with_mt.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_enfr_st_init_ctc_with_mt.pt?download=1), [CTC+CE](https://zenodo.org/record/7645707/files/mustc_enfr_st_init_ctc_ce0.1_with_mt.pt?download=1), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_enfr_st_init_ctc_ot0.1_with_mt.pt?download=1)
    * One-to-many: [CE](https://zenodo.org/record/7645707/files/mustc_all_st_init_ce_archl_with_mt.pt?download=1), [CTC](https://zenodo.org/record/7645707/files/mustc_all_st_init_ctc_archl_with_mt.pt?download=1), [CTC+CE](), [CTC+OT](https://zenodo.org/record/7645707/files/mustc_all_st_init_ctc_ot0.1_archl_with_mt.pt?download=1)

- **CoVoST-2**
    * One-to-many: [CE](https://zenodo.org/record/7646534/files/covost_en2xx_st_init_ce_with_mt.pt?download=1), [CTC](https://zenodo.org/record/7646534/files/covost_en2xx_st_init_ctc_with_mt.pt?download=1), [CTC+CE](https://zenodo.org/record/7646534/files/covost_en2xx_st_init_ctc_ce0.1_with_mt.pt?download=1), [CTC+OT](https://zenodo.org/record/7646534/files/covost_en2xx_st_init_ctc_ot0.1_with_mt.pt?download=1), [CTC+OT large](https://zenodo.org/record/7646534/files/covost_en2xx_st_init_ctc_ot0.1_archl_with_mt.pt?download=1)
    * Many-to-one: [CE](https://zenodo.org/record/7646534/files/covost_xx2en_st_init_ce_with_mt.pt?download=1), [CTC](https://zenodo.org/record/7646534/files/covost_xx2en_st_init_ctc_with_mt.pt?download=1), [CTC+CE](https://zenodo.org/record/7646534/files/covost_xx2en_st_init_ctc_ce0.1_with_mt.pt?download=1), [CTC+OT](https://zenodo.org/record/7646534/files/covost_xx2en_st_init_ctc_ot0.1_with_mt.pt?download=1).

# 2. Data Preparation
We follow the steps for data preparation provided by [`fairseq S2T`](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text) for both MuST-C and CoVoST-2, and follow the [Joint Speech Text Training example](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/ende-mustc.md) to get phoneme representations for the transcripts. 

The commands to prepare data for MuST-C, for example, are as follows.

* Prepare the `.tsv` files for the **ST task**. 
```bash
python examples/speech_to_text/prep_mustc_data.py \
        --data-root ${MUSTC_ROOT} --task st \
        --vocab-type unigram --vocab-size 10000 \
        --langs $LANG \
        --min-n-frames 1 \
        --max-n-frames 3000
```
This results in `${SPLIT}_st.tsv` files (where `${SPLIT}` is `train`, `dev`, `tst-COMMON`, and `tst-HE`) together with the configuration and vocabulary files. Each `.tsv` file has the following collumns: `id   audio   n_frames    tgt_text    speaker src_text    src_lang    tgt_lang`.

* Prepare the `.tsv` files for the **ASR pre-training task** using SentencePiece vocabulary.
```bash
python examples/speech_to_text/prep_mustc_data.py \
        --data-root ${MUSTC_ROOT} --task asr \
        --vocab-type unigram --vocab-size 10000 \
        --langs $LANG \
        --min-n-frames 1 \
        --max-n-frames 3000
```
This command produces similar files as the above, but target text (`tgt_text`) is the transcripts instead of translation and the suffixes of the files are `_asr` instead of `_st`. 

* Get the phoneme representations of source text for the **ASR pre-training task** using phoneme vocabulary. 
    - First extract the source text from the field `src_text` of the `.tsv` files obtained from the step above. This can be done in Python for example as below.
    ```python
    from examples.speech_to_text.data_utils import load_df_from_tsv
    data = load_df_from_tsv(f"{MUSTC_ROOT}/en-{LANG}/{SPLIT}_st.tsv")
    texts = data[f"src_text"].tolist()
    with open(f"{MUSTC_ROOT}/en-{LANG}/{SPLIT}_src.txt", "w") as f:
        for line in texts:
            f.write("%s\n" % line)
    ```

    - Get the phoneme representations of the source text.
    ```bash
    python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
            --lower-case --do-filter --use-word-start --no-punc \
            --reserve-word examples/speech_text_joint_to_text/configs/mustc_noise.list \
            --data-path ${MUSTC_ROOT}/en-${LANG}${SPLIT}_src.txt \
            --out-path ${MUSTC_ROOT}/en-${LANG}${SPLIT}_src_phoneme.txt
    ```

    - Add phoneme to the `.tsv` file: the `src_text` and `tgt_text` columns need to be the transcripts in its phoneme form.
    ```python
    from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
    data = load_df_from_tsv(f"{MUSTC_ROOT}/en-{LANG}/{SPLIT}_st.tsv")
    texts = []
    with open(f"{MUSTC_ROOT}/en-{LANG}{SPLIT}_src_phoneme.txt", "r") as f:
        for line in f:
            texts.append(line.strip())
    data[f"src_text"] = texts
    data[f"tgt_text"] = texts
    save_df_to_tsv(data, f"{MUSTC_ROOT}/en-{LANG}/{SPLIT}_asr_phoneme.tsv")
    ```

# 3. Model Training
The common command for training on 1 GPU is as follow.
```bash
fairseq-train ${DATA_ROOT} \
        --config-yaml ${CONFIG_YAML} \
        --train-subset ${TRAIN_SUBSET} \
        --valid-subset ${VALID_SUBDET} \
        --save-dir ${SAVE_DIR} \
        --update-freq 8 \
        --max-tokens 40000 \
        --max-epoch ${MAX_EPOCH} \
        --warmup-updates 10000 \
        --user-dir examples/speech_text_siamese \
        --task ${FAIRSEQ_TASK} \
        --arch siamese_st2t_transformer_m \
        --criterion ${CRITERION} \
        --optimizer adam \
        --adam-betas "(0.9,0.98)" \
        --lr-scheduler inverse_sqrt \
        --lr 2e-3 \
        --clip-norm 10.0 \
        --dropout 0.2 \
        --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
        --keep-last-epochs 10 \
        --log-interval 500 \
        --skip-invalid-size-inputs-valid-test \
        --seed 1 \
        --ddp-backend no_c10d
```
where `${DATA_ROOT}`is path to where the `.tsv` files, the configuration and vocabulary files are saved. `${MAX_EPOCH}` is set to 50 and 100 for ST and ASR training stages, respectively. `${SAVE_DIR}` and ` {TENSORBOARD_LOGDIR}` are where checkpoints and tensorboard will be saved, respectively. The specific parameters for each type of training are given in their corresponding sections.

## 3.1. ASR Pre-training
### Pre-training with cross-entropy loss
```bash
CONFIG_YAML=config_unigram10000_asr.yaml
TRAIN_SUBSET=train_asr
VALID_SUBDET=dev_asr
FAIRSEQ_TASK=siamese_speech_text_to_text
CRITERION=label_smoothed_cross_entropy
```

Additional arguments to be added to the command line:
```bash
--label-smoothing 0.1 \
--report-accuracy \
--no-text-encoder \
--use-speech-decoder
```

### Pre-training with CTC loss
```bash
CONFIG_YAML=config_unigram10000_asr.yaml
TRAIN_SUBSET=train_asr
VALID_SUBDET=dev_asr
FAIRSEQ_TASK=siamese_speech_text_to_text
CRITERION=wasserstein_augmented_loss
```

Additional arguments to be added to the command line:
```bash
--no-text-encoder \
--use-ctc-module \
--ctc-weight 1.0 \
--zero-infinity
```

### Pre-training with CTC+OT loss
``bash
CONFIG_YAML=config_phoneme_asr.yaml
TRAIN_SUBSET=train_asr_pho
VALID_SUBDET=dev_asr_pho
FAIRSEQ_TASK=siamese_speech_text_to_text
CRITERION=wasserstein_augmented_loss
```

Additional arguments to be added to the command line:
```bash
--no-text-encoder \
--use-text-encoder-aux \
--use-ctc-module \
--ctc-weight 1.0 \
--zero-infinity \
--ot-weight 0.1 \
--ot-position-weight 1.0 \
--load-pretrain-text-encoder ${PATH_TO_PRETRAINED_MT_MODEL}
```

## 3.2. ST Training
```bash
CONFIG_YAML=config_unigram10000_st.yaml
TRAIN_SUBSET=train_st
VALID_SUBDET=dev_st
FAIRSEQ_TASK=siamese_speech_text_to_text
CRITERION=label_smoothed_cross_entropy
```

Additional arguments to be added to the command line:
```bash
--label-smoothing 0.1 \
--report-accuracy \
--no-text-encoder \
--use-speech-decoder \
--load-pretrain-speech-decoder ${PATH_TO_PRETRAINED_ASR_MODEL}
```

# 4. Decoding
We average the last 10 checkpoints with the following command.
```bash
python scripts/average_checkpoints.py \
                --inputs ${SAVE_DIR} \
                --num-epoch-checkpoints 10 \
                --output ${SAVE_DIR}/avg_last_10_checkpoint.pt
```

Then we run the following command for decoding.
```bash
fairseq-generate ${DATA_ROOT} \
                --config-yaml ${CONFIG_YAML} \
                --gen-subset ${GEN_SUBSET} \
                --task siamese_speech_text_to_text \
                --user-dir examples/speech_text_siamese \
                --path ${SAVE_DIR}/avg_last_10_checkpoint.pt \
                --max-tokens 50000 --beam 5 \
                --max-source-positions 60000 \
                --results-path ${SAVE_DIR}/beam5 \
                --load-speech-only \
                --scoring sacrebleu
```

# 5. References
If you find the resources in this repository useful, please cite the following paper:
```
@article{le2023pretraining,
  author    = {Phuong{-}Hang Le and
               Hongyu Gong and
               Changhan Wang and
               Juan Pino and
               Benjamin Lecouteux and
               Didier Schwab},
  title     = {Pre-training for Speech Translation: {CTC} Meets Optimal Transport},
  journal   = {CoRR},
  volume    = {abs/2301.11716},
  year      = {2023}
}
```
Our implementation is based on the `fairseq` and `fairseq S2T` toolkits.
```
@inproceedings{wang2020fairseqs2t,
  title = {fairseq S2T: Fast Speech-to-Text Modeling with fairseq},
  author = {Changhan Wang and Yun Tang and Xutai Ma and Anne Wu and Dmytro Okhonko and Juan Pino},
  booktitle = {Proceedings of the 2020 Conference of the Asian Chapter of the Association for Computational Linguistics (AACL): System Demonstrations},
  year = {2020},
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```