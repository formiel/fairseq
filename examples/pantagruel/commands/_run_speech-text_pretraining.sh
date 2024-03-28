#!/bin/bash

################################################################
##### SPEECH and TEXT
##### Task: pre-training 
##### Dataset: French MLS and French Wikipedia
################################################################

# 1. Prepare data: same as speech-only and text-only pretraining

# 2. Run PRE-TRAINING
PARTITION=gpu_p5
TASK=pretraining
MODALITY=speech-text
USER_DIR=$FAIRSEQ/examples/data2vec
# TIME_LIMIT=1430
TIME_LIMIT=110
HOURS=2
JOBS=1

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_mls1k_text_wiki19
# CONFIG=base_speech_mls1k_text_wiki19_hparams
CONFIG=base_speech_mls1k_text_wiki19_hparams-1
GPUs=16

# EXPNAME="${CONFIG}_lr3e-4" # ===== CHECK THIS =====
EXPNAME="${CONFIG}_${PARTITION}_gpus${GPUs}"

# Jean zay
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

submit run gpu_p5 $GPUs ${HOURS} ${JOBS} $EXPNAME "fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

# base_speech_mls1k_text_wiki19
# (failed due to low pred_var, max_tokens=1.4M, batch_size=16, lr=7.5e-4, update_freq=4, ngpu=16)

# base_speech_mls1k_text_wiki19_lr3e-4
# (max_tokens=1.4M, batch_size=16, lr=3e-4, update_freq=1, ngpu=32)
# (low pred_var)

# base_speech_mls1k_text_wiki19_hparams
# job 0: 939193 (pred_var too low)

# base_speech_mls1k_text_wiki19_hparams-1_gpu_p5_gpus16
# job 0: 955758 (not collapse, finished epoch1)

# Adastra
PARTITION=mi250
TASK=pretraining
MODALITY=speech-text
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=1430
HOURS=5
JOBS=1
TOKENIZER=gpt2_bpe
# TOKENIZER=byteBPE

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_ls_text_enwiki
CONFIG=base_speech_ls_text_enwiki-ema-enc
GPUs=64

AUDIO_DATA=/lus/work/CT10/c1615074/tphle/Data/prepared/MLS_French
TEXT_DATA=/lus/work/CT10/c1615074/tphle/Data/prepared/Wikipedia/enwiki_20240201/data-bin-v1/${TOKENIZER}

EXPNAME="${CONFIG}_${TOKENIZER}_${PARTITION}_gpus${GPUs}" #===== CHECK THIS =====

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/adastra/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/adastra/${MODALITY}/${TASK}/${EXPNAME}

submit run ${PARTITION} ${GPUs} ${HOURS} ${JOBS} ${EXPNAME} "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=${AUDIO_DATA} task.text.data=${TEXT_DATA} distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

# base_speech_ls_text_enwiki_mi250_gpus16
# job 0: 720290 (failed: Cannot allocate memory)

# base_speech_ls_text_enwiki_mi250_gpus32
# job 0: 720295 (failed:)

# base_speech_ls_text_enwiki-ema-enc_mi250_gpus16
# job 0: 720305 (failed: Cannot allocate memory)

# base_speech_ls_text_enwiki-ema-enc_mi250_gpus32
# job 0: 720310 (failed: )

# base_speech_ls_text_enwiki_gpt2_bpe_mi250_gpus8
# job 0: 720603 (failed)

# base_speech_ls_text_enwiki-ema-enc_gpt2_bpe_mi250_gpus64


