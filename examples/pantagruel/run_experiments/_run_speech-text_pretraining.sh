#!/bin/bash

################################################################
##### SPEECH and TEXT
##### Task: pre-training 
##### Dataset: English LibriSpeech and English Wikipedia
################################################################

# 1. Prepare data: same as speech-only and text-only pretraining

# 2. Run PRE-TRAINING
PARTITION=gpu_p2
TASK=pretraining
MODALITY=speech-text
FAIRSEQ=$HOME/code/fairspeech
# USER_DIR=$FAIRSEQ/examples/data2vec
USER_DIR=$FAIRSEQ/examples/pantagruel
TIME_LIMIT=1180
HOURS=20
JOBS=2
GPUs=16
SUFFIX=

MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# CONFIG=base_speech
# CONFIG=base_text
# CONFIG=base_speech_dummy_text_factor0.0
# CONFIG=base_speech_dummy_text_factor0.01
CONFIG=base-speech-text-dfactor0.0

## Data
if [[ $PARTITION != "mi250" ]]; then
    # Jean zay
    DATA_ROOT=/gpfswork/rech/ahm/umz16dj/Data
else
    ## Adastra
    DATA_ROOT=/lus/home/CT10/c1615074/tphle/Data/prepared
fi
AUDIO_DATA=$DATA_ROOT/LibriSpeech
TEXT_DATA=$DATA_ROOT/Wikipedia/enwiki_20240201/data-bin/gpt2_bpe
if [[ $SUFFIX == *"debug"* ]]; then
    AUDIO_DATA=$AUDIO_DATA/data_small
    TEXT_DATA=$TEXT_DATA/data_small
fi

# ===== CHECK THIS =====
EXPNAME="${CONFIG}_${PARTITION}_gpus${GPUs}${SUFFIX}"

if [[ $SUFFIX == *"debug" ]]; then
    AUDIO_DATA=$AUDIO_DATA/debug
    TEXT_DATA=$TEXT_DATA/debug/data-bin
fi

# Jean zay and Adastra
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

echo "=== EXP_NAME: ${EXPNAME} ==="

export TMPDIR=$SCRATCH/tmp
mkdir -p $TMPDIR

# audio-only or audio-text_dummy
submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.user_dir=${USER_DIR} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=$AUDIO_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

# text-only or text-audio_dummy
submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.user_dir=${USER_DIR} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.text.data=$TEXT_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

# joint speech-text
submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.user_dir=${USER_DIR} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=$AUDIO_DATA task.text.data=$TEXT_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"



# base_speech_gpu_p2_gpus16

# base_text_gpu_p2_gpus16
job 0: 683028
job 1: 683029 (after 683028)

# base_speech_dummy_text_factor0.0_gpu_p2_gpus16
job 0: 675609
job 1: 675610 (after 675609)

# base_speech_dummy_text_factor0.01_gpu_p2_gpus16
job 0: 675614
job 1: 675615 (after 675614)
