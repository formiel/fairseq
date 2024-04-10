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
USER_DIR=$FAIRSEQ/examples/data2vec
# TIME_LIMIT=1430
TIME_LIMIT=1190
HOURS=2
JOBS=1

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_text_en
CONFIG=base_speech_text_en_bsz8frq4
GPUs=16

## Data
if [[ $PARTITION != "mi250" ]]; then
    # Jean zay
    DATA_ROOT=/gpfswork/rech/ahm/umz16dj/Data
else
    ## Adastra
    DATA_ROOT=/lus/home/CT10/c1615074/tphle/Data/prepared
fi
AUDIO_DATA=$DATA_ROOT/LibriSpeech
TEXT_DATA=$DATA_ROOT/Wikipedia/enwiki_20240201/data-bin/byteBPE

# ===== CHECK THIS =====
SUFFIX=_debug
EXPNAME="${CONFIG}_${PARTITION}_gpus${GPUs}${SUFFIX}"

# Jean zay and Adastra
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

echo "=== EXP_NAME: ${EXPNAME} ==="

submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=$AUDIO_DATA task.text.data=$TEXT_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

##### Jean Zay #####
####################
# base_speech_text_en_gpu_p5_gpus1_debug_full: finished and save first checkpoint at 10k updates
# base_speech_text_en_gpu_p5_gpus8_debug: 1440658, can run without error 
# base_speech_text_en_bsz8frq4_gpu_p2_gpus16_debug: 1445080
# base_speech_text_en_gpu_p5_gpus16: 1432598
# job 0: 1433334
# job 1: 1433335 (after 1433334)
# job 2: 1433336 (after 1433335)
# job 3: 1433346 (after 1433336)
# job 4: 1433347 (after 1433346)

##### Adastra #####
###################
# base_speech_text_en_200k_mi250_gpus8
# job 0: 788390

