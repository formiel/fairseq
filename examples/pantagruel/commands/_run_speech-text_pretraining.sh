#!/bin/bash

################################################################
##### SPEECH and TEXT
##### Task: pre-training 
##### Dataset: English LibriSpeech and English Wikipedia
################################################################

# 1. Prepare data: same as speech-only and text-only pretraining

# 2. Run PRE-TRAINING
PARTITION=mi250
TASK=pretraining
MODALITY=speech-text
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=590
# TIME_LIMIT=25
HOURS=10
JOBS=5

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_text_en
CONFIG=base_speech_text_en_bsz8frq8
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
SUFFIX=
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
# base_speech_text_en_gpu_p5_gpus8_debug: ok
# base_speech_text_en_bsz8frq4_gpu_p2_gpus16_debug: ok

# base_speech_text_en_bsz8frq8_gpu_p2_gpus32: 1446934

# base_speech_text_en_gpu_p5_gpus16 (bsz=16, freq=4, lr=3e-4, local_grad_mult=2.5 for speech, 1.0 for text)
# job 0: 1433334
# job 1: 1433335 (after 1433334)
# job 2: 1433336 (after 1433335)
# job 3: 1433346 (after 1433336)
# job 4: 1433347 (after 1433346)

##### Adastra #####
###################
# base_speech_text_en_bsz8frq8_mi250_gpus8_debug: ok
# base_speech_text_en_bsz8frq8_mi250_gpus16_debug: ok

# base_speech_text_en_bsz8frq8_mi250_gpus16
# job 0: 790417
# job 1: 790418 (after 790417)
# job 2: 790420 (after 790418)
# job 3: 790421 (after 790420)
# job 4: 790422 (after 790421)


