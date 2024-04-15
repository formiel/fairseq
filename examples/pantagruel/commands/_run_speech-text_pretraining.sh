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
HOURS=0.25
JOBS=1
GPUs=16
SUFFIX=_debug

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_text_en_bsz16frq16
CONFIG=base_speech_text_en_bsz1frq1
# CONFIG=base_speech_text_en_bsz16frq16_worker1


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

submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=$AUDIO_DATA task.text.data=$TEXT_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

##### Jean Zay #####
####################
# base_speech_text_en_gpu_p5_gpus1_debug_full: finished and save first checkpoint at 10k updates
# base_speech_text_en_gpu_p5_gpus8_debug: ok
# base_speech_text_en_bsz8frq4_gpu_p2_gpus16_debug: ok

# base_speech_text_en_bsz8frq8_gpu_p2_gpus32: 1446934 (GPU usage looking good with max usage observed: 31.7GB)

# base_speech_text_en_gpu_p5_gpus16 (bsz=16, freq=4, lr=3e-4, local_grad_mult=2.5 for speech, 1.0 for text)
# job 0: 1433334 (val loss seems to increase)

# base_speech_text_en_bsz16frq16_gpu_p5_gpus16
# job 0: 1507733
# job 0: 1507746
# job 1: 1507749 (after 1507746)
# job 2: 1507750 (after 1507749)
# job 3: 1507752 (after 1507750)
# job 4: 1507754 (after 1507752)

##### Adastra #####
###################
# base_speech_text_en_bsz16frq16_worker3_mi250_gpus32
# job 0: 810871
# job 1: 810872 (after 810871)
# job 2: 810873 (after 810872)
# job 3: 810874 (after 810873)
# job 4: 810875 (after 810874)

# base_speech_text_en_bsz16frq16_worker1_mi250_gpus32
# job 0: 810881
# job 1: 810882 (after 810881)
# job 2: 810883 (after 810882)
# job 3: 810884 (after 810883)
# job 4: 810885 (after 810884)