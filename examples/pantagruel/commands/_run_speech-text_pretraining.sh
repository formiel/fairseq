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
# CONFIG=base_speech_text_en_bsz16frq16
CONFIG=base_speech_text_en_bsz8frq32
GPUs=64

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

# base_speech_text_en_bsz8frq8_gpu_p2_gpus32: 1446934 (GPU usage looking good with max usage observed: 31.7GB)

# base_speech_text_en_gpu_p5_gpus16 (bsz=16, freq=4, lr=3e-4, local_grad_mult=2.5 for speech, 1.0 for text)
# job 0: 1433334 (val loss seems to increase)

# base_speech_text_en_bsz16frq16_gpu_p5_gpus32
# job 0: 1457298
# job 1: 1457299 (after 1457298)
# job 2: 1457300 (after 1457299)
# job 3: 1457302 (after 1457300)
# job 4: 1457303 (after 1457302)


##### Adastra #####
###################
# base_speech_text_en_bsz8frq8_mi250_gpus8_debug: ok
# base_speech_text_en_bsz8frq8_mi250_gpus16_debug: ok

# base_speech_text_en_bsz8frq8_mi250_gpus16
# job 0: 790417 #FloatingPointError: Minimum loss scale reached (1e-06). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.

# base_speech_text_en_bsz8frq8_mi250_gpus32
# job 0: 790834 #FloatingPointError: Minimum loss scale reached (1e-06). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.

# base_speech_text_en_bsz8frq8_mi250_gpus64
# job 0: 791411 Minimum loss scale reached (1e-06).

# base_speech_text_en_bsz8frq32_mi250_gpus64
# job 0: s
# job 1: 795186 (after 795184)
# job 2: 795188 (after 795186)
# job 3: 795190 (after 795188)
# job 4: 795192 (after 795190)