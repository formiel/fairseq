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
GPUs=16
SUFFIX=

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4
# CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_layerdrop0.05_clipnorm10
# CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_layerdrop0.05_clipnorm10_ema_encoder_only
# CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_ema_encoder_only


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

submit run ${PARTITION} $GPUs ${HOURS} ${JOBS} $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} task.audio.data=$AUDIO_DATA task.text.data=$TEXT_DATA distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT} optimization.stop_time_hours=${TIME_LIMIT}"

##### Jean Zay #####
####################
# speech-only pre-training: 
# lr: 0.00075 loaded 280531, skipped 710 samples
# max tokens per device = 1000000 and max sentences per device = None
# trained on 16 GPUs
# grouped total_num_itrs = 3876

# text-only pre-training: 
# max tokens per device = None and max sentences per device = 6
# lr: 0.0003 
# trained on 16 GPUs
# grouped total_num_itrs = 96111


##### Adastra #####
###################
# base_speech_text_en_bsz8frq1_mi250_gpus64: 
# speech: loaded 280531, skipped 710 samples
# dataset Modality.AUDIO batch number is 35066
# dataset Modality.TEXT batch number is 1153330 
# trained on 64 GPUs
# grouped total_num_itrs = 18567

# base_speech_text_en_maxtok500k_bsz6_mi250_gpus16 (lr=3e-4)
# max tokens per device = 500000 and max sentences per device = 6
# dataset Modality.AUDIO batch number is 140455 (140455/16=8778) 
# dataset Modality.TEXT batch number is 1537774 (1537774/16=96111)
# grouped total_num_itrs = 104888
# job 0: 812422

# base_speech_text_en_maxtok500k_bsz6_frq1_mi250_gpus16
# job 0: 813835, then pred_var_AUDIO is 0.00991882011294365 < 0.01

# base_speech_text_en_maxtok500k_bsz6_frq1_layerdrop0.05_mi250_gpus16
# job 0: 814742

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_mi250_gpus16
# job 0: 814635 (done)
# job 0: 814819 (done)
# job 0: 816914
# job 1: 816915 (after 816914)
# job 2: 816916 (after 816915)
# job 3: 816917 (after 816916)
# job 4: 816918 (after 816917)

# base_speech_text_en_maxtok500k_bsz6_frq1_gradclip10_mi250_gpus16
# job 0: 814745

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_layerdrop0.05_clipnorm10_mi250_gpus16
# job 0: 814849

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_layerdrop0.05_clipnorm10_ema_encoder_only_mi250_gpus16
# job 0: 815031

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_ema_encoder_only_mi250_gpus16
# job 0: 815209 (Watchdog caught collective operation timeout: WorkNCCL)

# base_speech_text_en_maxtok500k_bsz6_frq2_mi250_gpus16 (lr=3e-5, frq=2)
# job 0: 814141

# base_speech_text_en_maxtok500k_bsz6_frq1_audio10_mi250_gpus16 (FloatingPointError: Minimum loss scale reached (1e-06). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.)
# job 0: 814734

# base_speech_text_en_maxtok500k_bsz6_frq1_mi250_gpus32
# job 0: 814726