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
# TIME_LIMIT=1190
# TIME_LIMIT=1430
TIME_LIMIT=710
HOURS=5
JOBS=1
GPUs=16
SUFFIX=

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# CONFIG=base_speech_text_en_debug
# CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4
# CONFIG=base_speech_text_en_maxtok100k_bsz16_frq1_lr3e-4_maxupdate1M
CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M
# CONFIG=base_speech_text_en_maxtok1000k_bsz16_frq1_lr3e-4_wo_lr_cycles_maxupdate1M
# CONFIG=base_speech_text_en_maxtok1000k_bsz16_frq1_lr3e-4_wo_lr_cycles_maxupdate1M
CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_ema_encoder_only


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

# base_speech_text_en_maxtok1000k_bsz16_frq1_lr3e-4_wo_lr_cycles_maxupdate1M_gpu_p5_gpus16_dummy_random
# job 0: 1552709
# job 1: 1552710 (after 1552709)
# job 2: 1552711 (after 1552710)
# job 3: 1552712 (after 1552711)
# job 4: 1552713 (after 1552712)

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


##### No dummy ops
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_mi250_gpus16_no_dummy
# job 0: 817506 (done)
# job 0: 817917 (running)

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_no_dummy

##### Dummy operations in modality-specific encoders before transformer blocks: with or without
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_mi250_gpus16
# job 0: 814635 (done)
# job 0: 814819 (done)
# job 0: 816914 (done)
# job 0: 817922 (running)

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16
# job 0: 817219 (running)

##### Dummy operations in modality-specific encoders and decoder
####### zero input tensors
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_mi250_gpus16_dummy_ops_in_encoder_decoder
# job 0: 817926 (running)

####### random input tensors
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
# job 0: 818323 (done) (10h/1job)
# job 1: 818324 (done) (10h/1job)
# job 2: 818325 (done) (10h/1job)
# job 0: 827245 (done)
# job 1: 827246 (after 827245)
# job 2: 827247 (after 827246)
# job 3: 827248 (after 827247)
# job 4: 827249 (after 827248)

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_apex_fp16_memory
# job 0: 822611

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_apex
# job 0: 828194

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_ema_encoder_only_mi250_gpus16
# job 0: 829704