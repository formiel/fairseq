#!/bin/bash


################################################################
##### SPEECH
##### Task: Fine-tuning for ASR
##### Arch: Using Transformer decoder with cross-entropy loss
##### Dataset: on CommonVoice/fr datasets
################################################################

# parsing by command-line args
TASK=finetuning
DATA=$WORK/Data/CommonVoice/fr
LABEL="fr.ltr"
TIME_LIMIT=470

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/speech/${TASK}
CKPT_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/speech

# Base architecture
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_fr
CONFIG=base_commonvoice_w2v_transf_decoder
GPUS=8

# Large architecture
# PRETRAIN_CONFIG=large_audio_only_task_ngpu32_freq2_fr_bsz_lr
# CONFIG=large_commonvoice
# GPUS=24

PRETRAIN_CKPT=$CKPT_DIR/pretraining/${PRETRAIN_CONFIG}/checkpoint_best.pt

EXPNAME="${CONFIG}_fr.ltr" # to be changed
TENSORBOARD=${TENSORBOARD_DIR}/${EXPNAME}
SAVE_DIR=${CKPT_DIR}/${TASK}/${EXPNAME}
submit run gpu_p5 $GPUS 8 1 $CONFIG "fairseq-hydra-train task.data=$DATA task.labels=${LABEL} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD} common.user_dir=examples/data2vec checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$PRETRAIN_CKPT --config-dir $CONFIG_DIR --config-name ${CONFIG}.yaml"

# base_commonvoice_w2v_transf_decoder_fr.ltr
# (look better than using CTC in the beginning, maybe because of beam search decoding)

# Run decoding for model trained with CE