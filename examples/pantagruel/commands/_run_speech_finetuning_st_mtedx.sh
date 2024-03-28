#!/bin/bash

################################################################
##### FINE-TUNING WITH TRANSFORMER DECODER FOR ST TASK
##### Dataset:: mTEDx
################################################################

##### PRE-PROCESSING 
# 1.1. Split files into several files based on segments
bash $FAIRSEQ/examples/pantagruel/scripts/_split_wav_files.sh $SCRATCH/Data/mTEDx/fr-en

# 1.2. Learn dictionary
python $FAIRSEQ/examples/pantagruel/scripts/_learn_vocab.py --input-path $SCRATCH/Data/mTEDx/fr-en/data/train/txt/train.en --output-dir $WORK/Data/mTEDx/fr-en --vocab-size 1000
## character dictionary
python $FAIRSEQ/examples/pantagruel/scripts/_learn_vocab.py --input-path $SCRATCH/Data/mTEDx/fr-en/data/train/txt/train.en --output-dir $WORK/Data/mTEDx/fr-en --model-type char --model-prefix spm_char

# 1.3. Create manifest and label files
# will based on segment files and the split audios
SPLITS="valid test train"
for SPLIT in $SPLITS; do
    # python examples/pantagruel/scripts/_mtedx_manifest.py $SCRATCH/Data/mTEDx/fr-en --split ${SPLIT} --dest $WORK/Data/mTEDx/fr-en --bpe-model $WORK/Data/mTEDx/fr-en/spm_char.model
    # bash examples/pantagruel/scripts/_modify_paths.sh /lus/work/CT10/c1615074/tphle/Data/prepared/mTEDx/fr-en/${SPLIT}.tsv /gpfsscratch/rech/ahm/umz16dj/Data /lus/work/CT10/c1615074/tphle/Data/raw
done
sort -nk2,2 $WORK/Data/mTEDx/fr-en/${SPLIT}.tsv | head -20 | cut -d ' ' -f3

##### TRAINING
# 2. Run training
# parsing by command-line args
PARTITION=gpu_p5
TASK=finetuning
DATA=$WORK/Data/mTEDx/fr-en
TIME_LIMIT=110
GPUs=8

LABEL="bpe1000"
# LABEL="char"

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/speech/${TASK}
CKPT_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/speech
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_fr
PRETRAIN_CKPT=$CKPT_DIR/pretraining/${PRETRAIN_CONFIG}/checkpoint_best.pt

CONFIG=base_mtedx_w2v_transf_decoder-${LABEL}-${PARTITION}
EXPNAME="${CONFIG}_gpus${GPUs}" # to be changed
TENSORBOARD=${TENSORBOARD_DIR}/${EXPNAME}
SAVE_DIR=${CKPT_DIR}/${TASK}/${EXPNAME}
submit run gpu_p5 $GPUs 2 1 ${EXPNAME} "fairseq-hydra-train task.data=$DATA task.labels=${LABEL} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD} common.user_dir=examples/data2vec checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$PRETRAIN_CKPT --config-dir $CONFIG_DIR --config-name ${CONFIG}.yaml"

# base_mtedx_w2v_transf_decoder-bpe1000-gpu_p5_gpus8
# job 0: 1025720

# base_mtedx_w2v_transf_decoder-char-gpu_p5_gpus8



PARTITION=mi250
TASK=finetuning
DATA=$WORK/Data/prepared/mTEDx/fr-en
TIME_LIMIT=1430
TIME_LIMIT=590
GPUs=16
HOURS=2
JOBS=1

LABEL="bpe1000"
# LABEL="char"

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/adastra/speech/${TASK}

PRETRAIN_CONFIG=base_audio_only_task_ngpu16_fr
CKPT_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/adastra/speech
PRETRAIN_CKPT=$CKPT_DIR/pretraining/${PRETRAIN_CONFIG}/checkpoint_best.pt

# Base architecture
CONFIG=base_mtedx_w2v_transf_decoder-${LABEL}-${PARTITION}
EXPNAME="${CONFIG}_gpus${GPUs}" # to be changed
TENSORBOARD=${TENSORBOARD_DIR}/${EXPNAME}
SAVE_DIR=${CKPT_DIR}/${TASK}/${EXPNAME}

submit run ${PARTITION} ${GPUs} ${HOURS} ${JOBS} ${EXPNAME} "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA} task.labels=${LABEL} common.time_limit=${TIME_LIMIT} common.user_dir=examples/data2vec common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$PRETRAIN_CKPT"

# base_mtedx_w2v_transf_decoder-bpe1000-mi250_gpus16
# job 0: 725609
# job 1: 725610 (after 725609)

# base_mtedx_w2v_transf_decoder-char-mi250_gpus16
