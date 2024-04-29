#!/bin/bash

################################################################
#### FINE-TUNING WITH TRANSFORMER DECODER FOR ST TASK
#### CoVoST-2
################################################################

# 0. Download, extract, and convert to wav files
LANGS="en es pt"
for LG in $LANGS; do
    tar -xzvf mtedx_fr-${LG}.tgz
done

bash examples/pantagruel/scripts/_convert_to_wav.sh

DEST_DIR=$HOME/Data/covost2/fr/ssl-encoder
# 1. Prepare manifest files
SPLITS="dev test train"
for SPLIT in $SPLITS; do
    python examples/pantagruel/scripts/_covost2_manifest.py $HOME/Data/covost2/fr --split ${SPLIT} --dest $DEST_DIR
done


# 2. Learn dictionary from file
## for letter dictionary
python $FAIRSEQ/fairseq_cli/preprocess.py \
    --source-lang "ltr" \
    --trainpref "$DEST_DIR/train" \
    --validpref "$DEST_DIR/dev" \
    --destdir "$DEST_DIR" \
    --only-source \
    --dict-only \
    --workers 8;
## for SPM character
python $FAIRSEQ/examples/pantagruel/scripts/_learn_vocab.py --input-path $DEST_DIR/train.wrd \
                                                            --model-type char \
                                                            --model-prefix spm_char
SPLITS="dev test train"
for SPLIT in $SPLITS; do
    python examples/pantagruel/scripts/_covost2_manifest.py $HOME/Data/covost2/fr --split ${SPLIT} --dest $DEST_DIR --bpe-model $DEST_DIR/spm_char.model
done



# 3. Run finetuning for ST task
TASK=finetuning
DATA=$HOME/Data/covost2/fr/ssl-encoder
# LABEL="spm-char"
LABEL="spm1k"
TIME_LIMIT=1190

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/speech/${TASK}
CKPT_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/speech

# Base architecture
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_fr
CONFIG=base_covost_w2v_transf_decoder_${LABEL}
GPUS=8

# Large architecture
# PRETRAIN_CONFIG=large_audio_only_task_ngpu32_freq2_fr_bsz_lr
# CONFIG=large_commonvoice
# GPUS=24

PRETRAIN_CKPT=$CKPT_DIR/pretraining/${PRETRAIN_CONFIG}/checkpoint_best.pt

EXPNAME="${CONFIG}" # to be changed
TENSORBOARD=${TENSORBOARD_DIR}/${EXPNAME}
SAVE_DIR=${CKPT_DIR}/${TASK}/${EXPNAME}
submit run gpu_p5 $GPUS 20 2 $CONFIG "fairseq-hydra-train task.data=$DATA task.labels=${LABEL} common.time_limit=${TIME_LIMIT} common.tensorboard_logdir=${TENSORBOARD} common.user_dir=examples/data2vec checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$PRETRAIN_CKPT --config-dir $CONFIG_DIR --config-name ${CONFIG}.yaml"
# base_covost_w2v_transf_decoder_spm-char: DONE

# base_covost_w2v_transf_decoder_ltr: DONE

##### INFERENCE
fairseq-generate ${COVOST_ROOT} \
    --config-yaml config_st.yaml --gen-subset tst-COMMON_${LANG}_st --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --user-dir
    --max-tokens 50000 --beam 5 --scoring sacrebleu


