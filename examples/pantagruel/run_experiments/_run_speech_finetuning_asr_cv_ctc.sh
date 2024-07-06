#!/bin/bash

################################################################
##### SPEECH
##### Task: Fine-tuning for ASR
##### Arch: Using CTC module
##### Dataset: on CommonVoice/fr datasets
################################################################

##### Keep original text
# 1. Prepare manifest files
SPLITS="dev test train"
DATADIR=$DSDIR/CommonVoice/cv-corpus-6.1-2020-12-11/fr
DEST_DIR=${WORK}/Data/CommonVoice/fr
for SPLIT in $SPLITS; do
    echo "Preparing ${SPLIT}"
    python $FAIRSEQ/examples/pantagruel/scripts/_common_voice_manifest.py $DATADIR \
                --split ${SPLIT} --dest $DEST_DIR
    # ln -s $DEST_DIR/${SPLIT}.tsv $DEST_DIR/${SPLIT}.bpe500.tsv
done

# 2. Learn dictionary from file
python $FAIRSEQ/fairseq_cli/preprocess.py \
    --source-lang "fr.ltr" \
    --trainpref "$DEST_DIR/train" \
    --validpref "$DEST_DIR/dev" \
    --destdir "$DEST_DIR" \
    --only-source \
    --dict-only \
    --workers 8

# 2. Prepare label file: BPE500, then encode, and prepare label file

# 3. Run finetuning for ASR task
TASK=finetuning
MODALITY=speech
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=1190

MASTER_PORT=$(shuf -i 20000-65000 -n 1)
GPUS=16

DATA_DIR=$WORK/Data/CommonVoice/fr
LABEL="fr.ltr"

# Pre-trained checkpoint
# PRETRAIN_CONFIG=base_audio_only_task_ngpu16_fr
PRETRAIN_CONFIG=large_audio_only_lb14k_v5_maxupdate100000_wu5000_mi250_gpus128
# CKPT=checkpoint_best
CKPT=avg_last_10_checkpoint
PRETRAIN_CKPT=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/pretraining/${PRETRAIN_CONFIG}/${CKPT}.pt
# PRETRAIN_CONFIG=large_audio_only_task_ngpu48_fr
# PRETRAIN_CKPT=$WORK/experiments/fairseq_checkpoints/pantagruel/${PRETRAIN_CONFIG}/checkpoint_best_200k.pt
# PRETRAIN_CONFIG=large_audio_only_lb14k_no_bin_maxtok640k_maxupdate600000_mi250_gpus48
# PRETRAIN_CKPT=$WORK/experiments/fairseq_checkpoints/pantagruel/speech/pretraining/${PRETRAIN_CONFIG}/checkpoint_best.pt

# CONFIG=base_commonvoice_hparams_bszx4
# CONFIG=base_commonvoice
CONFIG=large_commonvoice
EXPNAME="${CONFIG}_ngpu${GPUS}_${LABEL}_pt_${PRETRAIN_CONFIG}_${CKPT}" # !!!UNIQUE for each experiment!!!

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/speech/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

submit run gpu_p2 $GPUS 20 3 $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} task.labels=${LABEL} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=${PRETRAIN_CKPT} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT}"

# base_commonvoice_fr.ltr: WER on dev 8.96
#
# base_commonvoice_full_fr.ltr: WER on dev 8.92
# (to compare if data loader is ok when doing every 2h)

# base_commonvoice_ngpu16_fr.ltr_pt_base_audio_only_task_ngpu16_fr_avg_last_10_checkpoint
# job 0: 2656
# job 1: 2657 (after 2656)

# base_commonvoice_hparams_fr.ltr 
# (mask_channel_prob=0.25 instead of 0.1)
# (looking better than base_commonvoice_full_fr.ltr)
#  (cancelling as not as good as using larger bsz)
#
# base_commonvoice_hparams_bsz_fr.ltr: best WER on dev 8.92
# (increasing batch size from 3.2M to 4.8M, look better)
# (much better compared to base_commonvoice_hparams_fr.ltr => cancel the latter)

# base_commonvoice_hparams_bszx4_fr.ltr
# (look better than base_commonvoice_hparams_bsz_fr.ltr)
# (loss much more overfitting, though WER looks better for the first 150k steps)

# large_commonvoice_ngpu16_fr.ltr
# job 0: 947844 (completed)
# job 1: 947845 (after 947844)

# large_commonvoice_ngpu16_fr.ltr_pt_large_audio_only_lb14k_no_bin_maxtok640k_maxupdate600000_mi250_gpus48 (best WER on dev 7.84%)
# job 0: 2084561
# job 1: 2084562

# large_commonvoice_ngpu16_fr.ltr_pt_large_audio_only_lb14k_v5_maxupdate100000_wu5000_mi250_gpus128_checkpoint_best
# job 0: 398069
# job 1: 398070 (after 398069)
# job 2: 398071 (after 398070)

# large_commonvoice_ngpu16_fr.ltr_pt_large_audio_only_lb14k_v5_maxupdate100000_wu5000_mi250_gpus128_avg_last_10_checkpoint
# job 0: 398076
# job 1: 398077 (after 398076)
# job 2: 398078 (after 398077)

# 4.decoding data2vec2.0 model trained with CTC
SPLIT=test
CONFIG=fr_finetune_commonvoice
DATA=$WORK/Data/CommonVoice/fr
CHECKPOINT=$SCRATCH/Experiments/fairseq_checkpoints/pantagruel/${CONFIG}/checkpoint_best.pt
# CHECKPOINT=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/speech/${CONFIG}.pt
python examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
--config-name infer task=audio_finetuning task.data=${DATA} common.user_dir=examples/data2vec \
common_eval.results_path=$SCRATCH/Experiments/fairseq_checkpoints/${CONFIG} \
common_eval.quiet=false
task.labels=ltr \
dataset.gen_subset=$SPLIT \
common_eval.path=${CHECKPOINT} decoding.beam=5
