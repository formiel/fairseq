
ssh jean-zay-ccfr.idris.fr
MODALITY=speech-text
TASK=finetuning
# PRETRAIN_CONFIG=base_speech_text_en_cnnx3_tokentype_mi250_gpus16
# PRETRAIN_CKPT_NAME=checkpoint_8_735000
# PRETRAIN_CONFIG=base_speech_text_en_cnnx2_tokentype_mi250_gpus16
# PRETRAIN_CKPT_NAME=checkpoint_8_775000
PRETRAIN_CONFIG=base_speech_text_en_rep_mi250_gpus16
PRETRAIN_CKPT_NAME=checkpoint_6_535000
# mkdir -p $WORK/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}

# rsync -zarvm tphle@adastra-ccfr.cines.fr:/lus/work/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/${PRETRAIN_CKPT_NAME}.pt /gpfswork/rech/ahm/umz16dj/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/

python examples/pantagruel/scripts/_change_checkpoints.py --path $WORK/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/${PRETRAIN_CKPT_NAME}.pt

CONFIG=base_10h
GPUS=2
TIME_LIMIT=240
HOURS=4
EXPNAME=${PRETRAIN_CONFIG}_ft_${CONFIG}
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/${PRETRAIN_CKPT_NAME}_updated_path_audio.pt

submit run gpu_p5 $GPUS $HOURS 1 $CONFIG "${FAIRSEQ}/fairseq_cli/hydra_train.py -m task.data=$DATA common.user_dir=examples/data2vec common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}"

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h 
# job 0: 1570240
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep2
# job 0: 1587511
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep4
# job 0: 1597538
# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep5
# job 0: 1629245

# base_speech_text_en_cnnx2_tokentype_mi250_gpus16_ft_base_10h
# job 0: 1710174


task=rte
MODALITY=speech-text
# Jean zay
# data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/MNLI/MNLI-bin
data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE/RTE-bin
PRETRAIN_CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
# PRETRAIN_MODEL=/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/pretraining/base_wikipedia_enwiki_20240201/checkpoint_best.pt
PRETRAIN_MODEL=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${MODALITY}/pretraining/${PRETRAIN_CONFIG}/checkpoint_5_485000_updated_path_text.pt
TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/finetuning/${task}
SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/finetuning/${task}
TIME_LIMIT=110
NUMS="1 2 3 4 5 6 7 8 9"
GPUS=1
HOURS=1
CONFIG=speech-text-finetuning 
for N in $NUMS; do
    rm $SAVE_DIR/*.pt
    torchrun ${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir examples/pantagruel/configs/text/finetuning --config-name $task common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} \
        common.user_dir=examples/data2vec \
        task.data=$data_path checkpoint.save_dir=${SAVE_DIR} \
        model.model_path=${PRETRAIN_MODEL} common.seed=${N} |& tee $SAVE_DIR/finetune-${N}.log
done
