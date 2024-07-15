CONFIG=base_10h
GPUS=2
TIME_LIMIT=590
HOURS=10

PARTITION=gpu_p2
FAIRSEQ=$HOME/code/fairspeech
USER_DIR=$FAIRSEQ/examples/pantagruel

MODALITY=speech-text
TASK=finetuning

# PRETRAIN_CONFIG=base_speech_gpu_p2_gpus16
PRETRAIN_CONFIG=base_speech_dummy_text_factor0.01_gpu_p2_gpus16

EXPNAME=${CONFIG}_pt_${PRETRAIN_CONFIG}

DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled

TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

CHECKPOINT=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/pretraining/${PRETRAIN_CONFIG}/checkpoint_last.pt

submit run ${PARTITION} $GPUS $HOURS 1 $CONFIG "${FAIRSEQ}/fairseq_cli/hydra_train.py -m task.data=$DATA common.user_dir=$USER_DIR common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/${MODALITY}/finetuning --config-name ${CONFIG}"

# base_10h_pt_base_speech_gpu_p2_gpus16
job 0: 683602

# base_10h_pt_base_speech_dummy_text_factor0.0_gpu_p2_gpus16
job 0: 683727
