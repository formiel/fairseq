
MODALITY=speech-text
TASK=finetuning
PRETRAIN_CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
CONFIG=base_10h
GPUS=2
TIME_LIMIT=350
HOURS=6
EXPNAME=${PRETRAIN_CONFIG}_ft_${CONFIG}
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}_ep2
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}_ep2
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/checkpoint_2_200000_updated_path.pt

submit run gpu_p5 $GPUS $HOURS 1 $CONFIG "${FAIRSEQ}/fairseq_cli/hydra_train.py -m task.data=$DATA common.user_dir=examples/data2vec common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}"

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h 
# job 0: 1570240

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep2
# job 0: 1587511

