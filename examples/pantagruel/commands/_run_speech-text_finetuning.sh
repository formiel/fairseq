
MODALITY=speech-text
TASK=finetuning
PRETRAIN_CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
CONFIG=base_10h
GPUS=2
TIME_LIMIT=350
HOURS=5
EXPNAME=${PRETRAIN_CONFIG}_ft_${CONFIG}
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}_ep4
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}_ep4
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining/${PRETRAIN_CONFIG}/checkpoint_4_315000_updated_path.pt

submit run gpu_p5 $GPUS $HOURS 1 $CONFIG "${FAIRSEQ}/fairseq_cli/hydra_train.py -m task.data=$DATA common.user_dir=examples/data2vec common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} checkpoint.save_dir=${SAVE_DIR} model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}"

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h 
# job 0: 1570240

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep2
# job 0: 1587511

# base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random_ft_base_10h_ep4
# job 0: 1597538


task=rte
MODALITY=speech-text
# Jean zay
# data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/MNLI/MNLI-bin
data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE/RTE-bin
PRETRAIN_CONFIG=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
# PRETRAIN_MODEL=/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/pretraining/base_wikipedia_enwiki_20240201/checkpoint_best.pt
PRETRAIN_MODEL=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${MODALITY}/pretraining/${PRETRAIN_CONFIG}/checkpoint_4_315000_updated_path_text.pt
TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/finetuning/${task}
SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/finetuning/${task}
TIME_LIMIT=110
NUMS="2 3 4 5 6 7 8 9"
for N in $NUMS; do
    rm $SAVE_DIR/*.pt
    python fairseq_cli/hydra_train.py -m --config-dir examples/pantagruel/configs/text/finetuning \
        --config-name $task common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} \
        common.user_dir=examples/data2vec \
        task.data=$data_path checkpoint.save_dir=${SAVE_DIR} \
        model.model_path="${PRETRAIN_MODEL}" |& tee ${SAVE_DIR}/finetune-${N}.log
done

