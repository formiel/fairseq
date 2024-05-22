
### Reproduce results on GLUE
# ref: https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md

# 1. Download data
# Code not working, links not available
# https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md
# wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
# python download_glue_data.py --data_dir glue_data --tasks 'XNLI'

wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
unzip MNLI.zip

# 2. Preprocess glue task data
# ./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
# MNLI: seems to be compliated
bash $FAIRSEQ/examples/roberta/preprocess_GLUE_tasks.sh $WORK/Data/glue_data RTE

# Evaluate English pre-trained model on GLUE tasks
task=rte
# Jean zay
# data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/MNLI/MNLI-bin
data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE/RTE-bin
PRETRAIN_MODEL=/linkhome/rech/genlig01/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/pretraining/base_wikipedia_enwiki_20240201/checkpoint_best.pt
TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/text/finetuning/${task}
SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/text/finetuning/${task}
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


task=rte
# Jean zay
# data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/MNLI/MNLI-bin
data_path=/gpfswork/rech/ahm/umz16dj/Data/glue_data/RTE-fairseq-pretrained/RTE-bin
PRETRAIN_MODEL=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/text/nlp_base_changed_path.pt
TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/text/finetuning/${task}_pretrained_from_fairseq
SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/text/finetuning/${task}_pretrained_from_fairseq
mkdir -p $SAVE_DIR
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


# French models
# Evaluate French pre-trained model on PAWSX tasks
task=pawsx
# Jean zay
data_path=$WORK/Data/flue_data/pawsx/x-final/fr/d2v-frwiki19-bin/byteBPE
# PRETRAINED_NAME=base_wikipedia_frwiki_20190701
# PRETRAINED_NAME=base_wikipedia2_frwiki_20190701_gpu_p13_gpus16
# PRETRAINED_NAME=base_text_only_task_4gb_frwiki_20190701_gpu_p13_gpus16
PRETRAINED_NAMES="base_wikipedia_frwiki_20190701 
                  base_wikipedia2_frwiki_20190701_gpu_p13_gpus16 
                  base_text_only_task_4gb_frwiki_20190701_gpu_p13_gpus16"
PRETRAINED_NAMES="base_text_only_task_4gb_frwiki_20190701_gpu_p13_gpus16"
TIME_LIMIT=110
for PRETRAINED_NAME in $PRETRAINED_NAMES; do
    PRETRAIN_MODEL=$HOME/experiments/fairseq_checkpoints/pantagruel/text/pretraining/${PRETRAINED_NAME}/checkpoint_best.pt
    NUMS="1 2 3 4 5 6 7 8 9"
    for N in $NUMS; do
        TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/text/finetuning/${PRETRAINED_NAME}_${task}_${N}
        SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/text/finetuning/${PRETRAINED_NAME}_${task}_${N}
        submit run gpu_p13 1 3 1 finetune_${PRETRAINED_NAME}_${task}_${N} "${FAIRSEQ}/fairseq_cli/hydra_train.py --config-dir examples/pantagruel/configs/text/finetuning --config-name $task common.seed=${N} common.tensorboard_logdir=${TENSORBOARD_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=examples/data2vec task.data=$data_path checkpoint.save_dir=${SAVE_DIR} model.model_path=${PRETRAIN_MODEL}"
    done
done

task=pawsx-camembert
# Jean zay
PRETRAINED_NAME=camembert-base-wikipedia-4gb
data_path=$WORK/Data/flue_data/pawsx/x-final/fr/${PRETRAINED_NAME}-bin
ROBERTA_PATH=$WORK/pretrained_models/roberta/camembert/${PRETRAINED_NAME}/model.pt
TIME_LIMIT=110
NUMS="1 2 3 4 5 6 7 8 9"
for N in $NUMS; do
    TENSORBOARD_DIR=$HOME/experiments/fairseq_tensorboard/pantagruel/text/finetuning/${PRETRAINED_NAME}_${task}_${N}
    SAVE_DIR=$HOME/experiments/fairseq_checkpoints/pantagruel/text/finetuning/${PRETRAINED_NAME}_${task}_${N}
    submit run gpu_p13 1 3 1 finetune_${PRETRAINED_NAME}_${N} "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir examples/pantagruel/configs/text/finetuning --config-name $task common.tensorboard_logdir=${TENSORBOARD_DIR} common.seed=${N} checkpoint.save_dir=${SAVE_DIR} task.data=$data_path checkpoint.restore_file=$ROBERTA_PATH"
done