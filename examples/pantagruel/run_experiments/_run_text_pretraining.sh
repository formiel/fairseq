#!/bin/bash

###############################################################################
###############################################################################
##### JEAN ZAY
###############################################################################
###############################################################################
PARTITION=gpu_p13
TASK=pretraining
MODALITY=text
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=1190

GPUS=16
MASTER_PORT=$(shuf -i 20000-45000 -n 1)
CONFIG=base_text_only_task_4gb

# WIKINAME=enwiki_20240201
WIKINAME=frwiki_20190701
EXPNAME="${CONFIG}_${WIKINAME}_${PARTITION}_gpus${GPUS}"
# DATA_DIR=/gpfswork/rech/ahm/umz16dj/Data/wikitext-103/data-bin/debug
DATA_DIR=$SCRATCH/Data/Wikipedia/${WIKINAME}/data-bin/byteBPE
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}

TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

submit run ${PARTITION} $GPUS 20 2 $EXPNAME "${FAIRSEQ}/fairseq_cli/hydra_train.py --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=$USER_DIR common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT}"

# base_wikipedia_enwiki_20240201: watch 10G
# training, evaluated on RTE: OK
# TODO: re-train with gpt2_bpe tokenizer: no need, checked the length of encoded tokens using different tokenizers already
[2024-03-03 02:30:56,008][fairseq_cli.train][INFO] - training on 16 devices (GPUs/TPUs)
[2024-03-03 02:30:56,009][fairseq_cli.train][INFO] - max tokens per device = None and max sentences per device = 6
[2024-03-03 02:31:03,016][fairseq.data.data_utils][INFO] - loaded 190,504,555 examples from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/enwiki_20240201/data-bin/byteBPE/train
[2024-03-03 02:31:05,895][fairseq.tasks.masked_lm][INFO] - loaded 9226646 blocks from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/enwiki_20240201/data-bin/byteBPE/train
[2024-03-03 02:31:18,848][fairseq.data.iterators][INFO] - grouped total_num_itrs = 96111
# average last 10 ckpts: /gpfswork/rech/ahm/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/pretraining/base_wikipedia_enwiki_20240201/avg_last_10_checkpoint.pt

# experiments/stdlogs/run/base_wikipedia_875663.log: [2024-03-03 02:30:44,975] -> [2024-03-03 21:46:03,186]
# experiments/stdlogs/run/base_wikipedia_875657.log: -> [2024-03-02 07:17:07,154]
# experiments/stdlogs/run/base_wikipedia_875666.log: -> [2024-03-04 21:21:30,669]
# experiments/stdlogs/run/base_wikipedia_861275.log: -> [2024-03-01 03:11:05,521]
# experiments/stdlogs/run/base_wikipedia_875660.log: -> [2024-03-03 02:30:18,319]
# experiments/stdlogs/run/base_wikipedia_875671.log: -> [2024-03-06 01:03:49,264] (done training epoch 11 @ 1000000 updates) (4h for 25k steps)

# base_wikipedia_frwiki_20190701
# 875710: `Minimum loss scale reached (0.0001). Your loss is probably exploding`
# [2024-03-02 06:55:04,956][fairseq.data.data_utils][INFO] - loaded 66,911,314 examples from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/frwiki_20190701/data-bin/byteBPE/train
# [2024-03-02 06:55:05,855][fairseq.tasks.masked_lm][INFO] - loaded 2793151 blocks from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/frwiki_20190701/data-bin/byteBPE/train
# average last 10 ckpts: /gpfswork/rech/ahm/umz16dj/experiments/fairseq_checkpoints/pantagruel/text/pretraining/base_wikipedia_frwiki_20190701/avg_last_10_checkpoint.pt

# base_wikipedia2_frwiki_20190701_gpu_p13_gpus16
# 0: [2024-05-15 15:34:11,574][fairseq.data.data_utils][INFO] - loaded 66,911,314 examples from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/frwiki_20190701/data-bin/byteBPE/train
# 0: [2024-05-15 15:34:12,790][fairseq.tasks.masked_lm][INFO] - loaded 2793151 blocks from: /gpfsscratch/rech/ahm/umz16dj/Data/Wikipedia/frwiki_20190701/data-bin/byteBPE/train
# job 0: 1909621 (20h)
# job 0: 1936341 (done)
# job 1: 1936342 (done)

# base_wikipedia_lr2e-5_wu1k_frwiki_20190701_gpu_p13_gpus16
# job 0: 1936336
# job 1: 1936337 (after 1936336)
# job 2: 1936338 (after 1936337)
# job 3: 1936339 (after 1936338)
# job 4: 1936340 (after 1936339)

# base_text_only_task_4gb_frwiki_20190701_gpu_p13_gpus16
# job 0: 1952893
# job 1: 1952894 (after 1952893)



###############################################################################
###############################################################################
##### Adastra
###############################################################################
###############################################################################
PARTITION=mi250
TASK=pretraining
MODALITY=text
USER_DIR=$FAIRSEQ/examples/data2vec
# TIME_LIMIT=1430
TIME_LIMIT=1190
# TOKENIZER=gpt2_bpe
TOKENIZER=byteBPE
HOURS=20
JOBS=4

GPUs=16
MASTER_PORT=$(shuf -i 20000-40000 -n 1)
# CONFIG=base_wikipedia
CONFIG=base_wikipedia_lr1.2e-3

# WIKINAME=enwiki_20240201
WIKINAME=frwiki_20190701
EXPNAME="${CONFIG}_${WIKINAME}_${TOKENIZER}_${PARTITION}_gpus${GPUs}"
DATA_DIR=$WORK/Data/prepared/Wikipedia/${WIKINAME}/data-bin-v1/${TOKENIZER}
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}

TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/adastra/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/adastra/${MODALITY}/${TASK}/${EXPNAME}

# # working with 8 GPUs of 1 node
# torchrun ${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} \
# --config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT} 

# submitted using 2 nodes
submit run ${PARTITION} ${GPUs} ${HOURS} ${JOBS} ${EXPNAME} "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT}"

# base_wikipedia_bszx10_frwiki_20190701_adastra
# (exact same config as base_wikipedia_bszx10_frwiki_20190701)
# (not look as good as English model)

# base_wikipedia_fr_lr6e-5_frwiki_20190701_mi250_gpus16
# (not look as good as English model)

# base_wikipedia-4g_frwiki_20190701_mi250_gpus256
# job 0: 715073
# job 0: 715671
# job 1: 715672 (after 715671)

# base_wikipedia-4g-freq4_frwiki_20190701_mi250_gpus64
# job 0: 715371
# job 1: 715372 (after 715371)

# base_wikipedia-2_frwiki_20190701_mi250_gpus8
# base_wikipedia-2_frwiki_20190701_mi250_gpus16
# base_wikipedia-2_frwiki_20190701_mi250_gpus32
# base_wikipedia-2_frwiki_20190701_mi250_gpus64

# base_wikipedia-2-lr1e-4_frwiki_20190701_mi250_gpus8
# base_wikipedia-2-lr1e-4_frwiki_20190701_mi250_gpus16
# base_wikipedia-2-lr1e-4_frwiki_20190701_mi250_gpus32

# base_wikipedia-3_frwiki_20190701_mi250_gpus16
# base_wikipedia-3_frwiki_20190701_mi250_gpus32

# base_wikipedia_enwiki_20240201_gpt2_bpe_mi250_gpus16 (look better than byteBPE)
# job 0: 720601 (failed: c10::DistBackendError)
# job 1: 720602 (after 720601)
# job 0: 721070 (PENDING)
# job 1: 721071 (after 721070)

# base_wikipedia_enwiki_20240201_byteBPE_mi250_gpus16
# job 0: 720606 (looking good, similar to the one trained on JZ)
# job 1: 720607 (after 720606) (pending)

# base_wikipedia_frwiki_20190701_gpt2_bpe_mi250_gpus16
# job 0: 720608 (looking good)
# job 1: 720609 (after 720608)
# /lus/work/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/adastra/text/pretraining/base_wikipedia_frwiki_20190701_gpt2_bpe_mi250_gpus16

# base_wikipedia_frwiki_20190701_byteBPE_mi250_gpus16
# job 0: 723568
# job 1: 723569 (after 723568)
# job 2: 723570 (after 723569)
# job 3: 723571 (after 723570)
# /lus/home/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/adastra/text/pretraining/base_wikipedia_frwiki_20190701_byteBPE_mi250_gpus16/

# base_wikipedia_frwiki_20190701_gpt2_bpe_mi250_gpus64
# /lus/home/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/adastra/text/pretraining/base_wikipedia_frwiki_20190701_gpt2_bpe_mi250_gpus64

# base_wikipedia_frwiki_20190701_byteBPE_mi250_gpus64
# /lus/home/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/adastra/text/pretraining/base_wikipedia_frwiki_20190701_byteBPE_mi250_gpus64

# base_wikipedia_lr1.2e-3_frwiki_20190701_gpt2_bpe_mi250_gpus16
# base_wikipedia_lr1.2e-3_frwiki_20190701_gpt2_bpe_mi250_gpus64
# base_wikipedia_lr1.2e-3_frwiki_20190701_byteBPE_mi250_gpus16
# base_wikipedia_lr1.2e-3_frwiki_20190701_byteBPE_mi250_gpus64