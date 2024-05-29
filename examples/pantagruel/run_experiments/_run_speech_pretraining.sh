#!/bin/bash

###############################################################################
###############################################################################
##### SPEECH
##### Task: pre-training 
##### Dataset: French MLS
##### Server: Jean Zay
###############################################################################
###############################################################################

# 1. Prepare manifest file including list of audio files to be input to model training 
DATA_DIR=/gpfsscratch/rech/ahm/umz16dj/Data/mls_french_jz
DST_DIR=/gpfswork/rech/ahm/umz16dj/Data/MLS_French
SPLITS="train dev test"
for SPLT in $SPLITS; do
    echo "Processing ${SPLT}"
    python examples/wav2vec/wav2vec_manifest.py $DATA_DIR/${SPLT}/audio --valid-percent 0 --dest ${DST_DIR} --ext flac
    mv $DST_DIR/train.tsv $DST_DIR/${SPLT}.tsv
    wc -l $DST_DIR/${SPLT}.tsv
done

# 2. Run PRE-TRAINING
TASK=pretraining
MODALITY=speech
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=110

MASTER_PORT=$(shuf -i 20000-65000 -n 1)
# CONFIG=base_mls1k
CONFIG=base_audio_only_task_ngpu16_fr_maxtok2.8M_lr1.5e-3
GPUS=16

# CONFIG=large_mls1k
# GPUS=32

EXPNAME="${CONFIG}" # ===== CHECK THIS =====
DATA_DIR=$WORK/Data/MLS_French
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

submit run gpu_p5 $GPUS 2 1 $EXPNAME "fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT}"
# # base_audio_only_task_ngpu16_fr -> Speech_Base_fr_1K
# COMPLETED 
# bsz: 1M x 16gpus x 1freq, lr=7.5e-4
# (16h on 16gpus with eff bsz: max_tok 1M x 16gpus x frq1) (25k: 14m)
# (good, evaluated downstream performance for ASR task, better than LeBenchmark large)
# (loss still decreases, could further improve by tuning hyper-params)
# (Will tune later, not now!)
# job 676472: [2024-02-16 17:33:16,173] - [2024-02-17 05:09:50,470]: 41795s ~ 697min
# job 676473: [2024-02-17 05:13:12,214] - [2024-02-17 16:53:14,154]: 42003s ~ 700min
# job 676474: [2024-02-18 09:43:35,313] - [2024-02-18 17:40:25,294]: 28610s ~ 477min
# total: 697+700+477 = 1874m (31 hours)

# # base_audio_only_task_ngpu8_fr_frq8_bsz (28min for 760updates)
# (watching 38G)
# (looking good compared to the original hyper-params)
# (not finised, but took so long to run)
#
# # base_audio_only_task_ngpu8_fr_frq8_bsz_lr7.5e-3
# (stopped because of low pred var)
#
# # base_audio_only_task_ngpu8_fr_frq8_bsz_lr3.75e-3
# (gradient overflow)
#
# # base_audio_only_task_ngpu16_frq2_bsz89.6M


# # large_audio_only_task_ngpu48_fr
# (looking good compared to base arch's loss)
# (24m for 5365 updates)=> 44h for 600k updates
# (403m for 25k steps from 175k-200k) => 160 for 600k updates
# path: experiments/fairseq_checkpoints/pantagruel/large_audio_only_task_ngpu48_fr
USER_DIR=$FAIRSEQ/examples/data2vec
CONFIG=large_audio_only_task_ngpu48_fr
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/pretraining
GPUS=48
DATA_DIR=$WORK/Data/MLS_French
submit run gpu_p5 $GPUS 20 5 $CONFIG "fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} common.user_dir=${USER_DIR}"
# 806301: [2024-02-26 15:27:50,992] - [2024-02-27 01:15:16,226]
# 806302: [2024-02-28 00:28:07,471] - [2024-02-28 10:15:54,650]
# 806303: [2024-02-29 20:54:38,946] - [2024-03-01 06:44:33,462]
# 806304: [2024-03-03 03:02:40,716] - [2024-03-03 12:53:04,346]
# 901780: [2024-03-05 16:35:03,886] - [2024-03-06 11:09:54,880]
# 901781: [2024-03-07 20:21:17,864] - [2024-03-08 16:06:30,243]
# 901782 (after 901781)
# job 4: 901783 (after 901782)

# large_audio_only_task_ngpu32_freq2_fr_bsz_lr 
# (same as LB-1K-large, but pred_var too low and training stopped)
# (run and stopped)

# large_audio_only_task_ngpu32_freq2_fr_bsz_lr2.5e-3
# (error)

###############################################################################
###############################################################################
##### SPEECH
##### Task: pre-training 
##### Dataset: French MLS
##### Server: Adastra
###############################################################################
###############################################################################

# 1. After copying preprocessed data from JZ to Adastra
# The raw data in $SCRATCH (JZ) is put under $WORK/Data/raw (Ada)
# while the preprocessed data in $WORK (JZ) is put under $WORK/Data/preprocessed (Ada)
# we need to modify the root path in tsv manifest files

SPLITS="train train-debug dev dev-debug test"
for SPLIT in $SPLITS; do
    bash $FAIRSEQ/examples/pantagruel/scripts/_modify_paths.sh /lus/work/CT10/c1615074/tphle/Data/prepared/MLS_French/${SPLIT}.tsv
done



# 2. Run training
TASK=pretraining
MODALITY=speech
# FAIRSEQ=$HOME/code/fairspeech_torch23
FAIRSEQ=$HOME/code/fairspeech
USER_DIR=$FAIRSEQ/examples/data2vec
TIME_LIMIT=1430
PARTITION=mi250

# submitted using 2 nodes
GPUs=48
# GPUs=64
HOURS=24
JOBS=5
UPDATE=300000

MASTER_PORT=$(shuf -i 20000-40000 -n 1)
# CONFIG=base_mls1k
# CONFIG=base_audio_only_task
# CONFIG=base_audio_only_task_ngpu16_fr_adastra
# CONFIG=base_audio_only_task_ngpu16_fr_adastra_maxtok1.4M_lr1e-3
# CONFIG=large_audio_only_task_ngpu48_fr_adastra
# CONFIG=large_audio_only_task_ngpu64_fr_bsz89.6M_adastra

CONFIG=large_audio_only_lb14k
CONFIG=large_audio_only_lb14k_no_bin
CONFIG=large_audio_only_lb14k_no_bin_maxtok320k
CONFIG=large_audio_only_lb14k_no_bin_maxtok640k
CONFIG=large_audio_only_lb14k_no_bin_maxtok640k_wu5k

SUFFIX= # ===== CHECK THIS =====
EXPNAME="${CONFIG}_maxupdate${UPDATE}_${PARTITION}_gpus${GPUs}${SUFFIX}"
# DATA_DIR=$WORK/Data/prepared/MLS_French
# DATA_DIR=$SCRATCH/Data/LeBenchmark_prepared/data-bin
DATA_DIR=$SCRATCH/Data/LeBenchmark_prepared/model_training

CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

# # working with 8 GPUs of 1 node
# torchrun ${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} \
# --config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT} 

submit run ${PARTITION} ${GPUs} ${HOURS} ${JOBS} ${EXPNAME} "${FAIRSEQ}/fairseq_cli/hydra_train.py -m --config-dir ${CONFIG_DIR} \
--config-name $CONFIG.yaml task.data=${DATA_DIR} common.time_limit=${TIME_LIMIT} common.user_dir=${USER_DIR} common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUs} distributed_training.distributed_port=${MASTER_PORT} optimization.max_update=${UPDATE}"

# base_audio_only_task_ngpu16_fr_adastra => Speech_Base_fr_1K_mi250x 
# (exact same training settings as "base_audio_only_task_ngpu16_fr" trained on JZ)
# 25k: 26m 
# job 0: 704391 [2024-03-06 07:47:50,532] - [2024-03-07 07:12:42,795]: ~24h
# job 1: 704392 [2024-03-07 07:58:31,000] - [2024-03-08 07:49:39,708]: ~24h
# job 0: 845731 [2024-04-29 15:47:45,652] - [2024-04-30 01:07:44,119]: ~12h
# job 1: 845732 (done training) [2024-04-30 01:23:43,862] - [2024-04-30 08:17:51,056]: ~7h (torch220 worked after installing torch23)
# Total: 67 hours

# base_audio_only_task_ngpu16_fr_adastra_torch23
# job 0: 852754 (not install apex yet) (same speed as with torch22 (no flash attention))
# job 1: 852755 (after 852754) (OSError: [Errno 12] Cannot allocate memory)
# job 0: 875985

# base_audio_only_task_ngpu16_fr_adastra_maxtok1.4M_lr1e-3
# job 0: 704393 [2024-03-06 08:08:57,336] - [2024-03-07 07:51:48,894]: 112977updates

# large_audio_only_task_ngpu48_fr_adastra (failed)

# large_audio_only_task_ngpu64_fr_bsz89.6M_adastra (failed)

# large_audio_only_lb14k_fr (min_sample_size: 3k)
# loaded 2,639,922 examples from: /lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared/data-bin/train
# grouped total_num_itrs = 38180
# job 0: 886599 (done till 16k steps)
# job 1: 886600 (error) g[1130,1132-1136]'
# job 2: 886601 (after 886600) 'g[1130,1132-1136]'
# job 3: 886602 (after 886601) 'g[1130,1132-1136]'
# job 4: 886603 (after 886602) 'g[1130,1132-1136]'
# job 0: 888268
# job 1: 888269 (after 888268)


# large_audio_only_lb14k_no_bin_fr (min_sample_size: 32k as binarized dataset does not support min_sample_size)
# loaded 2498162, skipped 141760 samples, grouped total_num_itrs = 38051
# job 0: 888291 (24h)
# job 1: 888292 (after 888291) (6.5h)
# job 0: 889949
# job 1: 889950 (after 889949)
# job 2: 889951 (after 889950) (18h)

# large_audio_only_lb14k_no_bin_maxtok320k_fr
# loaded 2498162, skipped 141760 samples, grouped total_num_itrs = 44530
# job 0: 894643
# job 1: 894644 (after 894643) (Running, save in 3 more hours)


# large_audio_only_lb14k_no_bin_maxtok320k_mi250_gpus64
# loaded 2498162, skipped 141760 samples, grouped total_num_itrs = 33397
# job 0: 894861
# job 1: 894862 (after 894861) (Running, save in 6 more hours)


# large_audio_only_lb14k_no_bin_maxtok640k_maxupdate600000_mi250_gpus48
# loaded 2498162, skipped 141760 samples, grouped total_num_itrs = 21073
# job 0: 895428 (16h)
# job 1: 895429 (after 895428) (Running, save in 17 more hours)
# job 0: 900391 (too much logging)
# job 1: 900392 (after 900391) (too much logging)
# job 2: 900393 (after 900392) (cancelled, ran so slow)
# job 3: 900394 (after 900393)
# job 4: 900396 (after 900394) (cancelled as loss increasing)

# large_audio_only_lb14k_no_bin_maxtok640k_wu5k_maxupdate300000_mi250_gpus48
# job 0: 913123
# job 1: 913124 (after 913123)
# job 2: 913125 (after 913124)
# job 3: 913126 (after 913125)
# job 4: 913128 (after 913126)
