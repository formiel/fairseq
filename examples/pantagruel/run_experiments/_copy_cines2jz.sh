#!/bin/bash

MODEL_ROOT=/lus/work/CT10/c1615074/tphle/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining
DST_ROOT=/gpfswork/rech/ahm/umz16dj/experiments/fairseq_checkpoints/pantagruel/speech-text/pretraining
MODEL_NAMES="base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random 
             base_speech_text_en_cnnx3_tokentype_mi250_gpus16  
             base_speech_text_en_cnnx2_tokentype_mi250_gpus16 
             base_speech_text_en_rep_mi250_gpus16 
             base_speech_text_en_rep_token_type_mi250_gpus16" # done
MODEL_NAME=base_speech_text_en_maxtok500k_bsz6_frq1_lr1e-4_wo_lr_cycles_maxupdate1M_mi250_gpus16_dummy_random
mkdir $DST_ROOT/$MODEL_NAME
CKPTPATH=$MODEL_ROOT/$MODEL_NAME/checkpoint_best.pt
rsync -zarvm tphle@adastra-ccfr.cines.fr:$CKPTPATH $DST_ROOT/$MODEL_NAME/