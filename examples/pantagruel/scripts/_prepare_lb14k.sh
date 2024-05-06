#!/bin/bash
# """
# Author: Phuong-Hang Le (hangtp.le@gmail.com)
# Date: 26 April 2024
# """

###### Prepare and binarize manifest files from raw audio folders obtained from pre-prcessing steps
###### For each corpus, the steps are as follows:
###### 0. Create symlibks to raw audio files
###### 1. Zip audio files to reduce the number of files and I/O reading
###### 2. Prepare manifest files 
###### 3. Binarize manifest files for training
###### The whole training data will be sharded, each shard corresponds to a corpus

# First stage of the project: using only fully open data. The combination of MaSS + EPAC + NCCFr + GEMEP + Niger-Mali amounts to 1784 hours, which is 12.3% of LeBenchmark2 training data, so maybe we can first try excluding them and see how the performance turns out in comparison with LeBenchmark.


SRC_DIR=/lus/work/CT10/lig3801/SHARED/pretraining_data/Modified/LeBenchmark/All_extracted
RAW_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_raw
PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared
# mls_french_jz: train/dev/test
DATASETS="audiocite_with_metadata 
          studios-tamani-kalangou-french 
          African_Accented_French 
          Att-HACK_SLR88 
          CaFE 
          CFPP_corrected 
          ESLO 
          EPAC_flowbert 
          GEMEP 
          MPF 
          Portmedia 
          TCOF_corrected 
          MaSS 
          NCCFr 
          voxpopuli_unlabeled 
          voxpopuli_transcribed"
DATASETS="MPF 
          TCOF_corrected"
FAIRSEQ=$HOME/code/fairspeech_torch23


for DATA in $DATASETS; do
    echo "------------- DATASET: ${DATA} -------------"
    if [[ $DATA != *"mls_french_jz"* ]] && [[ ! -L $RAW_DEST_DIR/$DATA ]]; then
        echo "Creating symlinks to raw data for $DATA"
        ln -s $SRC_DIR/$DATA $RAW_DEST_DIR/$DATA
    fi
    bash $HOME/code/slurmx/tools/submit.sh run mi250 1 10 1 prepare_data_$DATA "$FAIRSEQ/examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA --audio-root $RAW_DEST_DIR --output-root $PREPARED_DEST_DIR --workers 1"
    # python $FAIRSEQ/examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA --audio-root $RAW_DEST_DIR --output-root $PREPARED_DEST_DIR
done
# 857817: voxpopuli_transcribed: OK (77387)
# 861357: mls_french_jz
# 861233 African_Accented_French OK (14483 + 1747 + 515  = 16745)
# 861234 Att-HACK_SLR88 OK (36634)
# 861235 CaFE OK (936)
# 861236 CFPP_corrected
# 861405 ESLO (62918)
# 861238 GEMEP (1260)
# 861239 MPF
# 861408 Portmedia (20400) 
# 861241 TCOF_corrected
# 861410 MaSS OK (8219)
# 861411 NCCFr (29421)
# 861403 studios-tamani-kalangou-french OK (38332)
# 861404 CFPP_corrected (12881)
# 861406 EPAC_flowbert (R)
# 861734 MPF error (zipping)
# 861735 TCOF_corrected error
# 861412 voxpopuli_unlabeled