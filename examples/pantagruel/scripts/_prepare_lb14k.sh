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

STEP=$1

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


echo "Step 0: Create data folder with symlinks to raw audio folders"
for DATA in $DATASETS; do
    if [[ $DATA != *"mls_french_jz"* ]] && [[ ! -L $RAW_DEST_DIR/$DATA ]]; then
        echo "Creating symlinks to raw data for $DATA"
        ln -s $SRC_DIR/$DATA $RAW_DEST_DIR/$DATA
        # mkdir -p $RAW_DEST_DIR/$DATA
        # for subset_path in $SRC_DIR/$DATA/*; do
        #     subset_name=$(basename $subset_path)
        #     echo "full path: ${subset_path}, basename: ${subset_name}"
        #     ln -s $subset_path $RAW_DEST_DIR/$DATA/$subset_name
        # done
    fi
done

echo "Step 1: Restructure data into TRAIN/VALID/TEST splits and create manifest files"
for DATA in $DATASETS; do
    python examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA \
        --audio-dir $RAW_DEST_DIR \
        --output-dir $PREPARED_DEST_DIR |& tee $PREPARED_DEST_DIR/logs/prepare_log_$DATA.log
done