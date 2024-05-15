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
PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared_valid1pct # 1% or 100 examples
# PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared_not_split_valid
DATASETS="mls_french_jz  
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
          voxpopuli_transcribed 
          audiocite_with_metadata"
FAIRSEQ=$HOME/code/fairspeech_torch23


for DATA in $DATASETS; do
    echo "------------- DATASET: ${DATA} -------------"
    if [[ $DATA != *"mls_french_jz"* ]] && [[ $DATA != *"MPF_copy"* ]] && [[ ! -L $RAW_DEST_DIR/$DATA ]]; then
        echo "Creating symlinks to raw data for $DATA"
        ln -s $SRC_DIR/$DATA $RAW_DEST_DIR/$DATA
    fi
    VALID_PERCENT=0.01
    MAX_VALID=100
    if [[ $DATA == "mls_french_jz" ]] || [[ $DATA == "African_Accented_French" ]]; then
        VALID_PERCENT=0.0
        MAX_VALID=0 
    fi
    # VALID_PERCENT=0.0
    # MAX_VALID=0
    bash $HOME/code/slurmx/tools/submit.sh run mi250 1 20 1 prepare_data_$DATA "$FAIRSEQ/examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA --audio-root $RAW_DEST_DIR --output-root $PREPARED_DEST_DIR --valid-percent $VALID_PERCENT --max-valid-samples $MAX_VALID"
    # python $FAIRSEQ/examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA --audio-root $RAW_DEST_DIR --output-root $PREPARED_DEST_DIR --valid-percent $VALID_PERCENT --max-valid-samples $MAX_VALID|& tee $PREPARED_DEST_DIR/logs/$DATA.log
done