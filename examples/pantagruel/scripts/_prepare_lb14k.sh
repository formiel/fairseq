#!/bin/bash
# """
# Author: Phuong-Hang Le (hangtp.le@gmail.com)
# Date: 26 April 2024
# """

###### Prepare and binarize manifest files from raw audio folders obtained from pre-prcessing steps
###### For each corpus, the steps are as follows:
###### 0. Create symlinks to raw audio files
###### 1. Zip audio files to reduce the number of files and I/O reading
###### 2. Prepare manifest files 
###### 3. Binarize manifest files for training
###### The whole training data will be sharded, each shard corresponds to a corpus

# First stage of the project: using only fully open data. The combination of MaSS + EPAC + NCCFr + GEMEP + Niger-Mali amounts to 1784 hours, which is 12.3% of LeBenchmark2 training data, so maybe we can first try excluding them and see how the performance turns out in comparison with LeBenchmark.


SRC_DIR=/lus/work/CT10/lig3801/SHARED/pretraining_data/Modified/LeBenchmark/All_extracted
RAW_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_raw
# PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared_valid1pct # 1% or 100 examples
# PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared_not_split_valid
PREPARED_DEST_DIR=/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared

# mls_french_jz
DATASETS="studios-tamani-kalangou-french 
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
    VALID_PERCENT=0.1
    MAX_VALID=1000
    if [[ $DATA == "mls_french_jz" ]] || [[ $DATA == "African_Accented_French" ]]; then
        VALID_PERCENT=0.0
        MAX_VALID=0 
    fi

    # prepare manifest tsv file
    bash $HOME/code/slurmx/tools/submit.sh run mi250 1 20 1 prepare_data_$DATA "$FAIRSEQ/examples/pantagruel/scripts/data/prepare_audio_manifests.py --dataset $DATA --audio-root $RAW_DEST_DIR --output-root $PREPARED_DEST_DIR --valid-percent $VALID_PERCENT --max-valid-samples $MAX_VALID"

    # binarize data
    bash $FAIRSEQ/examples/pantagruel/scripts/data/binarize_manifest.sh $PREPARED_DEST_DIR $DATA
done

# combine all datasets into 1 file
cp $PREPARED_DEST_DIR/model_training/train_mls_french_jz.tsv $PREPARED_DEST_DIR/data-bin-combined/train_full_16_datasets_except_mass.tsv
cp $PREPARED_DEST_DIR/model_training/valid_mls_french_jz.tsv $PREPARED_DEST_DIR/data-bin-combined/valid_full_16_datasets_except_mass.tsv
SPLITS="train valid"
for DATA in $DATASETS; do
    for SPLIT in $SPLITS; do
        tail -n +2 $PREPARED_DEST_DIR/model_training/${SPLIT}_${DATA}.tsv >> $PREPARED_DEST_DIR/data-bin-combined/${SPLIT}_full_16_datasets_except_mass.tsv
    done
done
wc -l $PREPARED_DEST_DIR/data-bin-combined/train_full_16_datasets_except_mass.tsv
wc -l $PREPARED_DEST_DIR/data-bin-combined/valid_full_16_datasets_except_mass.tsv
bash examples/wav2vec/scripts/binarize_manifest.sh /lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared/data-bin-combined /lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared/data-bin-combined/train_full_16_datasets_except_mass /lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared/data-bin-combined/valid_full_16_datasets_except_mass /lus/home/CT10/c1615074/tphle/code/fairspeech_torch23