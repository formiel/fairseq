#!/usr/bin/env bash

# usage: bash binarize_manifest <DATA_ROOT> <DATASET>

DATA_ROOT=$1
DATASET=$2
TRAIN_SPLIT=train_${DATASET}
VALID_SPLIT=valid_${DATASET}
FAIRSEQ_ROOT=$HOME/code/fairspeech_torch23

TSV_DIR=$DATA_ROOT/model_training
DATA_BIN=$DATA_ROOT/data-bin

mkdir -p $DATA_BIN


# split file path and lengths into separate files
cut -f1 $TSV_DIR/$TRAIN_SPLIT.tsv > $DATA_BIN/${TRAIN_SPLIT}_fnames.txt
cut -f1 $TSV_DIR/$VALID_SPLIT.tsv > $DATA_BIN/${VALID_SPLIT}_fnames.txt
cut -f2 $TSV_DIR/$TRAIN_SPLIT.tsv > $DATA_BIN/${TRAIN_SPLIT}.lengths
cut -f2 $TSV_DIR/$VALID_SPLIT.tsv > $DATA_BIN/${VALID_SPLIT}.lengths

# copy root directory
head -1 $TSV_DIR/$TRAIN_SPLIT.tsv > $DATA_BIN/${TRAIN_SPLIT}.root
head -1 $TSV_DIR/$VALID_SPLIT.tsv > $DATA_BIN/${VALID_SPLIT}.root

# remove root directory
sed -i '1d' $DATA_BIN/${TRAIN_SPLIT}_fnames.txt
sed -i '1d' $DATA_BIN/${VALID_SPLIT}_fnames.txt
sed -i '1d' $DATA_BIN/${TRAIN_SPLIT}.lengths
sed -i '1d' $DATA_BIN/${VALID_SPLIT}.lengths

# insert spaces between characters
sed -i -e 's/\(.\)/\1 /g' $DATA_BIN/${TRAIN_SPLIT}_fnames.txt
sed -i -e 's/\(.\)/\1 /g' $DATA_BIN/${VALID_SPLIT}_fnames.txt

# run preprocessor
PYTHONPATH=$FAIRSEQ_ROOT python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $DATA_BIN/${TRAIN_SPLIT}_fnames.txt --validpref $DATA_BIN/${VALID_SPLIT}_fnames.txt --workers 8 --only-source --destdir $DATA_BIN

# rename
mv $DATA_BIN/preprocess.log $DATA_BIN/preprocess_${DATASET}.log
mv $DATA_BIN/dict.txt $DATA_BIN/dict_${DATASET}.txt
SPLITS="train valid"
EXTS="bin idx"
for SPLIT in $SPLITS; do
    for EXT in $EXTS; do
        mv $DATA_BIN/${SPLIT}.${EXT} $DATA_BIN/${SPLIT}_${DATASET}.${EXT}
    done
done
