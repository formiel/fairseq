#!/bin/bash

# DATASET=mTEDx
# DATADIR=/lus/work/CT10/c1615074/tphle/Data/prepared/mTEDx/fr-en

# rm $DATADIR/dict.spm-char.txt
# ln -s $DATADIR/spm_char.txt $DATADIR/dict.spm-char.txt

# rm $DATADIR/dict.spm1k.txt
# ln -s $DATADIR/spm_bpe1000.txt $DATADIR/dict.spm1k.txt

# rm $DATADIR/raw
# ln -s /lus/work/CT10/c1615074/tphle/Data/raw/mTEDx/fr-en $DATADIR/raw

SPLITS="train train-debug valid valid-debug test"
for SPLIT in $SPLITS; do
    bash examples/pantagruel/scripts/_modify_paths.sh /lus/work/CT10/c1615074/tphle/Data/prepared/mTEDx/fr-en/${SPLIT}.tsv \
            "/gpfsscratch/rech/ahm/umz16dj/Data/mTEDx/fr-en" \
            "/lus/home/CT10/c1615074/tphle/Data/prepared/mTEDx/fr-en/raw"
done