#!/bin/bash

INPUT_DIR=/Users/hang/github/formiel/fairspeech/examples/pantagruel/_tb_list
FNAME=$1
INPUT_FILE="${INPUT_DIR}/$FNAME"
TB_ROOT=/Users/hang/tensorboard/pantagruel
DEST_DIR=/Users/hang/tensorboard/_pantagruel/${FNAME}
rm -r $DEST_DIR
mkdir -p $DEST_DIR

while IFS= read -r line; do
    src=$(find $TB_ROOT -type d -name "$line")
    # echo "$line: ${src}: ${#src}"
    if [[ ${#src} -gt 0 ]]; then
        echo "$line: ${src}: ${#src}"
        ln -s $src $DEST_DIR/$line
    fi
done < "$INPUT_FILE"
