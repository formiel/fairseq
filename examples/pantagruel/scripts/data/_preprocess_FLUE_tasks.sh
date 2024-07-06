#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


GLUE_DATA_FOLDER=$1
num_workers=$2
MODEL=d2v-frwiki19

# download bpe encoder.json, vocabulary and fairseq dictionary

TASKS=PAWSX
TASK_DATA_FOLDER=$GLUE_DATA_FOLDER/pawsx/x-final/fr
TASK_DATA_BIN=$TASK_DATA_FOLDER/${MODEL}-bin/byteBPE
mkdir -p $TASK_DATA_BIN
DICT_ROOT=/gpfswork/rech/ahm/umz16dj/Data/Wikipedia/frwiki_20190701/data-bin-v1/byteBPE
FILES="vocab.bpe encoder.json dict.txt"
for FILE in $FILES; do
  ln -s $DICT_ROOT/$FILE $TASK_DATA_BIN/$FILE
done

INPUT_COLUMNS=( 2 3 )
TEST_INPUT_COLUMNS=( 2 3 )
LABEL_COLUMN=4
INPUT_COUNT=2
SPLITS="translated_train dev_2k test_2k"

# Strip out header and filter lines that don't have expected number of fields.
rm -rf "$TASK_DATA_FOLDER/processed"
mkdir -p "$TASK_DATA_FOLDER/processed"

for SPLIT in $SPLITS; do
  # remove header
  tail -n +2 "$TASK_DATA_FOLDER/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
  cp "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv";
  rm "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";

  # get inputs into each file
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1))); do
    COLUMN_NUMBER=${TEST_INPUT_COLUMNS[$INPUT_TYPE]}
    cut -f"$COLUMN_NUMBER" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.raw.input$INPUT_TYPE";
  done
  # label
  cut -f"$LABEL_COLUMN" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.label";

  # BPE encode.
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1))); do
    LANG="input$INPUT_TYPE"
    echo "BPE encoding $SPLIT/$LANG"
    NEW_SPLIT=train
    if [[ $SPLIT = *"dev"* ]]; then
      NEW_SPLIT=dev
    elif [[ $SPLIT = *"test"* ]]; then
      NEW_SPLIT=test
    fi
    python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json $TASK_DATA_BIN/encoder.json \
      --vocab-bpe $TASK_DATA_BIN/vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LANG" \
      --outputs "$TASK_DATA_FOLDER/processed/${NEW_SPLIT}.$LANG" \
      --workers ${num_workers} \
      --keep-empty;
    done
done

# Remove output directory.
# rm -rf "$TASK_DATA_BIN"

DEVPREF="$TASK_DATA_FOLDER/processed/dev.LANG"
TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"
cp $TASK_DATA_FOLDER/processed/translated_train.label $TASK_DATA_FOLDER/processed/train.label
cp $TASK_DATA_FOLDER/processed/dev_2k.label $TASK_DATA_FOLDER/processed/dev.label

# Run fairseq preprocessing:
for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1))); do
  LANG="input$INPUT_TYPE"
  fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
    --validpref "${DEVPREF//LANG/$LANG}" \
    --testpref "${TESTPREF//LANG/$LANG}" \
    --destdir "$TASK_DATA_BIN/$LANG" \
    --workers $num_workers \
    --srcdict $TASK_DATA_BIN/dict.txt;
done

# prepare label file
fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
    --validpref "${DEVPREF//LANG/label}" \
    --destdir "$TASK_DATA_BIN/label" \
    --workers ${num_workers};