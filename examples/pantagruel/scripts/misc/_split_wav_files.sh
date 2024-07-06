#!/bin/bash

# module load ffmpeg/4.2.2

# Split wav files based on pre-specified segments for mTEDx corpus
DATA_ROOT=$1
SPLITS="valid test train"

for SPLIT in $SPLITS; do 
    echo "Splitting files for split: ${SPLIT}"
    INPUT_DIR="$DATA_ROOT/data/${SPLIT}/wav"
    OUTPUT_DIR="$DATA_ROOT/data/${SPLIT}/wav_splits_v1"
    rm -r $OUTPUT_DIR
    SEGMENT_FILE="$DATA_ROOT/data/${SPLIT}/txt/segments"
    echo "Deleting previous incorrect output files..."
    mkdir -p $OUTPUT_DIR
    count=0
    while read line; do
        echo "................................................................."
        stringarray=($line)
        output_wav="$OUTPUT_DIR/${stringarray[0]}.wav"
        input_wav="$INPUT_DIR/${stringarray[1]}.flac"
        start=${stringarray[2]}
        end=${stringarray[3]}
        # duration=`python -c "print($end - $start)"`
        # echo $output_wav $input_wav $start $duration
        count=$(( count + 1 ))
        </dev/null ffmpeg -i $input_wav -ss $start -to $end -ar 16000 $output_wav
        # </dev/null ffmpeg -ss $start -i $input_wav -t $duration -c copy -ar 16000 $output_wav
    done < $SEGMENT_FILE
    echo "Number of lines in segment file: $count"
    echo "---------------------------------------------------------------------"
done