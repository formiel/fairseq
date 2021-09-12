#!/bin/bash

# Split wav files based on pre-specified segments
INPUT_DIR=$1 # ~/mtedx/fr-en/data/train/wav
OUTPUT_DIR=$2 # ~/mtedx/fr-en/data/train/wav_split
SEGMENT_FILE=$3 # ~/mtedx/fr-en/data/train/txt/segments

mkdir -p $OUTPUT_DIR
count=0
while read line; do
    stringarray=($line)
    output_wav="$OUTPUT_DIR/${stringarray[0]}.wav"
    input_wav="$INPUT_DIR/${stringarray[1]}.wav"
    start=${stringarray[2]}
    end=${stringarray[3]}
    duration=`python -c "print($end - $start)"`
    echo $output_wav $input_wav $start $duration
    count=$(( count + 1 ))
    </dev/null ffmpeg -ss $start -i $input_wav -t $duration -c copy $output_wav
done < $SEGMENT_FILE
echo "Number of lines in segmen file: $count"