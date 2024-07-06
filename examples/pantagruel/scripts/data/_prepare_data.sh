#!/bin/bash

# NOT USED

DSTDIR=$1
DATA_DIR=/gpfsdswork/dataset/LibriSpeech
SPLITS="train-clean-100 
        train-clean-360 
        train-other-500 
        dev-clean 
        dev-other 
        test-clean  
        test-other"
max_duration=35
chunk_length=20
count=0

for SPLIT in $SPLITS; do
    mkdir -p $DSTDIR/$SPLIT
    for file in $DATA_DIR/$SPLIT/*/*/*; do
        fname=$(basename $file)
        fname_noext="${fname%.*}"
        if [[ $fname == *".flac"* ]]; then
            duration=$(sox --info -D $file)
            # check if file is over max_duration
            if (( $(echo "$duration > $max_duration" |bc -l) )); then
                echo "$file: $duration" >> $DSTDIR/$SPLIT/over_max_duration_list.txt
                count=$(( count + 1 ))
                # split the files into chunks of less than chunk_length
                sox $file $DSTDIR/$SPLIT/$fname_noext-.flac trim 0 $chunk_length : newfile : restart
            else
                # create symlink
                ln -s $file $DSTDIR/$SPLIT/$fname
            fi 
        fi
    done
done
echo "Number of files over ${max_duration}s: ${count}"
echo "Splitted these files into chunks of less than ${count}"
echo "Saving these filenames in $DSTDIR/$SPLIT/over_max_duration_list.txt"