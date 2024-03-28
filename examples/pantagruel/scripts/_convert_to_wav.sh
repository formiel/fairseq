#!/bin/sh

# module load ffmpeg/4.2.2
DATA_RAW=$SCRATCH/Data/covost2/raw/fr
mkdir -p $DATA_RAW/clips_wav

count=0
for file in $DATA_RAW/clips/*.mp3; do
    basename=$(basename "${file%.*}")
    echo $basename
    count=$(( count + 1 ))
    ffmpeg -i "$file" -ar 16000 "$DATA_RAW/clips_wav/${basename}.wav"
    if [[ ! -f "${DATA_RAW}/clips_wav/${basename}.wav" ]]; then
        echo "Not successfully convert to wav for ${file}!" > $DATA_RAW/mp3_to_wav.fail.log
    fi
done
echo "Total number of files: ${count}"

# # Check number of files
# ls -lth clips/* | wc -l
# ls -lth clips_wav/* | wc -l
# # There are missing files!!!!
# for file in $DATA_RAW/clips/*; do
#         basename=$(basename $file)
#         # echo $basename
#         if [[ ! -f $DATA_RAW/clips_wav/${basename}.wav ]]; then
#                 # echo "$basename NOT converted to wav file yet!"
#                 ffmpeg -i $file $DATA_RAW/clips_wav/${basename}.wav
#         fi
# done