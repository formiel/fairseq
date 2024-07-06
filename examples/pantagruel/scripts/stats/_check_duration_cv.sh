#!/bin/sh

# module load sox/14.4.2

AUDIO_DIR=/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr/clips_wav
# structure: */*.wav
total_duration=0
for folder in $AUDIO_DIR/*; do
    for file in $folder/*; do
        duration=$(sox --info -D $file)
        echo "$file: ${duration} (s)"
        total_duration=$(bc <<< "${total_duration:-0} + ${duration// /+}")
    done
done
echo "Total duration of CommonVoice fr: ${total_duration} (s)"
# Total duration of CommonVoice fr: 2458423.995406 (s)