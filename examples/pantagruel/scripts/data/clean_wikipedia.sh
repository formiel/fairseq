# Author: Phuong-Hang Le (hangtp.le@gmail.com)
# Date: 26 April 2024

#!/bin/sh
set -e

########################
# Download, extract, split train/dev, and clean data for training
# Vocabulary is built using byte-BPE from HuggingFace tokenizers
# Data is then preprocessed (binarized) ready for training using fairseq-preprocess
########################

FAIRSEQ=$HOME/code/fairseq

DATA_DIR=$1
LG=$2
DATE=$3 #latest: 20240201, reproduced to compare with CamemBERT: "20190701" 
tokenizer=$4 # choose between ["cl100k_base", "byteBPE", "gpt2_bpe", "none"]
num_workers=$5

DST_DIR=$DATA_DIR/${LG}wiki_${DATE} #$SCRATCH/Data/Wikipedia/${LG}wiki_${DATE}
echo "DST_DIR: ${DST_DIR}"
mkdir -p $DST_DIR

XMLFILE=${LG}wiki-${DATE}-pages-articles-multistream.xml
COMPFILE=${XMLFILE}.bz2

GPT2_DICTS=$WORK/pretrained/tokenizers/gpt2_bpe
BIN_NAME=data-bin-add-prefix-false

### Download Wikipedia if not exists
if [[ ! -f $DST_DIR/${COMPFILE} ]]; then
    wget -P $DST_DIR https://dumps.wikimedia.org/${LG}wiki/${DATE}/${COMPFILE}
else
    echo "Wikipedia dump for lang:${LG} at date ${DATE} has already been downloaded".
fi

#### Decompress bz2 file
cd $DST_DIR
echo "Changing to directory: ${DST_DIR}"
if [[ ! -f ${XMLFILE} ]]; then
    bzip2 -dk $COMPFILE
else
    echo "bz2 file has already been de-compressed".
fi

### Exract text with Wikiextractor: using branch e4abb4cb
if [[ ! -f "${DST_DIR}/${LG}wiki.full.raw" ]]; then
    cd $HOME/code/wikiextractor
    echo "Changing to directory: $HOME/code/wikiextractor"
    # Using python 2 to avoid deprecation warnings
    module unload python
    module load python/2.7.16
    echo "Extracting text using wikiextractor ..."
    mkdir -p ${DST_DIR}/raw_e4abb4cb
    python -m WikiExtractor ${DST_DIR}/${XMLFILE} -o ${DST_DIR}/raw_e4abb4cb \
                                            -b 100G \
                                            --no_templates \
                                            --min_text_length 50 \
                                            --processes ${num_workers} \
                                            --filter_disambig_pages \
                                            --log_file log.${LG}  \
                                            --sections \
                                            --lists \
                                            |& tee $DST_DIR/raw_e4abb4cb/wikiextractor.log
                                            # --keep_tables \
    ln -s ${DST_DIR}/raw_e4abb4cb/AA/wiki_00 ${DST_DIR}/${LG}wiki.full.raw
    module unload python
    module load python/3.8.8
fi

cd $FAIRSEQ
echo "Changing to $FAIRSEQ"
### Split into train and dev sets based on articles
if [[ ! -f "${DST_DIR}/${LG}wiki.train" ]]; then
    echo "Splitting train and dev sets..."
    python $FAIRSEQ/examples/pantagruel/scripts/data/split_wikipedia.py ${DATA_DIR} --lang ${LG} --version ${DATE} \
                        |& tee $DST_DIR/split_train_dev.log
fi

# | sed "/^\s*\$/d" \
# | grep -v "^<doc id=" \
# | grep -v "</doc>\$" \

# | awk '/^<\/doc>$/{ $0="\n" } 1' \

### Cleaning
SPLITS="train dev"
for SPLT in $SPLITS; do
    if [[ ! -f "${DST_DIR}/${LG}wiki.${SPLT}.clean" ]]; then
        echo "Cleaning ${SPLT} data..."
        cat ${DST_DIR}/${LG}wiki.${SPLT} \
            | sed '/^\s*$/d' \
            | grep -v "</doc>\$" \
            | awk '/^<doc id=.*$/{ $0="\n" } 1' \
            | awk '/^Section:.*$/ { print "\n" } 1' \
            | sed '1d' \
            > ${DST_DIR}/${LG}wiki.${SPLT}.clean
    fi
done

if [[ ${tokenizer} != "none" ]]; then
    DATA_BIN=$DST_DIR/$BIN_NAME/${tokenizer}
    echo "Creating folder for tokenizer at: ${DATA_BIN}"
    mkdir -p $DATA_BIN
    ### Building vocabulary using HuggingFace tokenizers
    if [[ $tokenizer == "byteBPE" ]]; then
        echo "Learning byte BPE 50K tokenizer using HuggingFace..."
        python $FAIRSEQ/examples/pantagruel/scripts/_learn_tokenizers.py --files $DST_DIR/${LG}wiki.train.cle* --out $DATA_BIN
        ln -s $DATA_BIN/bpe-bytelevel-vocab.json $DATA_BIN/encoder.json
        ln -s $DATA_BIN/bpe-bytelevel-merges.txt $DATA_BIN/vocab.bpe
    elif [[ $tokenizer == "cl100k_base" ]]; then
        echo "Using tokenizer from OpenAI's."
    elif [[ $tokenizer == "gpt2_bpe" ]]; then
        echo "Using gpt2_bpe tokenizer from fairseq."
        FILES="encoder.json dict.txt vocab.bpe"
        for FILE in $FILES; do
            if [[ ! -f ${DATA_BIN}/${FILE} ]]; then
                ln -s ${GPT2_DICTS}/${FILE} $DATA_BIN/${FILE}
            fi
        done
    else
        echo "Invalid tokenizer!"
        exit 1
    fi
fi

SPLITS="dev train"
if [[ ! -f "$DATA_BIN/${LG}wiki.train.bpe"  ]]; then
    if [[ $tokenizer == "byteBPE" ]]; then
        for SPLIT in $SPLITS; do
            echo "Encoding ${SPLIT} using learned byteBPE 50K..."
            python $FAIRSEQ/examples/roberta/multiprocessing_bpe_encoder.py \
                --encoder-json $DATA_BIN/encoder.json \
                --vocab-bpe $DATA_BIN/vocab.bpe \
                --inputs ${DST_DIR}/${LG}wiki.${SPLIT}.clean \
                --outputs ${DATA_BIN}/${LG}wiki.${SPLIT}.bpe \
                --keep-empty \
                --workers ${num_workers};
        done
    elif [[ $tokenizer == "cl100k_base" ]]; then
        echo "Encoding dev set using cl100k_base tokenizer from OpenAI..."
        python $FAIRSEQ/examples/pantagruel/scripts/_encoding_tiktoken.py \
            --inputs ${DST_DIR}/${LG}wiki.dev.clean \
            --outputs ${DATA_BIN}/${LG}wiki.dev.bpe \
            --keep-empty \
            --workers ${num_workers};
        if [[ $LG == "en" ]]; then
            echo "Split train into two files..."
            split -d -b 10G -a 2 $DST_DIR/${LG}wiki.train.clean ${DATA_BIN}/${LG}wiki.train.clean.
            NUMBERS="00 01"
            for NUMBER in $NUMBERS; do
                echo "Encoding train set no.${NUMBER} using cl100k_base tokenizer from OpenAI..."
                python $FAIRSEQ/examples/pantagruel/scripts/_encoding_tiktoken.py \
                    --inputs ${DATA_BIN}/${LG}wiki.train.clean.${NUMBER} \
                    --outputs ${DATA_BIN}/${LG}wiki.train.bpe.${NUMBER} \
                    --keep-empty \
                    --workers ${num_workers};
                wc -l ${DATA_BIN}/${LG}wiki.train.clean.${NUMBER}
                wc -l ${DATA_BIN}/${LG}wiki.train.bpe.${NUMBER}
                cat ${DATA_BIN}/${LG}wiki.train.bpe.${NUMBER} >> ${DATA_BIN}/${LG}wiki.train.bpe
                wc -l ${DATA_BIN}/${LG}wiki.train.bpe
            done
        fi
    elif [[ $tokenizer == "gpt2_bpe" ]]; then
        for SPLIT in $SPLITS; do
            python $FAIRSEQ/examples/roberta/multiprocessing_bpe_encoder.py \
                    --encoder-json $DATA_BIN/encoder.json \
                    --vocab-bpe $DATA_BIN/vocab.bpe \
                    --inputs ${DST_DIR}/${LG}wiki.${SPLIT}.clean \
                    --outputs ${DATA_BIN}/${LG}wiki.${SPLIT}.bpe \
                    --keep-empty \
                    --workers ${num_workers}; \
        done
    else
        echo "Invalid tokenizer!"
        exit 1
    fi
fi

### Binarized data using fairseq
# tricky: change fairseq to not having to change tokenizer learned using HuggingFace
# Line 95 fairseq/tasks/fairseq_task.py: add_special_symbols=False
# Line 35: fairseq/data/dictionary.py:         
        # self.nspecial = 0
        # self.bos_index = 0
        # self.pad_index = 1
        # self.eos_index = 2
        # self.unk_index = 3
# Line 140: fairseq/tasks/masked_lm.py: self.mask_idx = 4

echo "Binarizing data..."
if [[ ! -f "$DATA_BIN/${LG}wiki.train.idx"  ]]; then
    python $FAIRSEQ/examples/pantagruel/scripts/_json2dict.py --json $DATA_BIN/encoder.json
    fairseq-preprocess \
        --only-source \
        --trainpref ${DATA_BIN}/${LG}wiki.train.bpe \
        --validpref ${DATA_BIN}/${LG}wiki.dev.bpe \
        --destdir ${DATA_BIN} \
        --workers ${num_workers} \
        --srcdict $DATA_BIN/dict.txt
fi