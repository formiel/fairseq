# pysync
{
        "localPath": "/Users/hang/github/formiel/fairspeech/",
        "remotePath": "adastra:/lus/work/CT10/c1615074/tphle/code/fairspeech/"
        },


# Sync data
rsync -chavzP --stats \
        /Users/hang/github/formiel/fairspeech/ \
        umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/fairspeech/

scp -r -p -3 lethip@decore1:ms_data/bert_fr.tar.gz \
        umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/Data/

scp -r -p -3 lethip@decore1:/home/getalp/evains/MLS/mls_french \
        umz16dj@jean-zay.idris.fr:/gpfsscratch/rech/ahm/umz16dj/Data/

cp -rp $mff_CCFRWORK/Data $ahm_CCFRWORK/ 

rsync -chavzP --stats \
        /Users/hang/github/formiel/fairspeech/ \
        umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/code/fairspeech/
rsyncpass -zarvm --exclude="*.git*" \
        /Users/hang/github/formiel/fairspeech/ \
        adastra:/lus/home/CT10/c1615074/tphle/code/fairspeech/

rsync -chavzP --stats \
        umz16dj@jean-zay.idris.fr:/gpfsscratch/rech/ahm/umz16dj/Data/mTEDx/fr-en/data/train/wav_splits/If92mr3B_Og_0028.wav \
        /Users/hang/Downloads/pantagruel/
        
        
        
grep -n "京" /gpfswork/rech/ahm/umz16dj/Data/CommonVoice/fr/train.wrd
> 197038: Seconde exposition du groupe des Étoiles 星星, Musée des Beaux Arts de Pékin 北京美术馆. 
sed -n '52p'

rsync -chavzP --stats --exclude "*debug*" --exclude "*base_speech_mls1k*" \
        umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/experiments/fairseq_tensorboard/* \
        /Users/hang/tensorboard/

rsyncpass -zarvm --max-size=100M --exclude="*debug*" \
        adastra://lus/work/CT10/c1615074/tphle/experiments/fairseq_tensorboard/* \
        /Users/hang/tensorboard/


rsyncpass -zarvm --max-size=100M --exclude="*debug*" \
        adastra:/lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_raw/MPF/output_waves/Auphelie1b_ANON_060318_151118_s_1_spk_Aurélie.wav \
       /Users/hang/Downloads/adastra/

ssh adastra-ccfr.cines.fr
rsync -zarvm /lus/home/CT10/c1615074/tphle/experiments/fairseq_tensorboard/pantagruel/adastra umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/experiments/fairseq_tensorboard/pantagruel/

ssh jean-zay-ccfr.idris.fr
rsync -zarvm /gpfswork/rech/ahm/umz16dj/Data/Wikipedia/enwiki_20240201/data-bin/byteBPE/debug \
        tphle@adastra-ccfr.cines.fr:/lus/work/CT10/c1615074/tphle/Data/prepared/Wikipedia/enwiki_20240201/data-bin/byteBPE/
        


# Common errors
```python
from omegaconf import open_dict
# no key in construct
if not getattr(task_cfg, "multi_corpus_keys", None):
        with open_dict(task_cfg):
                task_cfg.multi_corpus_keys = None
```

# Replicate data2vec recipe for English
## Speech: Librispeech
- Prepare manifest with wav2vec_manifest.py
- Collect all training examples into 1 tsv file
- Run training:
```bash
### on 1 GPU
FAIRSEQ=$HOME/code/fairspeech
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/pretraining
CONFIG=debugg
DATA_DIR=$WORK/Data/LibriSpeech
USER_DIR=$FAIRSEQ/examples/data2vec
GPUS=1
fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml +task.data=${DATA_DIR} common.user_dir=${USER_DIR}

### on many GPUs
GPUS=16
FAIRSEQ=$HOME/code/fairspeech
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/speech/pretraining
DATA_DIR=$WORK/Data/LibriSpeech
USER_DIR=$FAIRSEQ/examples/data2vec
CONFIG=base_audio_only_task_ngpu16_reproduce_en_torch2sdpa_stable
submit run gpu_p5 $GPUS 12 3 $CONFIG "fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml +task.data=${DATA_DIR} common.user_dir=$USER_DIR"
# job 0: 664469, 665909
# job 1: 665910 (after 665909)
# job 2: 665911 (after 665910)
```

- Fine-tuning with CTC
```bash
## prepare labelled manifest with libri_labels.py for LibriSpeech
SPLITS="train-full dev-clean dev-other test-clean test-other"
output_dir=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech
for split in $SPLITS; do
        tsv_path=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/${split}.tsv
        python examples/wav2vec/libri_labels.py ${tsv_path} --output-dir ${output_dir} --output-name $split
done


## prepare labelled manifest with libri_labels.py for labelled data of LibriLight
# prepare manifest for 10h labeled data of LibriLight
SPLITS="1h 9h"
DST=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
for SPLIT in $SPLITS; do
        ROOT=/gpfsscratch/rech/ahm/umz16dj/Data/LibriSpeech_raw/librispeech_finetuning/${SPLIT}
        python examples/wav2vec/wav2vec_manifest.py $ROOT --valid-percent 0 --dest $DST
        mv $DST/train.tsv $DST/train-labelled-${SPLIT}.tsv
done
# Collect all training examples into 1 tsv file
python examples/pantagruel/scripts/_modify_manifest.py --tsv-root $DST --tsv-files "train-labelled-1h,train-labelled-9h" --output-file train-labelled-10h

# prepare labelled manifest from the "unlabelled mmanifest"
split=train-labelled-10h
tsv_path=$DST/${split}.tsv
python examples/wav2vec/libri_labels.py ${tsv_path} --output-dir $DST --output-name $split

SPLITS="dev-clean dev-other"
SFXS="tsv ltr wrd"
SRC=$WORK/Data/LibriSpeech
DST=$SRC/librilight_10h_labelled
for split in $SPLITS; do
        for SFX in $SFXS; do
                ln -s $SRC/${split}.${SFX} $DST/${split}.${SFX}
        done
done

# Run fine-tuning
GPUS=4
FAIRSEQ=$HOME/code/fairspeech
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_reproduce_en
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${PRETRAIN_CONFIG}/checkpoint_best.pt
# CHECKPOINT=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/speech/base_libri.pt
CONFIG=vox_10h
# CONFIG=vox_10h_fairseq_ckpt
# CONFIG=vox_10h_hyprparam_w2v2_incrbsz
# CONFIG=base_10h_reproduce_2
submit run gpu_p1 $GPUS 10 4 $CONFIG "fairseq-hydra-train task.data=$DATA common.user_dir=examples/data2vec model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}"
# job 0: 567572
# job 1: 567574 (after 567572)
# job 2: 567575 (after 567574)
# job 3: 567576 (after 567575)

# on multiple GPUs on 1 node
CONFIG=base_10h_reproduce_torch2sdpa_stable_adam
GPUS=2
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_reproduce_en_torch2sdpa_stable
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${PRETRAIN_CONFIG}/checkpoint_best.pt
submit run gpu_p5 $GPUS 8 1 $CONFIG "fairseq-hydra-train task.data=$DATA common.user_dir=examples/data2vec model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}"
# job 0: 695436

# on multiple GPUs on 1 node
CONFIG=debug
fairseq-hydra-train task.data=$DATA common.user_dir=examples/data2vec model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}

# seq2seq model: using pre-trained speech data2vec as feature extractor
PRETRAIN_CONFIG=base_audio_only_task_ngpu16_reproduce_en
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${PRETRAIN_CONFIG}/checkpoint_best.pt
CONFIG=en_speech_base_librispeech1k_features_asr_encdec
fairseq-hydra-train task.data=$DATA common.user_dir=examples/data2vec model.w2v_path=$CHECKPOINT --config-dir examples/pantagruel/configs/speech/finetuning --config-name ${CONFIG}
```

- Decoding a fine-tuned model without external LM
```bash
CONFIG=base_10h_reproduce
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${CONFIG}/checkpoint_best.pt
# CHECKPOINT=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/speech/${CONFIG}.pt
python $FAIRSEQ/examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
--config-name infer task=audio_finetuning task.data=${DATA} common.user_dir=examples/data2vec \
common_eval.results_path=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/pantagruel/${CONFIG} \
task.labels=ltr \
dataset.gen_subset=dev-other \
common_eval.path=${CHECKPOINT} decoding.beam=5
# using sdpa:
# [2024-02-15 14:17:44,737][__main__][INFO] - Processed 2862 sentences (267906 tokens) in 36.1s 79.35 sentences per second, 7372.78 tokens per second)                                                            
# [2024-02-15 14:17:44,742][__main__][INFO] - Word error rate: 11.5102
# previous implementation


# decoding data2vec2.0 model
CONFIG=base_libri_960h
DATA=/gpfswork/rech/ahm/umz16dj/Data/LibriSpeech/librilight_10h_labelled
# CHECKPOINT=/gpfsscratch/rech/ahm/umz16dj/Experiments/fairseq_checkpoints/${PRETRAIN_CONFIG}/checkpoint_best.pt
CHECKPOINT=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/speech/${CONFIG}.pt
python examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
--config-name infer task=audio_finetuning task.data=${DATA} common.user_dir=examples/data2vec \
common_eval.results_path=/gpfswork/rech/ahm/umz16dj/pretrained_models/data2vec/v2/speech \
task.labels=ltr \
dataset.gen_subset=dev-other \
common_eval.path=${CHECKPOINT} decoding.beam=5
```


## Text:
- English/French Wikipedia latest dump on 01/02/2024
LG=fr
wiki_dump=$SCRATCH/Data/Wikipedia/${LG}wiki/${LG}wiki-20240201-pages-articles-multistream.xml.bz2
python -m wikiextractor.WikiExtractor $wiki_dump -o $SCRATCH/Data/Wikipedia/${LG}wiki/WikiExtractor_output -b 100M

<!-- # English:
# Using commit e4abb4cbd019b0257824ee47c23dd163919b731b
# INFO: Finished 48-process extraction of 6624735 articles in 7239.7s (915.1 art/s)
# INFO: total of page: 12107421, total of articl page: 6777403; total of used articl page: 6624735 (count using python started with docid: 6611255)

# Using master branch of wikiextractor without `--min_text_length 50 --filter_disambig_pages --sections --lists`
# INFO: Finished 48-process extraction of 17457964 articles in 11050.2s (1579.9 art/s)

# Upon preliminary observation: content of list items are not kept using the master branch

# Learning BPE50K using SentencePiece implementation
trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
trainer_interface.cc(537) LOG(INFO) all chars count=20409058966
trainer_interface.cc(548) LOG(INFO) Done: 100% characters are covered.
trainer_interface.cc(558) LOG(INFO) Alphabet size=22922
trainer_interface.cc(559) LOG(INFO) Final character coverage=1
trainer_interface.cc(591) LOG(INFO) Done! preprocessed 129903080 sentences.
trainer_interface.cc(597) LOG(INFO) Tokenizing input sentences with whitespace: 129903080
trainer_interface.cc(608) LOG(INFO) Done! 36905957


# French:
# INFO: Finished 48-process extraction of 2587937 articles in 3032.4s (853.4 art/s)
# INFO: total of page: 3803278, total of articl page: 2587938; total of used articl page: 2587937 
INFO: Finished 48-process extraction of 2587937 articles in 9472.4s (273.2 art/s)
INFO: total of page: 3803278, total of articl page: 2587938; total of used articl page: 2587937
-->

<!-- Ở dòng 77 vợ thử thay `self._optimizer = Adam(params, **self.optimizer_config)` bằng `self._optimizer = torch.optim.AdamW(params, **self.optimizer_config)`. Sau đó trong configuration của training ở chỗ optimizer thêm `fused=True`. Hắn sẽ nhanh thêm được kha khá đó vợ. -->

- Split into train, dev sets and clean using script: examples/pantagruel/_clean_wikipedia.sh
- Learn vocabulary using SentencePiece: examples/pantagruel/_learn_vocab.sh
- Prepare binarized data using fairseq-preprocess
```bash
DATADIR=$SCRATCH/Data/Wikipedia/enwiki

fairseq-preprocess \
    --only-source \
    --srcdict $DATADIR/databin-bpe50k/spm_bpe50K.txt \
    --trainpref $DATADIR/enwiki.train.clean \
    --validpref $DATADIR/enwiki.dev.clean \
    --destdir $DATADIR/databin-bpe50k \
    --workers 48  

# Preprocess using tokenizer from tiktoken
LG=en
DST_DIR=$SCRATCH/Data/Wikipedia/${LG}wiki
DATA_BIN=$DST_DIR/data-bin/cl100k_base
SPLITS="dev train"
# mkdir -p $DATA_BIN
# split -d -b 1G -a 2 $DST_DIR/${LG}wiki.train.clean $DATA_BIN/${LG}wiki.train.clean.
NUMBERS="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19"

for NUMBER in $NUMBERS; do
        # python $FAIRSEQ/examples/pantagruel/scripts/_encoding_tiktoken.py \
        #     --inputs ${DATA_BIN}/${LG}wiki.train.clean.${NUMBER} \
        #     --outputs ${DATA_BIN}/${LG}wiki.train.bpe.${NUMBER} \
        #     --keep-empty \
        #     --workers 48;
        wc -l ${DATA_BIN}/${LG}wiki.train.clean.${NUMBER}
        wc -l ${DATA_BIN}/${LG}wiki.train.bpe.${NUMBER}
done
```

- Pre-training
```bash
TASK=pretraining
MODALITY=text
USER_DIR=$FAIRSEQ/examples/data2vec

DATA_DIR=$WORK/Data/wikitext-103/data-bin/wikitext-103
CONFIG_DIR=$FAIRSEQ/examples/pantagruel/configs/${MODALITY}/${TASK}

CONFIG=base_wikipedia

GPUS=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1)
EXPNAME="${CONFIG}_debug" # to be changed
TENSORBOARD_DIR=$WORK/experiments/fairseq_tensorboard/pantagruel/${MODALITY}/${TASK}/${EXPNAME}
SAVE_DIR=$WORK/experiments/fairseq_checkpoints/pantagruel/${MODALITY}/${TASK}/${EXPNAME}

submit run gpu_p5 $GPUS 2 6 $CONFIG "fairseq-hydra-train --config-dir ${CONFIG_DIR} --config-name $CONFIG.yaml task.data=${DATA_DIR} common.user_dir=$USER_DIR common.tensorboard_logdir=${TENSORBOARD_DIR} checkpoint.save_dir=${SAVE_DIR} distributed_training.distributed_world_size=${GPUS} distributed_training.distributed_port=${MASTER_PORT}"
```


# Notes
- No need to call python as it is already included in torchrun
- If using 1 fairseq: should put PYTHONPATH and FAIRSEQ in bash_profile


# Data structure in each server
## JEAN ZAY
$HOME/Data -> $WORK/Data (should pay attention to data on $SCRATCH)
- CommonVoice: used to evaluate the French-based speech model (clips in manifest tsv files point to $DSDIR)
- covost2: used for both ST and pantagruel
  + covost2/fr
    * fr.tar.gz -> /gpfsstore/rech/yul/umz16dj/COVOST2/fr.tar.gz
    * clips_wav -> /gpfsscratch/rech/ahm/umz16dj/Data/covost2/raw/fr/clips_wav
    * ssl-encoder (including data for finetuning SSL encoder)
- Flaubert1_data
- flue_data
- glue_data
- LibriSpeech:
  + raw -> /gpfsscratch/rech/ahm/umz16dj/Data/LibriSpeech_raw
- MLS_French:
  + raw -> /gpfsssd/scratch/rech/ahm/umz16dj/Data/mls_french_jz
- mTEDx: currently used for pantaguem with subset fr-en
  + raw -> /gpfsscratch/rech/ahm/umz16dj/Data/mTEDx/fr-en
- mustc: dictionaries work well with Siamese project (haven't checked with adapters yet, but need to re-factor adaptor code anyway). Should use this dictionary when re-training dual-decoder!
- Wikipedia -> $SCRATCH/Data/Wikipedia
- wikitext-103
- wmt14_en_de
- wmt14_en_fr
- wmt17_en_de

$SCRATCH/Data
- BookCorpus: bookcorpus.txt (4GB), downloaded from HuggingFace, in lower-case and space tokenized
- CommonVoice: the raw data of cv-corpus-6.1-2020-12-11 copied from $DSDIR, but currently not used, to check and remove to save space and avoid confusion
  + clips
  + clips_wav
- CommonVoice.tar.gz
- covost2
- covost2.tar.gz
- glue_data
- LibriSpeech_raw
- mls_french: data provided by Solène, with $SPLIT.json included audio and corresponding transcripts 
- mls_french_jz: copied from JZ, don't need to use this, to check and remove to save space and avoid confusion 
- mls_french_jz.tar.gz
- mTEDx: raw data, should be kept!!!
- mTEDx.tar.gz
- mustc: included en-fr/dev/waveform which were used for analysis in the PhD defense, the samples are cut based on segment and then these wave forms 
- Wikipedia: VERY important raw data, should be kept!!!

## Adastra
HOME=/lus/home/CT10/c1615074/tphle
WORK=/lus/scratch/CT10/c1615074/tphle
$HOME/Data -> $WORK/Data
- prepared
  + CommonVoice: not run fine-tuning on Adastra yet!
  + LibriSpeech
  + MLS_French
  + Wikipedia
  + covost2
  + flue_data
  + glue_data
  + mTEDx
  + mustc
- raw
  + CommonVoice.tar.gz
  + covost2.tar.gz
  + mTEDx.tar.gz
  + mls_french_jz.tar.gz
$SCRATCH/Data
- LibriSpeech_raw
- mTEDx
- mustc

