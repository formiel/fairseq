# Speech
## French MLS
Two sets of experiments
- Original French MLS: without removing files over 30seconds
- Same as Lebenchmark S model: using json files provided

### Phase 1: PRE-TRAINING
- _run_speech_pretraining.sh

### Phase 2: FINE-TUNING WITH CTC FOR ASR TASK
#### 1. Fine-tune on CommonVoice/fr datasets with CTC module
- _run_speech_finetuning_asr_cv_ctc.sh

#### 2. Fine-tune on CommonVoice/fr datasets with wav2vec seq2seq
- _run_speech_finetuning_asr_cv_ce.sh


### Phase 2: FINE-TUNING WITH TRANSFORMER DECODER FOR ST TASK
##### 1. CoVoST-2
- _run_speech_finetuning_st_cv2.sh

##### 2. mTEDx
- _run_speech_finetuning_st_mtedx.sh


# Text
## French Wikipedia pre-training
INFO: Finished 48-process extraction of 2121083 articles in 6975.3s (304.1 art/s)
INFO: total of page: 3018743, total of articl page: 2121106; total of used articl page: 2121083

## Phase 1: PRE-TRAINING
- _run_text_pretraining.sh