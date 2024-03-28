# First set of experiments for speech
## Replicate on English audio (LibriSpeech 960 hours)
1. Pre-training
- *base_audio_only_task_ngpu16_reproduce_en*
    + base model on 960h, training on 16 A100 80GB
    + very good loss, dev loss continues to decrease after training reaches maximum updates (400K)
    + 19.3h for 213K steps, 16h for 175K next steps (from 225K-400K): ~37 hours (paper: 43h on 16 A100 40GB GPUs)
- *base_audio_only_task_ngpu16_reproduce_en_torch2sdpa_stable*: replace origina implementation by Pytorch's sdpa
    + 32 hours (4h for 50k steps)
    + finetuning WER 11.22 (better than original implementation of 11.52)
2. Fine-tuning on 10h
- Results on paper on dev-other without external LM: WER a little bit less than 11.5% (shown on plots)
- *vox_10h_reproduce*: WER 11.77 (run on 4GPUs gpu_p1 of batch size 1.28M tokens, lr 1e-4, taking ~8 hours)
- *base_10h_reproduce*: WER 11.52 (run on 2GPUs gpu_p5 of batch size 3.2M tokens, lr 5e-5, taking ~9 hours)
- *base_10h_reproduce_torch2sdpa_stable*: WER 11.22 (taking ~5 hours)

## Pre-training on French MLS 1K
- Base model finetuned on CommonVoice: 
    + dev set: WER 8.92 vs. 9.49 in LeBenchmark 2.0 using Large model on French MLS 1K
    + test set: WER vs. 11.21
- Training large model... 

## Finetuning for ASR task on CommonVoice
- WER on dev set (evaluated during training): CTC is better than CE
- BUT ONE BIG ISSUE: WER looks good, but loss looks overfitting (will check further later) 


##### GPU usage
- Finetuning base model on CV with eff. bsz 3.2M x freq1 x gpu8 = 25.6M: 48G 