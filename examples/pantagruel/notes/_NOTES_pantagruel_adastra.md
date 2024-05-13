rsyncpass -zarvm --exclude=".git*" /Users/hang/github/formiel/slurmx adastra:/lus/home/CT10/c1615074/tphle/code/
rsyncpass -zarvm /Users/hang/github/formiel/fairspeech_dd_updated/ adastra:/lus/home/CT10/c1615074/tphle/code/fairspeech_dd/

cp -rp /lus/work/CT10/c1615074/tphle/Data/Wikipedia/frwiki_20190701 /lus/work/CT10/lig3801/SHARED/data/Modified/Flaubert/

{
    "localPath": "/Users/hang/github/formiel/fairspeech/",
    "remotePath": "umz16dj@jean-zay.idris.fr:/linkhome/rech/genlig01/umz16dj/code/fairspeech/"
    }

            {
            "localPath": "/Users/hang/github/formiel/fairspeech/",
            "remotePath": "adastra:/lus/work/CT10/c1615074/tphle/code/fairspeech_dd/"
        },

rsyncpass -zarvm --exclude=".git*" \
    adastra:/lus/home/CT10/c1615074/tphle/code/fairspeech_torch23/Noir_Une_Franaise_captive_chez_les_Peaux_Rouges_Chap18_3.wav \
    /Users/hang/Downloads/adastra/


# FOR THESIS
## Syncing MuST-C data to Adastra
ssh jean-zay-ccfr.idris.fr
source: /gpfswork/rech/ahm/umz16dj/Data/mustc/en-${LG}
destination: /lus/work/CT10/c1615074/tphle/Data/prepared/mustc

LG=de # t0-jz1
LG=es # t0-jz2
LG=fr # t0-jz3
LG=it # t0-jz4
LG=nl # t0-pp1
LG=pt # t0-pp2
LG=ro # t1-jz2
LG=ru # t1-jz3
rsync -zarvm /gpfswork/rech/ahm/umz16dj/Data/mustc/en-${LG} tphle@adastra-ccfr.cines.fr:/lus/work/CT10/c1615074/tphle/Data/prepared/mustc/

tmux3 jz-3: copy to shared folder on JZ: $ahm_ALL_CCFRWORK/Data
LANGS="de es fr it nl pt ro ru"
for LG in $LANGS; do
    echo "copying en-${LG} data..."
    cp -rp /gpfswork/rech/ahm/umz16dj/Data/mustc/en-${LG} $ahm_ALL_CCFRWORK/Data/mustc/
    echo "Finished copying en-${LG} data."
done 

## Install virtual env on Adastra
```bash
# mkdir $WORK/env
# ln -s $WORK/env $HOME/env
mkdir $HOME/env
module purge
module load cray-python/3.10.10
python3 -m pip install --user --upgrade pip
# Looking in indexes: https://gorgone.cines.fr//root/pypi/+simple/
# Requirement already satisfied: pip in ./.local/lib/python3.10/site-packages (24.0)
# DEPRECATION: omegaconf 2.0.6 has a non-standard dependency specifier PyYAML>=5.1.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of omegaconf or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063

pip3 install --user --upgrade virtualenv

python3 -m virtualenv ~/env/torch22
# created virtual environment CPython3.10.10.final.0-64 in 804ms
#   creator CPython3Posix(dest=/lus/home/CT10/c1615074/tphle/env/torch22, clear=False, no_vcs_ignore=False, global=False)
#   seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/lus/home/CT10/c1615074/tphle/.local/share/virtualenv)
#     added seed packages: pip==24.0, setuptools==69.1.0, wheel==0.42.0
#   activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

source $HOME/env/torch22/bin/activate

python3 -m pip install --upgrade pip
# Looking in indexes: https://gorgone.cines.fr//root/pypi/+simple/
# Requirement already satisfied: pip in ./env/torch22/lib/python3.10/site-packages (24.0)

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# Installing collected packages: mpmath, typing-extensions, sympy, pillow, numpy, networkx, MarkupSafe, fsspec, filelock, pytorch-triton-rocm, jinja2, torch, torchvision, torchaudio
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# fairseq 0.12.2 requires bitarray, which is not installed.
# fairseq 0.12.2 requires cffi, which is not installed.
# fairseq 0.12.2 requires cython, which is not installed.
# fairseq 0.12.2 requires hydra-core<1.1,>=1.0.7, which is not installed.
# fairseq 0.12.2 requires omegaconf<2.1, which is not installed.
# fairseq 0.12.2 requires packaging, which is not installed.
# fairseq 0.12.2 requires regex, which is not installed.
# fairseq 0.12.2 requires sacrebleu>=1.4.12, which is not installed.
# fairseq 0.12.2 requires scikit-learn, which is not installed.
# fairseq 0.12.2 requires tqdm, which is not installed.
# Successfully installed MarkupSafe-2.1.3 filelock-3.9.0 fsspec-2023.4.0 jinja2-3.1.2 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.3 pillow-10.2.0 pytorch-triton-rocm-2.2.0 sympy-1.12 torch-2.2.1+rocm5.7 torchaudio-2.2.1+rocm5.7 torchvision-0.17.1+rocm5.7 typing-extensions-4.8.0

pip install packaging tensorboard einops omegaconf timm==0.5.4

```
<!-- 
Dùng môi trường py39 vợ hấy, anh làm 3.9 để giảm rủi ro conflict với cái 3.10 trong local. Vợ chạy loadpy39 để vô hấy. Vợ “which loadpy39” để xem cái Path của hắn rồi mở ra xem nội dung hấy. Anh bỏ PYTHONPATH ra khỏi ~/.bash_profile rồi, anh định nghĩa hắn cho từng môi trường riêng, trong loadenv với loadpy39.  

Lúc cài đặt trong py39 thì vợ đừng dùng —user hấy

Anh cài PyTorch với fairseq nơi rồi 
Nếu vợ cài lại fairseq thì pip install -e ./ thôi chơ đừng thêm —user

Py39 sẽ dùng folder fairspeech_dd chơ không phải fairspeech vợ hấy

Vợ tạm thời chỉnh lại pysync để hắn sync từ fairspeech lên fairspeech_dd còn đừng đụng chi fairspeech trên nớ hết 

Hiện tại lúc import sẽ bị lỗi numpy, vợ chỉ cần chỉnh trong code lại, thay toàn bộ np.float bằng np.float64 là được 

Vì bản numpy mới không còn np.float nữa
-->

```bash
# installed in env
. $OWN_HOMEDIR/loadpy39.sh
cd $HOME/code/fairspeech_dd
pip install -e ./
# Building wheels for collected packages: fairseq
#   Building editable for fairseq (pyproject.toml) ... done
#   Created wheel for fairseq: filename=fairseq-1.0.0a0+fc3a4cc-0.editable-cp39-cp39-linux_x86_64.whl size=8682 sha256=a34030f63be7a0a6a13d3d4da8eac37ba00c725328c5081dd3bb0ccaec3c26b2
#   Stored in directory: /tmp/pip-ephem-wheel-cache-p6ldkhwb/wheels/e7/f8/2f/b5cb0ad6f46921116e4b98ce1ce7abe62ce6392925952cfa0e
# Successfully built fairseq
# DEPRECATION: omegaconf 2.0.6 has a non-standard dependency specifier PyYAML>=5.1.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of omegaconf or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
# Installing collected packages: fairseq
#   Attempting uninstall: fairseq
#     Found existing installation: fairseq 0.12.2
#     Not uninstalling fairseq at /lus/work/CT10/c1615074/tphle/code/fairspeech_dd, outside environment /lus/home/CT10/c1615074/tphle/env/py39
#     Can't uninstall 'fairseq'. No files were found to uninstall.
# Successfully installed fairseq-1.0.0a0+fc3a4cc
```

# DATA
- Dataset: mls_french_jz, number of audio files: 
INFO:root:Total number of audio files: 263055 (checked json files already, same number of examples, no split at all)
INFO:root:n_train=258213, n_val=2416, n_test=2426
(LB=263,055)
- Dataset: African_Accented_French, number of audio files: 16656 (LB=16,402)
- Dataset: Att-HACK_SLR88, number of audio files: 36634 (LB=36,339)
- Dataset: CaFE, number of audio files: 936 (LB=936)
- Dataset: GEMEP, number of audio files: 1260 (LB=1,236)
- Dataset: MaSS, number of audio files: 8219 (LB=8,219) xxx
- Dataset: voxpopuli_transcribed, number of audio files: 77030 (LB=76.281)
- Dataset: studios-tamani-kalangou-french, number of audio files: 38332
- Dataset: CFPP_corrected, number of audio files: 12577 (LB=9853)
- Dataset: Portmedia, number of audio files: 20264 (LB=19,627)
- Dataset: ESLO, number of audio files: 62918 (LB=62,918) xxx
- Dataset: NCCFr, number of audio files: 29421 (LB=29,421) xxx


- Dataset: audiocite_with_metadata, number of audio files: 818388
- Dataset: EPAC_flowbert, number of audio files: 623250 (LB=623,250) xxx
- Dataset: MPF, number of audio files: 40579 (LB=19,527)
- Dataset: TCOF_corrected, number of audio files: 84600 (LB=58,722)
- Dataset: voxpopuli_unlabeled, number of audio files: 570192 (LB=568,338)


```bash
# final preprocessing on-the-fly
# mls_french_jz (number examples matched with paper)
100%|██████████| 258213/258213 [03:46<00:00, 1137.82it/s]
0: INFO:root:Total duration of TRAIN: 17225288.38638889 (s) # should be divided by 16000
0: INFO:root:Writing manifest file for split:VALID
 73%|███████▎  | 1767/2416 [00:01<00:00, 1090.47it/s]
100%|██████████| 2416/2416 [00:02<00:00, 1072.82it/s]
0: INFO:root:Total duration of VALID: 161139.67111111112 (s)
0: INFO:root:Writing manifest file for split:TEST
 75%|███████▌  | 1825/2426 [00:01<00:00, 1158.81it/s]
100%|██████████| 2426/2426 [00:02<00:00, 1112.39it/s]
0: INFO:root:Total duration of TEST: 161075.07666666666 (s)
0: INFO:root:Total running time: 770.7958646615347 (minutes)

# studios-tamani-kalangou-french (umber examples matched with paper)
100%|██████████| 38332/38332 [00:38<00:00, 990.64it/s] ]
0: INFO:root:Total duration of TRAIN: 1776299.3236111111 (s)
0: INFO:root:Total running time: 344.5364379405975 (minutes)

# African_Accented_French (in paper: 16402 vs. 16491: add back ~1h of training data)
100%|██████████| 14313/14313 [00:10<00:00, 1423.17it/s]
0: INFO:root:Total duration of TRAIN: 287948.5519444444 (s)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1703/1703 [00:01<00:00, 1383.73it/s]
0: INFO:root:Total duration of VALID: 25302.222222222223 (s)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 475/475 [00:00<00:00, 1589.26it/s]
0: INFO:root:Total duration of TEST: 4859.500555555555 (s)
0: INFO:root:Total running time: 300.23300757010776 (minutes)

0: INFO:root:Number of audio files exist in corpus directory: 16638
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 16402
0: INFO:root:Number of missing files longer than 1s: 67
0: INFO:root:Total duration discarded (files less than 1s): 125.09518750000001 (seconds)
0: INFO:root:Total duration added back to training data: 3386.0 (seconds)
0: INFO:root:num_file_splits: 41
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 16491 - 14313/1703/475
0: INFO:root:Number of audio files split additionally: 41

# Att-HACK_SLR88 (number examples matched with paper)
0: 0%|██████████| 36339/36339 [00:25<00:00, 1439.26it/s]
0: INFO:root:Total duration of TRAIN: 432779.0077777778 (s)
0: INFO:root:Total running time: 312.86436237891513 (minutes)
0: INFO:root:Number of audio files exist in corpus directory: 36634
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 36339
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 248.23706250000006 (seconds)
0: INFO:root:Total duration added back to training data: 0 (seconds)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 36339 - 36339/0/0
0: INFO:root:Number of audio files split additionally: 0

# CaFE (number examples matched with paper)
0: INFO:root:Writing manifest file for split:TRAIN
100%|██████████| 936/936 [00:00<00:00, 3100.87it/s]
0: INFO:root:Total duration of TRAIN: 18477.628333333334 (s)
0: INFO:root:Total running time: 3.4625381310780843 (minutes)

# CFPP_corrected (paper: 9853, different from json)
100%|██████████| 12619/12619 [00:07<00:00, 1798.43it/s]
0: INFO:root:Total duration of TRAIN: 286779.635 (s)
0: INFO:root:Total running time: 298.22998795111977 (minutes)
0: INFO:root:Number of audio files exist in corpus directory: 12577
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 12577
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (seconds)
0: INFO:root:Total duration added back to training data: 0 (seconds)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 12619 - 12619/0/0
0: INFO:root:Number of audio files split additionally: 76

# ESLO (number examples matched with paper)
100%|██████████| 62918/62918 [00:46<00:00, 1367.73it/s]
0: INFO:root:Total duration of TRAIN: 547423.5283333333 (s)
0: INFO:root:Total running time: 324.9329766233762 (minutes)

# EPAC_flowbert (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 623250
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 623250
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (seconds)
0: INFO:root:Total duration added back to training data: 0 (seconds)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 623250 - 623250/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 26016707.75611111 (s)

# GEMEP (number examples matched with paper)
100%|██████████| 1236/1236 [02:55<00:00,  7.03it/s]
0: INFO:root:Writing manifest file for split:TRAIN
100%|██████████| 1236/1236 [00:00<00:00, 3059.41it/s]
0: INFO:root:Total duration of TRAIN: 13530.496388888889 (s)
0: INFO:root:Total running time: 3.7379492044448854 (minutes)

# MPF (error, job: 876008)

# Portmedia (paper: 19627)
100%|██████████| 19763/19763 [00:19<00:00, 1036.86it/s]
0: INFO:root:Total duration of TRAIN: 646721.3402777778 (s)
0: INFO:root:Total running time: 306.892623869578 (minutes)
0: INFO:root:Number of audio files exist in corpus directory: 20264
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 19627
0: INFO:root:Number of missing files longer than 1s: 117
0: INFO:root:Total duration discarded (files less than 1s): 413.3381249999994 (seconds)
0: INFO:root:Total duration added back to training data: 5146.336187499998 (seconds)

# TCOF_corrected (paper: 58722)
0: INFO:root:Number of audio files exist in corpus directory: 84600
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 84592
0: INFO:root:Number of missing files longer than 1s: 3
0: INFO:root:Total duration discarded (files less than 1s): 0.576 (seconds)
0: INFO:root:Total duration added back to training data: 5.723 (seconds)
100%|██████████| 84343/84343 [01:08<00:00, 1235.97it/s]
0: INFO:root:Total duration of TRAIN: 929453.7761111112 (s)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 350/350 [00:00<00:00, 1797.31it/s]
0: INFO:root:Total duration of TEST: 4066.9644444444443 (s)
0: INFO:root:Total running time: 345.4711463928223 (minutes)

# MaSS (number examples matched with paper)
100%|██████████| 8219/8219 [00:05<00:00, 1525.12it/s]
0: INFO:root:Total duration of TRAIN: 314891.8283333333 (s)
0: INFO:root:Total running time: 291.4240753173828 (minutes)

# NCCFr (number examples matched with paper)
100%|██████████| 29421/29421 [00:20<00:00, 1401.27it/s]
0: INFO:root:Total duration of TRAIN: 425404.31694444444 (s)
0: INFO:root:Total running time: 303.493234705925 (minutes)

# voxpopuli_unlabeled (paper 568,338)
0: INFO:root:Number of audio files exist in corpus directory: 570192
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 570192
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (seconds)
0: INFO:root:Total duration added back to training data: 0 (seconds)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 570192 - 570192/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 72763841.19166666 (s)
0: INFO:root:Total running time: 1100.8159702181815 (minutes)

# voxpopuli_transcribed (paper 76.281)
0: INFO:root:Number of audio files exist in corpus directory: 77030
0: INFO:root:Getting paths from json...
0: INFO:root:Number of audio files read from JSON: 77030
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (seconds)
0: INFO:root:Total duration added back to training data: 0 (seconds)
0: INFO:root:num_file_splits: 15
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 77038 - 77038/0/0
0: INFO:root:Number of audio files split additionally: 15
100%|██████████| 77038/77038 [01:22<00:00, 936.49it/s] 
0: INFO:root:Total duration of TRAIN: 3448953.0505555556 (s)
0: INFO:root:Total running time: 371.0742014408112 (minutes)
```
