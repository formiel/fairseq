rsyncpass -zarvm --exclude=".git*" /Users/hang/github/formiel/transformers adastra:/lus/home/CT10/c1615074/tphle/code/
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
#### LeBenchmark_prepared: 10% but not exceed 1k examples, remove examples less than 3000 frames
# 883597 -> 883613

# mls_french_jz
0: INFO:root:Number of audio files exist in corpus directory: 263055
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 263055 - 258213/2416/2426
0: INFO:root:Number of audio files split additionally: 
0: 100%|██████████| 258213/258213 [04:28<00:00, 962.26it/s]
0: INFO:root:Total duration of TRAIN: 1076.5805241493056 (h)
0: INFO:root:Writing manifest file for split:VALID
 70%|███████   | 1696/2416 [00:01<00:00, 976.78it/s] 
100%|██████████| 2416/2416 [00:02<00:00, 1007.73it/s]
0: INFO:root:Total duration of VALID: 10.071229444444445 (h)
0: INFO:root:Writing manifest file for split:TEST
 69%|██████▌   | 1583/2426 [00:01<00:00, 994.90it/s]
100%|██████████| 2426/2426 [00:02<00:00, 985.07it/s]
0: INFO:root:Total duration of TEST: 10.067192291666666 (h)
0: INFO:root:Total running time: 176.47625811894736 (minutes)

# studios-tamani-kalangou-french
100%|██████████| 37332/37332 [00:39<00:00, 946.29it/s] 
0: INFO:root:Total duration of TRAIN: 108.06742366319445 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 2569.76it/s]
0: INFO:root:Total duration of VALID: 2.9512840625 (h)
0: INFO:root:Total running time: 15.661083245277405 (minutes)

# African_Accented_French
0: INFO:root:Number of audio files exist in corpus directory: 16638
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 16402
0: INFO:root:Number of missing files longer than 1s: 67
0: INFO:root:Total duration discarded (files less than 1s): 125.09518750000001 (s)
0: INFO:root:Total duration added back to training data: 3386.0 (s)
100%|██████████| 14313/14313 [00:10<00:00, 1387.57it/s]
0: INFO:root:Total duration of TRAIN: 17.996784496527777 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1703/1703 [00:00<00:00, 3196.41it/s]
0: INFO:root:Total duration of VALID: 1.581388888888889 (h)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 475/475 [00:00<00:00, 3251.07it/s]
0: INFO:root:Total duration of TEST: 0.30371878472222225 (h)
0: INFO:root:Total running time: 3.3377296964327496 (minutes)

# Att-HACK_SLR88
0: INFO:root:Number of audio files exist in corpus directory: 36634
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 36339
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 248.23706250000006 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 36339 - 35339/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 35339/35339 [00:24<00:00, 1427.94it/s]
0: INFO:root:Total duration of TRAIN: 26.315056597222224 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 3103.63it/s]
0: INFO:root:Total duration of VALID: 0.7336313888888889 (h)
0: INFO:root:Total running time: 6.123992125193278 (minutes)

# CaFE
0: INFO:root:Number of audio files exist in corpus directory: 936
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 936
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 936 - 845/91/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 845/845 [00:00<00:00, 1271.01it/s]
0: INFO:root:Total duration of TRAIN: 1.042900329861111 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 91/91 [00:00<00:00, 1962.28it/s]
0: INFO:root:Total duration of VALID: 0.11195144097222223 (h)
0: INFO:root:Total running time: 0.2859409014383952 (minutes)

# CFPP_corrected
0: INFO:root:Number of audio files exist in corpus directory: 12577
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 12577
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:11572 / VALID:999 / TEST:0 / TOTAL:12571
0: INFO:root:Number of audio files split or converted: 76
100%|██████████| 11572/11572 [00:07<00:00, 1566.60it/s]
0: INFO:root:Total duration of TRAIN: 16.465031996527777 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 999/999 [00:00<00:00, 3035.34it/s]
0: INFO:root:Total duration of VALID: 1.4568487847222222 (h)
0: INFO:root:Total running time: 3.176550046602885 (minutes)

# ESLO
0: INFO:root:Number of audio files exist in corpus directory: 62918
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 62918
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:61918 / VALID:1000 / TEST:0 / TOTAL:62918
0: INFO:root:Number of audio files split or converted: 0
100%|██████████| 61918/61918 [00:43<00:00, 1417.93it/s]
0: 3.14it/s]
0: INFO:root:Total duration of TRAIN: 33.61559267361111 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 2847.10it/s]
0: INFO:root:Total duration of VALID: 0.5983778472222222 (h)
0: INFO:root:Total running time: 11.16605964899063 (minutes)

# EPAC_flowbert (883604)
0: INFO:root:Number of audio files exist in corpus directory: 623250
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 623250
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:622250 / VALID:1000 / TEST:0 / TOTAL:623250
0: INFO:root:Number of audio files split or converted: 0
100%|██████████| 500000/500000 [08:31<00:00, 977.47it/s]
100%|██████████| 122250/122250 [02:07<00:00, 960.03it/s]
0: INFO:root:Total duration of TRAIN: 1623.4680903125002 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1082.39it/s]
0: INFO:root:Total duration of VALID: 2.5761444444444446 (h)
0: INFO:root:Total running time: 400.57953464190166 (minutes)

# GEMEP
0: INFO:root:Number of audio files exist in corpus directory: 1260
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 1236
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 19.6 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:1117 / VALID:119 / TEST:0 / TOTAL:1236
0: INFO:root:Number of audio files split or converted: 0
0: INFO:root:Creating zip for TRAIN
 25%|██▍       | 274/1117 [00:02<00:07, 121.71it/s]
 55%|█████▍    | 610/1117 [00:04<00:02, 202.79it/s]
 90%|████████▉ | 1005/1117 [00:05<00:00, 234.57it/]
100%|██████████| 1117/1117 [00:06<00:00, 180.45it/s]
0: INFO:root:Creating zip for VALID
100%|██████████| 119/119 [00:00<00:00, 219.81it/s]
0: INFO:root:Writing manifest file for split:TRAIN
100%|██████████| 1117/1117 [00:00<00:00, 2925.93it/s]
0: INFO:root:Total duration of TRAIN: 0.7634827951388888 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 119/119 [00:00<00:00, 3246.96it/s]
0: INFO:root:Total duration of VALID: 0.08217322916666667 (h)
0: INFO:root:Total running time: 0.26866455872853595 (minutes)

# MPF
0: INFO:root:Number of audio files exist in corpus directory: 40579
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 40579
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:39579 / VALID:1000 / TEST:0 / TOTAL:40579
100%|██████████| 39579/39579 [00:29<00:00, 1356.39it/s]
0: INFO:root:Total duration of TRAIN: 36.121592586805555 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 2820.93it/s]
0: INFO:root:Total duration of VALID: 0.9836671527777778 (h)
0: INFO:root:Total running time: 11.20147408246994 (minutes)

# Portmedia
0: INFO:root:Number of audio files exist in corpus directory: 20264
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 19627
0: INFO:root:Number of missing files longer than 1s: 117
0: INFO:root:Total duration discarded (files less than 1s): 413.3381249999994 (s)
0: INFO:root:Total duration added back to training data: 5146.336187499998 (s)
100%|██████████| 18763/18763 [00:15<00:00, 1226.85it/s]
0: INFO:root:Total duration of TRAIN: 38.56389157986111 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 2951.23it/s]
0: INFO:root:Total duration of VALID: 1.8561921875 (h)
0: INFO:root:Total running time: 5.964762739340464 (minutes)

# TCOF_corrected
0: INFO:root:Number of audio files exist in corpus directory: 84600
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 84592
0: INFO:root:Number of missing files longer than 1s: 3
0: INFO:root:Total duration discarded (files less than 1s): 0.576 (s)
0: INFO:root:Total duration added back to training data: 5.723 (s
100%|██████████| 79681/79681 [00:58<00:00, 1355.66it/s]
0: INFO:root:Total duration of TRAIN: 56.9516740625 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 2842.24it/s]
0: INFO:root:Total duration of VALID: 1.0338196180555557 (h)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 316/316 [00:00<00:00, 3415.16it/s]
0: INFO:root:Total duration of TEST: 0.25328805555555556 (h)
0: INFO:root:Total running time: 17.465069393316906 (minutes)

# MaSS
0: INFO:root:Number of audio files exist in corpus directory: 8219
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 8219
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 8219 - 7384/835/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 7384/7384 [00:03<00:00, 1949.41it/s]
0: INFO:root:Total duration of TRAIN: 17.712009131944445 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 835/835 [00:00<00:00, 2801.67it/s]
0: INFO:root:Total duration of VALID: 1.968730138888889 (h)
0: INFO:root:Total running time: 1.7708897352218629 (minutes)

# NCCFr
0: INFO:root:Number of audio files exist in corpus directory: 29421
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 29421
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:28421 / VALID:1000 / TEST:0 / TOTAL:29421
0: INFO:root:Number of audio files split or converted: 0
100%|██████████| 28421/28421 [00:22<00:00, 1287.28it/s]
0: INFO:root:Total duration of TRAIN: 25.667795104166665 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1237.30it/s]
0: INFO:root:Total duration of VALID: 0.9199747048611111 (h)
0: INFO:root:Total running time: 5.830669637521108 (minutes)

# voxpopuli_unlabeled
0: INFO:root:Number of audio files exist in corpus directory: 570192
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 570192
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TRAIN:569192 / VALID:1000 / TEST:0 / TOTAL:570192
100%|██████████| 500000/500000 [13:43<00:00, 607.29it/s]
100%|██████████| 69192/69192 [01:45<00:00, 656.19it/s]
0: INFO:root:Total duration of TRAIN: 4539.806904982639 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 10001000 [00:01<00:00, 655.97it/s]
0: /1000 [00:01<00:00, 637.85it/s]
0: INFO:root:Total duration of VALID: 7.933169496527778 (h)
0: INFO:root:Total running time: 248.83346354961395 (minutes)

# voxpopuli_transcribed
0: INFO:root:Number of audio files exist in corpus directory: 77030
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 77030
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
100%|██████████| 76037/76037 [01:25<00:00, 885.76it/s] 
0: INFO:root:Total duration of TRAIN: 212.79701890625 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1001/1001 [00:00<00:00, 2840.62it/s]
0: INFO:root:Total duration of VALID: 2.7625467534722223 (h)
0: INFO:root:Total running time: 30.786180357138317 (minutes)

# audiocite_with_metadata
0: INFO:root:Number of audio files exist in corpus directory: 790288
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 789473
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 394.1650000000004 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:Total duration of TRAIN: 6450.733411006944 (h)
0: INFO:root:Writing manifest file for split:VALID
 67%|██████?   | 998/1574 [00:01<00:01, 533.70it/s]
100%|██████████| 1574/1574 [00:02<00:00, 561.32it/s]
0: INFO:root:Total duration of VALID: 12.904680277777778 (h)
0: INFO:root:Writing manifest file for split:TEST
 39%|███▉      | 937/2547 [00:01<00:02, 546.97it/s]
 74%|███████▏  | 1836/2547 [00:03<00:01, 547.10it/s]
100%|██████████| 2547/2547 [00:04<00:00, 544.81it/s]
0: INFO:root:Total duration of TEST: 21.015069444444446 (h)
0: INFO:root:Total running time: 430.7886329849561 (minutes)
```

# Data from INA
100h: /lus/work/CT10/lig3801/vpelloin/dl_100h/tar
```python
import tarfile
import soundfile as sf

with tarfile.open("/lus/work/CT10/lig3801/vpelloin/dl_100h/tar/00a.tar") as archive:
    for f in archive.getmembers():
        flac_file = archive.extractfile(f)
        if flac_file == None:
            continue
        data, sample_rate = sf.read(flac_file)
        print(f"{f} {data.shape} {sample_rate}")
```