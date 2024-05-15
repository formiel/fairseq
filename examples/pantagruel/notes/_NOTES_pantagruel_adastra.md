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
#### FINAL: 10% but not exceed 1k valid samples for each dataset
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
100%|██████████| 11619/11619 [00:07<00:00, 1462.03it/s]
0: INFO:root:Total duration of TRAIN: 16.466835069444443 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 3016.88it/s]
0: INFO:root:Total duration of VALID: 1.4568921180555556 (h)
0: INFO:root:Total running time: 3.070813504854838 (minutes)

# ESLO
0: INFO:root:Number of audio files exist in corpus directory: 62918
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 62918
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 62918 - 61918/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 61918/61918 [00:44<00:00, 1396.07it/s]
0: INFO:root:Total duration of TRAIN: 33.61559267361111 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 3060.46it/s]
0: INFO:root:Total duration of VALID: 0.5983778472222222 (h)
0: INFO:root:Total running time: 9.256109754244486 (minutes)

# EPAC_flowbert
0: INFO:root:Number of audio files exist in corpus directory: 623250
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 623250
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 623250 - 622250/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 500000/500000 [08:33<00:00, 973.05it/s]
0: ??| 122250/122250 [02:07<00:00, 956.58it/s]
0: INFO:root:Total duration of TRAIN: 1623.4680903125002 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1080.03it/s]
0: INFO:root:Total duration of VALID: 2.5761444444444446 (h)
0: INFO:root:Total running time: 399.1273416598638 (minutes)

# GEMEP
0: INFO:root:Number of audio files exist in corpus directory: 1260
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 1236
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 19.6 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 1236 - 1117/119/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Creating zip for VALID
100%|██████████| 119/119 [00:00<00:00, 207.20it/s]
0: INFO:root:Writing manifest file for split:TRAIN
100%|██████████| 1117/1117 [00:00<00:00, 3069.73it/s]
0: INFO:root:Total duration of TRAIN: 0.7634827951388888 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 119/119 [00:00<00:00, 3587.71it/s]

# MPF
0: INFO:root:Number of audio files exist in corpus directory: 40579
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 40579
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 40579 - 39579/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 39579/39579 [00:31<00:00, 1245.78it/s]
0: INFO:root:Total duration of TRAIN: 36.121592586805555 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1248.06it/s]
0: INFO:root:Total duration of VALID: 0.9836671527777778 (h)
0: INFO:root:Total running time: 9.440574721495311 (minutes)

# Portmedia
0: INFO:root:Number of audio files exist in corpus directory: 20264
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 19627
0: INFO:root:Number of missing files longer than 1s: 117
0: INFO:root:Total duration discarded (files less than 1s): 413.3381249999994 (s)
0: INFO:root:Total duration added back to training data: 5146.336187499998 (s)
100%|██████████| 18763/18763 [00:16<00:00, 1142.94it/s]
0: INFO:root:Total duration of TRAIN: 38.56389157986111 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1256.15it/s]
0: INFO:root:Total duration of VALID: 1.8561921875 (h)
0: INFO:root:Total running time: 5.189605216185252 (minutes)

# TCOF_corrected
0: INFO:root:Number of audio files exist in corpus directory: 84600
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 84592
0: INFO:root:Number of missing files longer than 1s: 3
0: INFO:root:Total duration discarded (files less than 1s): 0.576 (s)
0: INFO:root:Total duration added back to training data: 5.723 (s)
100%|██████████| 83343/83343 [01:01<00:00, 1349.29it/s]
0: INFO:root:Total duration of TRAIN: 57.057041388888884 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1230.79it/s]
0: INFO:root:Total duration of VALID: 1.0338196180555557 (h)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 350/350 [00:00<00:00, 1722.94it/s]
0: INFO:root:Total duration of TEST: 0.2541852777777778 (h)
0: INFO:root:Total running time: 13.692967696984608 (minutes)

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
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 29421 - 28421/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 28421/28421 [00:21<00:00, 1341.11it/s]
0: INFO:root:Total duration of TRAIN: 25.667795104166665 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1000/1000 [00:00<00:00, 1365.46it/s]
0: INFO:root:Total duration of VALID: 0.9199747048611111 (h)
0: INFO:root:Total running time: 4.318031605084737 (minutes)

# voxpopuli_unlabeled
0: INFO:root:Number of audio files exist in corpus directory: 570192
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 570192
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 570192 - 569192/1000/0
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 500000/500000 [14:47<00:00, 563.66it/s]
100%|██████████| 69192/69192 [02:04<00:00, 556.18it/s]
0: INFO:root:Total duration of TRAIN: 4539.806904982639 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 945/1000 [00:01<00:00, 575.96it/s]
0: | 1000/1000 [00:01<00:00, 599.26it/s]
0: INFO:root:Total duration of VALID: 7.933169496527778 (h)
0: INFO:root:Total running time: 232.07447178761166 (minutes)

# voxpopuli_transcribed
0: INFO:root:Number of audio files exist in corpus directory: 77030
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 77030
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
100%|██████████| 76037/76037 [01:21<00:00, 934.45it/s]
0: INFO:root:Total duration of TRAIN: 212.79701890625 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1001/1001 [00:00<00:00, 1996.84it/s]
0: INFO:root:Total duration of VALID: 2.7625467534722223 (h)
0: INFO:root:Total running time: 27.699066630999248 (minutes)

# audiocite_with_metadata
0: INFO:root:Number of audio files exist in corpus directory: 790288
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 789473
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 394.1650000000004 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 789473 - 785352/1574/2547
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 500000/500000 [15:00<00:00, 555.29it/s]
100%|██████████| 285352/285352 [08:52<00:00, 535.64it/s]
0: INFO:root:Total duration of TRAIN: 6452.5325479513895 (h)
0: INFO:root:Writing manifest file for split:VALID
 67%|██████?   | 992/1574 [00:01<00:01, 535.19it/s]
100%|██████████| 1574/1574 [00:02<00:00, 566.34it/s]
0: INFO:root:Total duration of VALID: 12.904680277777778 (h)
0: INFO:root:Writing manifest file for split:TEST
 37%|███▋      | 899/2547 [00:01<00:03, 519.51it/s]
 71%|████?█▉   | 1755/2547 [00:03<00:01, 523.28it/s]
100%|██████████| 2547/2547 [00:04<00:00, 529.11it/s]
0: 4<00:00, 525.07it/s]
0: INFO:root:Total duration of TEST: 21.015069444444446 (h)
0: INFO:root:Total running time: 396.8964731931686 (minutes)


# /lus/scratch/CT10/c1615074/tphle/Data/LeBenchmark_prepared_not_split_valid (no audiocite dataset)
# mls_french_jz (number examples matched with paper)
# 876650 -> 876665
0: INFO:root:Number of audio files exist in corpus directory: 263055
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 263055 - 258213/2416/2426
0: INFO:root:Number of audio files split additionally: 0
100%|██████████| 258213/258213 [03:59<00:00, 1077.44it/s]
0: INFO:root:Total duration of TRAIN: 1076.5805241493056 (h)
0: INFO:root:Writing manifest file for split:VALID
 74%|███████▍  | 1787/2416 [00:01<00:00, 1129.94it/s]
100%|██████████| 2416/2416 [00:02<00:00, 1118.77it/s]
0: INFO:root:Total duration of VALID: 10.071229444444445 (h)
0: INFO:root:Writing manifest file for split:TEST
 75%|███████▍  | 1811/2426 [00:01<00:00, 1130.09it/s]
100%|██████████| 2426/2426 [00:02<00:00, 1126.48it/s]
0: INFO:root:Total duration of TEST: 10.067192291666666 (h)
0: INFO:root:Total running time: 171.5478678703308 (minutes)

# studios-tamani-kalangou-french (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 38332
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 38332
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 38332 - 38332/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 111.01870772569445 (h)
0: INFO:root:Total running time: 18.338825849692025 (minutes)

# African_Accented_French (in paper: 16402 vs. 16491: add back ~1h of training data)
0: INFO:root:Number of audio files exist in corpus directory: 16638
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 16402
0: INFO:root:Number of missing files longer than 1s: 67
0: INFO:root:Total duration discarded (files less than 1s): 125.09518750000001 (s)
0: INFO:root:Total duration added back to training data: 3386.0 (s)
0: INFO:root:num_file_splits: 41
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 16491 - 14313/1703/475
0: INFO:root:Number of audio files split additionally: 41
0: INFO:root:Total duration of TRAIN: 17.996784496527777 (h)
0: INFO:root:Writing manifest file for split:VALID
100%|██████████| 1703/1703 [00:00<00:00, 2927.83it/s]
0: INFO:root:Total duration of VALID: 1.581388888888889 (h)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 475/475 [00:00<00:00, 2992.27it/s]
0: INFO:root:Total duration of TEST: 0.30371878472222225 (h)
0: INFO:root:Total running time: 4.1673245867093405 (minutes)

# Att-HACK_SLR88 (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 36634
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 36339
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 248.23706250000006 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 36339 - 36339/0/0
0: INFO:root:Total duration of TRAIN: 27.048687986111112 (h)
0: INFO:root:Total running time: 7.941949470837911 (minutes)

# CaFE (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 936
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 936
0: INFO:root:Total duration of TRAIN: 1.1548517708333332 (h)
0: INFO:root:Total running time: 0.2929395039876302 (minutes)

# CFPP_corrected (paper: 9853, different from json)
0: INFO:root:Number of audio files exist in corpus directory: 12577
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 12577
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 76
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 12619 - 12619/0/0
0: INFO:root:Number of audio files split additionally: 76
0: INFO:root:Total duration of TRAIN: 17.9237271875 (h)
0: INFO:root:Total running time: 3.88174045085907 (minutes)

# ESLO (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 62918
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 62918
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 62918 - 62918/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 34.213970520833335 (h)
0: INFO:root:Total running time: 12.780905425548553 (minutes)

# EPAC_flowbert (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 623250
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 623250
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:Total duration of TRAIN: 1626.0442347569444 (h)
0: INFO:root:Total running time: 451.64963444471357 (minutes)

# GEMEP (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 1260
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 1236
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 19.6 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 1236 - 1236/0/0
0: INFO:root:Total duration of TRAIN: 0.8456560243055555 (h)
0: INFO:root:Total running time: 0.33327593803405764 (minutes)

# MPF
0: INFO:root:Number of audio files exist in corpus directory: 40579
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 40579
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 40579 - 40579/0/0
0: INFO:root:Total duration of TRAIN: 37.10525973958334 (h)
0: INFO:root:Total running time: 12.686909588177999 (minutes)

# Portmedia (paper: 19627)
0: INFO:root:Number of audio files exist in corpus directory: 20264
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 19627
0: INFO:root:Number of missing files longer than 1s: 117
0: INFO:root:Total duration discarded (files less than 1s): 413.3381249999994 (s)
0: INFO:root:Total duration added back to training data: 5146.336187499998 (s)
0: INFO:root:num_file_splits: 31
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 19763 - 19763/0/0
0: INFO:root:Number of audio files split additionally: 31
0: INFO:root:Total duration of TRAIN: 40.42008376736111 (h)
0: INFO:root:Total running time: 6.452608346939087 (minutes)

# TCOF_corrected (paper: 58722)
0: INFO:root:Number of audio files exist in corpus directory: 84600
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 84592
0: INFO:root:Number of missing files longer than 1s: 3
0: INFO:root:Total duration discarded (files less than 1s): 0.576 (s)
0: INFO:root:Total duration added back to training data: 5.723 (s)
0: INFO:root:num_file_splits: 103
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 84693 - 84343/0/350
0: INFO:root:Number of audio files split additionally: 103
0: INFO:root:Total duration of TRAIN: 58.09086100694444 (h)
0: INFO:root:Writing manifest file for split:TEST
100%|██████████| 350/350 [00:00<00:00, 1640.37it/s]
0: INFO:root:Total duration of TEST: 0.2541852777777778 (h)
0: INFO:root:Total running time: 21.056022123495737 (minutes)

# MaSS (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 8219
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 8219
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 8219 - 8219/0/0
0: INFO:root:Total duration of TRAIN: 19.68073927083333 (h)
0: INFO:root:Total running time: 2.5202423850695292 (minutes)

# NCCFr (number examples matched with paper)
0: INFO:root:Number of audio files exist in corpus directory: 29421
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 29421
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 29421 - 29421/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 26.587769809027776 (h)
0: INFO:root:Total running time: 6.436931733290354 (minutes)

# voxpopuli_unlabeled (paper 568,338)
0: INFO:root:Number of audio files exist in corpus directory: 570192
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 570192
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 0
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 570192 - 570192/0/0
0: INFO:root:Number of audio files split additionally: 0
0: INFO:root:Total duration of TRAIN: 4547.740074479167 (h)
0: INFO:root:Total running time: 291.8596861680349 (minutes)

# voxpopuli_transcribed (paper 76.281)
0: INFO:root:Number of audio files exist in corpus directory: 77030
0: INFO:root:Getting paths from json...
0: INFO:root:Number of valid audio files (that exist) read from JSON: 77030
0: INFO:root:Number of missing files longer than 1s: 0
0: INFO:root:Total duration discarded (files less than 1s): 0 (s)
0: INFO:root:Total duration added back to training data: 0 (s)
0: INFO:root:num_file_splits: 15
0: INFO:root:TOTAL - TRAIN/VALID/TEST: 77038 - 77038/0/0
0: INFO:root:Number of audio files split additionally: 15
0: INFO:root:Total duration of TRAIN: 215.55956565972224 (h)
0: INFO:root:Total running time: 35.14710802237193 (minutes)

# audiocite_with_metadata
# job 0: 880241
```
