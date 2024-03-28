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

rsyncpass -zarvm --exclude=".git*" /Users/hang/Downloads/adastra/*.tsv \
                                    adastra:/lus/work/CT10/c1615074/tphle/Data/prepared/MLS_French/


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

````bash
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
                                    
