# Install

```
- Create anaconda env

conda create -n ped
conda activate ped
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

OR

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
for newer GPUs
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt

- install DCNv2

git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh

- install zed sdk python api for the conda env
python /usr/local/zed/get_python_api.py

- Donwgrade numpy
  pip install --upgrade numpy==1.21

```

# Paper

```
- install latex env for vs code
sudo apt install texlive-latex-extra texlive-publishers texlive-science latexmk biber
sudo apt install texlive-full
```

# SDOF-Tracker

![Tracking example](https://github.com/hitottiez/sdof-tracker/blob/main/docs/000180.jpg)

```

```
