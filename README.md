# SVP-T
This is the offical code of paper "SVP-T: A Shape-Level Variable-Position Transformer for Multivariate Time Series Classification".
## Setup
```bash
pip install -r requirements.txt
```


We use the UEA Download the datasets, download the convert data from https://drive.google.com/file/d/1wjdRYe8SjntCy7VekfGKBoDx24TF2RKz/view
One dataset BasicMotions is provided for running an example.

Put raw data in dir:
```bash
data/raw/
```
Put the preprocessed data in dir:
```bash
data/preprocess/
```

install requirement:
```bash
pip install -r requirements.txt
```

prepare experiments directory:
```bash
mkdir src/experiments/exp_1/checkpoints
mkdir src/experiments/exp_1/predictions
```


## Run an example
To run an example:
```bash
python src/main.py --data_path data/raw/ --dataset BasicMotions
```
See the result in:
```bash
vi new.log
```
## Train and test
You can set the parameters and exp dir in
```bash
vi src/setting.py
```

To train a dataset with name "dataset_name"
```bash
python src/main.py --data_path data/raw/  --dataset dataset_name
```


