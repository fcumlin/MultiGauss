# Multivariate Probabilistic Assessment of Speech Quality

This is the official implementation of "Multivariate Probabilistic Assessment of Speech Quality". Multivariate Probabilistic Assessment of Speech Quality provides a model that takes as input a speech clip, and outputs a multivariate Gaussian distribution over five speech quality dimensions. The dimensions are the overall speech quality (MOS), the intrusivness of the noise (NOI), the coloration quality (COL), the discontinuity quality (DIS), and the loudness (LOUD). All notations follow the definitions given in the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).

Authors: Fredrik Cumlin, Xinyu Liang

Emails: fcumlin@gmail.com, hopeliang990504@gmail.com

## Inference

TODO

## Installation

Installation with pip:
```
pip install -r requirements.txt
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Dataset preparation

[NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus)

## Training
The framework is Gin configurable; specifying model and dataset is done with a Gin config. See examples in `configs/*.gin`.

Example launch:
```
python train.py --gin_path "configs/tot.gin" --save_path "runs/tot"
```
