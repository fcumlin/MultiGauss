# Multivariate Probabilistic Assessment of Speech Quality

This is the official implementation of "Multivariate Probabilistic Assessment of Speech Quality". Multivariate Probabilistic Assessment of Speech Quality provides a model that takes as input a speech clip, and outputs a multivariate Gaussian distribution over five speech quality dimensions. The dimensions are the overall speech quality (MOS), the intrusivness of the noise (NOI), the coloration quality (COL), the discontinuity quality (DIS), and the loudness (LOUD). All notations follow the definitions given in the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).

Authors: Fredrik Cumlin, Xinyu Liang  
Emails: fcumlin@gmail.com, hopeliang990504@gmail.com

## Inference

Please see example in `example_inference.py`. This script can be used on single wav files, e.g.:
```
python example_inference.py --wav_path 'path/to/audio_to_be_processed.wav' --model runs/multigauss/model.pt
```

## Installation

See `requirements.txt`.

## Dataset preparation

1. Download the NISQA dataset: [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus)

2. Run the preprocessing script `preprocess/generate_ssl_features.sh` to preprocess the NISQA datasets with wav2vec. Example:
```
./preprocess/generate_ssl_features.sh 'path/to/NISQA_Corpus'
```

Please note that the default configuration will only save the features from the 12th layer (index 11). Hence, other layers specified in the Gin configuration during training will not work. 

## Training
The framework is Gin configurable; specifying model and dataset is done with a Gin config. See examples in `configs/*.gin`.

Example launch:
```
python train.py --gin_path "configs/tot.gin" --save_path "runs/tot"
```
