r"""Generates SSL features of audio.

Given a directory of audio files, this binary extracts features using a
specified self-supervised learning (SSL) model. The audio files are first
cropped (or repetitively padded) to a target duration. Then, features are
extracted from specified layers of the model and saved as NumPy arrays in a
feature directory. The binary supports various SSL models provided by
torchaudio.

Given the base wav directory, and assume an audio file is named `audio.wav`,
the path to the audio file is `{wav_dir}/audio.wav`, and the corresponding
feature file will be saved as
`{wav_dir}_feature_{model_name}_layer{i}/audio.npy` for each layer `i` specified
in `layers`.

Example usage:
```
python generate_ssl_features.py \
    --model_name "w2v2_xlsr_2b" \
    --wav_dir "/path/to/wav_files" \
    --layers 11 12 13 \
    --target_duration 8.0
```
"""

import argparse
import os
from typing import Sequence

import numpy as np
import torch
import torchaudio
import tqdm

import audio


def preprocess_data(
    model_name: str,
    target_duration: int,
    wav_dir: str,
    layers_to_include: Sequence[int]
) -> None:
    """Preprocesses audio data to extract SSL features.
    
    Given a directory of audio files, this function extracts features using a
    specified self-supervised learning (SSL) model. The audio files are
    resampled to a target sample rate and cropped to a target duration. Features
    are extracted from specified layers of the model and saved as NumPy arrays
    in a designated feature directory. The function supports various SSL models
    provided by torchaudio.
    
    Given the base wav directory, and assume an audio file is named `audio.wav`,
    the path to the audio file is `{wav_dir}/audio.wav`, and the
    corresponding feature file will be saved as
    `{wav_dir}_feature_{model_name}_layer{i}/audio.npy` for each layer `i`
    specified in `layers_to_include`.
    
    Args:
        model_name: Name of the SSL model to use.
        target_duration: Target duration of the audio clips in seconds.
        wav_dir: Directory containing the audio files.
        layers_to_include: List of layer indices to extract features from.
    """
    sample_rate = 16_000  # Default sample rate for the SSL models.
    target_length = int(sample_rate * target_duration)
    base_feature_path = f'{wav_dir}_feature_{model_name}'

    wav_paths = [path for path in os.listdir(wav_dir) if '.wav' in path]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'w2v2_xlsr_300m':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        model = bundle.get_model().to(device=device)
    elif model_name == 'w2v2_xlsr_1b':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
        model = bundle.get_model().to(device=device)
    elif model_name == 'w2v2_xlsr_2b':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
        model = bundle.get_model().to(device=device)
    elif model_name == 'hubert_xlarge':
        bundle = torchaudio.pipelines.HUBERT_XLARGE
        model = bundle.get_model().to(device=device)
    elif model_name == 'wavlm_large':
        bundle = torchaudio.pipelines.WAVLM_LARGE
        model = bundle.get_model().to(device=device)
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    for wav_path in tqdm.tqdm(wav_paths):
        full_wav_path = os.path.join(wav_dir, wav_path)
        wav_name = wav_path[:-4]
        audio_object = audio.Audio.read_wav(full_wav_path).resample(sample_rate)
        audio_object = audio_object.repetitive_crop(target_length)

        with torch.inference_mode():
            features, _ = model.extract_features(
                torch.Tensor(audio_object.samples).to(device=device)
            )
           
        for i in layers_to_include:
            features[i] = features[i].squeeze().T
            folder_path_i = f'{base_feature_path}_layer{i}'
            if not os.path.exists(folder_path_i):
                os.makedirs(folder_path_i)
            full_feature_path_i = os.path.join(folder_path_i, f'{wav_name}.npy')
            np.save(full_feature_path_i, features[i].cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description='Model and target length')
    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the SSL model to use.',
        required=True,
    )
    parser.add_argument(
        '--wav_dir',
        type=str,
        help='Path to the directory containing audio files.',
        required=True,
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[11],
        help='Layers to extract features from.',
    )
    parser.add_argument(
        '--target_duration',
        type=float,
        default=8.0,
        help='Target duration of the clips, in seconds.',
    )
    args = parser.parse_args()

    preprocess_data(
        model_name=args.model_name,
        target_duration=args.target_duration,
        wav_dir=args.wav_dir,
        layers_to_include=args.layers,
    )


if __name__ == '__main__':
    main()
