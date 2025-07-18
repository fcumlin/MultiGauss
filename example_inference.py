r"""Script for running MultiGauss inference on single wav files.

The model operates at 16 kHz sample rate and on signals of 10 s duration, hence,
all audio is resampled to 16 kHz and repeated or cropped to 10 s before
processing. Note that the sample rate implies that no energy with frequencies
above 8 kHz are seen by the model.

Example run:
```
python example_inference.py --wav_path 'path/to/audio_to_be_processed.wav'
```
"""

import argparse

import torch
import torchaudio

import model as model_lib


def _read_wav(file_path: str)-> tuple[torch.Tensor, int]:
    """Reads a WAV file and returns the waveform and sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def _optionally_resample_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16_000
) -> torch.Tensor:
    """Resamples the audio waveform to the target sample rate if necessary."""
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )(waveform)
    return waveform


def _repeat_and_crop_to_length(
    waveform: torch.Tensor,
    target_length: int = 160_000,
) -> torch.Tensor:
    """Repeates or crops the waveform to give it the target length."""
    current_length = waveform.shape[-1]
    if current_length < target_length:
        num_repeats = target_length // current_length + 1
        waveform = waveform.repeat(1, num_repeats)
    return waveform[:, :target_length]


def main():
    parser = argparse.ArgumentParser(
        description="Test inference with a pre-trained Wav2Vec2 model."
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        required=True,
        help="Path to the WAV file to be processed.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/probabilistic/model_best_state_dict.pt",
        help="Path to MultiGauss model.",
    )
    parser.add_argument(
        "--ssl_model_layer",
        type=int,
        default=11,
        help="The layer of the SSL model to extract the feature from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # Read and preprocess the WAV file.
    waveform, sample_rate = _read_wav(args.wav_path)
    waveform = _optionally_resample_audio(
        waveform, sample_rate, target_sample_rate=16_000
    )
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo.
    waveform = _repeat_and_crop_to_length(
        waveform,
        target_length=160_000,  # Training was done with 10 s of audio (16 kHz).
    )
    waveform = waveform.to(device=device)
    
    # Process the waveform with SSL model.
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    ssl_model = bundle.get_model().to(device=device)
    ssl_model.eval()
    with torch.no_grad():
        features, _ = ssl_model.extract_features(waveform)
        feature = features[args.ssl_model_layer].squeeze().T
    
    # Load the MultiGauss model and perform inference.
    multigauss_model = model_lib.ProjectionHead(in_shape=feature.shape)
    state_dict = torch.load(
        args.model_path,
        map_location=device,
        weights_only=True
    )
    multigauss_model.load_state_dict(state_dict)
    multigauss_model.eval()
    with torch.no_grad():
        feature = feature.unsqueeze(0)  # Add batch dimension.
        mean_prediction, covariance_prediction = multigauss_model(feature)
    
    print("Mean Prediction:", mean_prediction)
    print("Covariance Prediction:", covariance_prediction)


if __name__ == "__main__":
    main()
