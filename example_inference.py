import argparse

import torch
import torchaudio


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


def main():
    argparser = argparse.ArgumentParser(
        description="Test inference with a pre-trained Wav2Vec2 model."
    )
    argparser.add_argument(
        "--wav_path",
        type=str,
        required=True,
        help="Path to the WAV file to be processed.",
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MultiGauss model.",
    )
    argparser.add_argument(
        "--ssl_model_layer",
        type=int,
        default=11,
        help="The layer of the SSL model to extract the feature from.",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    args = argparser.parse_args()
    device = torch.device(args.device)

    # Read and preprocess the WAV file.
    waveform, sample_rate = _read_wav(args.wav_path)
    waveform = _optionally_resample_audio(
        waveform, sample_rate, target_sample_rate=16_000
    )
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo.
    waveform = waveform.to(device=device)
    
    # Process the waveform with SSL model.
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    ssl_model = bundle.get_model().to(device=device)
    features, _ = ssl_model.extract_features(waveform)
    feature = features[args.ssl_model_layer].squeeze().T
    
    # Load the MultiGauss model and perform inference.
    multigauss_model = torch.jit.load(
        args.model,
        map_location=device,
    )
    multigauss_model.eval()
    with torch.no_grad():
        feature = feature.unsqueeze(0)  # Add batch dimension.
        mean_prediction, covariance_prediction = multigauss_model(feature)
    
    print("Mean Prediction:", mean_prediction)
    print("Covariance Prediction:", covariance_prediction)


if __name__ == "__main__":
    main()
