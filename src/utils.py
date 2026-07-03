"""
utils.py

Small helper functions shared across the whole project:
  - loading and saving WAV files
  - STFT / iSTFT wrappers
  - mask application
  - directory setup
"""

import os
import math
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config


def load_audio(path: str) -> torch.Tensor:
    """
    Load a WAV file and return a 1-D float32 tensor (mono, 16kHz)

    If the file has multiple channels it is mixed down to mono
    If the sample rate differs from config.SAMPLE_RATE it is resampled

    Returns:
        waveform: shape (num_samples,)
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (num_samples, num_channels)

    # Mix down to mono if needed
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)

    data = data[:, 0]  # (num_samples,)

    # Resample if needed
    if sr != config.SAMPLE_RATE:
        gcd = math.gcd(sr, config.SAMPLE_RATE)
        up, down = config.SAMPLE_RATE // gcd, sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    return torch.from_numpy(data)   # (num_samples,)


def save_audio(waveform: torch.Tensor, path: str) -> None:
    """
    Save a 1-D float32 tensor as a 16kHz mono WAV file

    Args:
        waveform: shape (num_samples,)
        path:     destination file path (parent directory must exist)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, waveform.numpy(), config.SAMPLE_RATE)


def compute_stft(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Short-Time Fourier Transform of a waveform

    Returns the magnitude and phase separately so the model only
    needs to process the magnitude while the phase is kept for
    reconstruction later

    Args:
        waveform: shape (num_samples,)

    Returns:
        magnitude: shape (n_freq_bins, n_frames)  - non-negative real values
        phase:     shape (n_freq_bins, n_frames)  - values in [-pi, pi]
    """
    window = torch.hann_window(config.WIN_LENGTH, device=waveform.device)

    # stft returns complex tensor of shape (n_freq_bins, n_frames)
    stft_complex = torch.stft(
        waveform,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
        return_complex=True,
    )

    magnitude = stft_complex.abs()           # |z|
    phase     = torch.angle(stft_complex)    # angle of z in radians

    return magnitude, phase


def reconstruct_waveform(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a waveform from magnitude and phase via inverse STFT (iSTFT)

    The phase is the original (unmodified) phase saved during the forward STFT,
    which avoids phase distortion artifacts

    Args:
        magnitude: shape (n_freq_bins, n_frames)
        phase:     shape (n_freq_bins, n_frames)

    Returns:
        waveform: shape (num_samples,)
    """
    # Rebuild complex spectrogram: z = |z| * e^(i*angle(z))
    stft_complex = magnitude * torch.exp(1j * phase)

    window = torch.hann_window(config.WIN_LENGTH, device=magnitude.device)

    waveform = torch.istft(
        stft_complex,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
    )

    return waveform


def apply_mask(noisy_magnitude: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply the soft mask predicted by the model to the noisy magnitude

    The mask M ∈ [0, 1] acts as a per-bin gain:
        clean_estimate = M * noisy_magnitude

    Values close to 1 -> keep (likely speech)
    Values close to 0 -> suppress (likely noise)

    Args:
        noisy_magnitude: shape (batch, 1, n_freq_bins, n_frames)
        mask:            shape (batch, 1, n_freq_bins, n_frames), values in [0, 1]

    Returns:
        clean_magnitude: shape (batch, 1, n_freq_bins, n_frames)
    """
    return mask * noisy_magnitude


def log_compress(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply log(1 + magnitude) compression to reduce the dynamic range

    Without this, very loud components dominate and quiet speech details
    are effectively invisible to the network

    Args:
        magnitude: any shape, non-negative
        eps:       small constant to avoid log(0)

    Returns:
        compressed magnitude, same shape
    """
    return torch.log1p(magnitude + eps)


def normalise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z-score normalise a tensor along all dimensions

    Subtracting the mean and dividing by the standard deviation brings
    inputs to a roughly zero-mean, unit-variance range which helps
    gradient-based optimisation converge faster

    Returns:
        x_norm: normalised tensor
        mean:   scalar mean (saved so you can undo normalisation later)
        std:    scalar std
    """
    mean = x.mean()
    std  = x.std().clamp(min=1e-8)   # avoid division by zero on silent clips
    return (x - mean) / std, mean, std


def normalise_with_stats(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Normalise a tensor using externally provided mean and std

    Args:
        x:    tensor to normalise (any shape)
        mean: scalar mean from the reference signal (noisy)
        std:  scalar std from the reference signal (noisy)

    Returns:
        x_norm: normalised tensor, same shape as x
    """
    std_clamped = std.clamp(min=1e-8)
    return (x - mean) / std_clamped


def denormalise(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo z-score normalisation"""
    return x_norm * std + mean


def ensure_dirs() -> None:
    """Create all project output directories if they do not already exist"""
    for d in [config.PROCESSED_DIR, config.CHECKPOINT_DIR, config.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)