"""
baseline.py

Classical (non-ML) speech enhancement baselines for comparison:

  1. Spectral Subtraction
     Estimates the noise spectrum from the first few silent frames, then
     subtracts that estimate from every frame. Fast and simple but leaves
     "musical noise" artefacts and fails on non-stationary noise.

  2. Wiener Filter
     Computes the optimal linear filter that minimises the mean-squared
     error between clean and estimated signal.  Better than spectral
     subtraction but still assumes noise is stationary and Gaussian.

Run:
    python src/baseline.py
    
Prints the same metric table as evaluate.py so you can compare directly
"""

import os
import numpy as np
import torch

import config
import utils
from evaluate import (
    compute_pesq, compute_stoi, compute_si_sdr, compute_snr, snr_group,
)


def spectral_subtraction(
    noisy_wav:     torch.Tensor,
    noise_frames:  int   = 10,
    over_subtract: float = 1.0,
    beta:          float = 0.001,
) -> torch.Tensor:
    """
    Classic spectral subtraction denoiser

    Args:
        noisy_wav:     1-D waveform
        noise_frames:  how many initial frames to use for noise estimate
                       (assumes the clip starts with a noise-only segment)
        over_subtract: over-subtraction factor alfa >= 1 (larger -> more aggressive)
        beta:          spectral floor - prevents negative values after subtraction

    Returns:
        enhanced waveform (same length as input)
    """
    magnitude, phase = utils.compute_stft(noisy_wav)   # (F, T)
    power = magnitude ** 2                              # power spectrogram

    # Estimate noise power from the first noise_frames frames
    noise_estimate = power[:, :noise_frames].mean(dim=1, keepdim=True)   # (F, 1)

    # Subtract noise power, apply spectral floor to avoid negative values
    # spectral floor beta ensures we keep at least beta * original power
    enhanced_power = torch.clamp(power - over_subtract * noise_estimate,
                                  min=beta * power)
    enhanced_mag   = torch.sqrt(enhanced_power)

    return utils.reconstruct_waveform(enhanced_mag, phase)


def wiener_filter(
    noisy_wav:    torch.Tensor,
    noise_frames: int   = 10,
    eps:          float = 1e-8,
) -> torch.Tensor:
    """
    Frequency-domain Wiener filter

    The Wiener gain for each frequency bin f is:
        H(f) = SNR(f) / (1 + SNR(f))

    where SNR(f) is estimated as:
        SNR(f) = max(P_noisy(f) - P_noise(f), 0) / P_noise(f)

    When the signal SNR is high, H(f) -> 1 (keep everything)
    When the signal SNR is low,  H(f) -> 0 (suppress everything)

    Args:
        noisy_wav:    1-D waveform
        noise_frames: initial frames used to estimate noise power

    Returns:
        enhanced waveform
    """
    magnitude, phase = utils.compute_stft(noisy_wav)   # (F, T)
    power = magnitude ** 2

    # Estimate noise power from the first few frames (assumed noise-only)
    noise_power = power[:, :noise_frames].mean(dim=1, keepdim=True) + eps

    # Estimate signal power = total power - noise power (floored at 0)
    signal_power = torch.clamp(power - noise_power, min=0.0)

    # Per-bin SNR estimate
    snr_est   = signal_power / noise_power   # (F, T)

    # Wiener gain: H = SNR / (1 + SNR)
    gain = snr_est / (1.0 + snr_est)        # (F, T), values in [0, 1)

    enhanced_mag = gain * magnitude
    return utils.reconstruct_waveform(enhanced_mag, phase)


def evaluate_baseline(method: str = "wiener") -> None:
    """
    Run the chosen classical baseline over the test set and print a metric
    table in the same format as evaluate.py for easy comparison.
    Uses the same .pt test segments as evaluate.py for consistency

    Args:
        method: "spectral_subtraction" or "wiener"
    """
    import glob, random

    test_dir   = os.path.join(config.PROCESSED_DIR, "segments", "test")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.pt")))
    if not test_files:
        raise RuntimeError(f"No .pt files found in {test_dir}")
    random.seed(42)
    if len(test_files) > 1000:
        test_files = random.sample(test_files, 1000)
    print(f"[baseline:{method}] Test segments: {len(test_files)}")

    # Choose the denoiser function
    if method == "wiener":
        denoise_fn = wiener_filter
    elif method == "spectral_subtraction":
        denoise_fn = spectral_subtraction
    else:
        raise ValueError(f"Unknown method: {method}")

    results = {g: {"pesq": [], "stoi": [], "si_sdr": [], "delta_snri": []}
               for g in ["low", "mid", "high", "all"]}

    for pt_path in test_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        mean = data["mean"]
        std  = data["std"]

        # Reconstruct noisy waveform from saved spectrogram
        noisy_spec = utils.denormalise(data["noisy_spec"].squeeze(0), mean, std)
        noisy_mag  = torch.expm1(noisy_spec).clamp(min=0)
        noisy_wav  = utils.reconstruct_waveform(noisy_mag, data["noisy_phase"])

        # Reconstruct clean waveform from saved spectrogram
        clean_spec = utils.denormalise(data["clean_spec"].squeeze(0), mean, std)
        clean_mag  = torch.expm1(clean_spec).clamp(min=0)
        clean_wav  = utils.reconstruct_waveform(clean_mag, data["clean_phase"])

        pred_wav = denoise_fn(noisy_wav)

        min_len  = min(noisy_wav.shape[0], clean_wav.shape[0], pred_wav.shape[0])
        noisy_np = noisy_wav[:min_len].numpy()
        clean_np = clean_wav[:min_len].numpy()
        pred_np  = pred_wav[:min_len].numpy()

        pesq_score   = compute_pesq(clean_np, pred_np)
        stoi_score   = compute_stoi(clean_np, pred_np)
        si_sdr_score = compute_si_sdr(clean_np, pred_np)

        noise_before = noisy_np - clean_np
        noise_after  = pred_np  - clean_np
        snr_before   = compute_snr(clean_np, noise_before)
        snr_after    = compute_snr(clean_np, noise_after)
        delta_snri   = snr_after - snr_before

        group = snr_group(noisy_np, clean_np)
        for g in [group, "all"]:
            results[g]["pesq"].append(pesq_score)
            results[g]["stoi"].append(stoi_score)
            results[g]["si_sdr"].append(si_sdr_score)
            results[g]["delta_snri"].append(delta_snri)

    # Print table
    print(f"\n{'='*60}")
    print(f"Baseline: {method}")
    print(f"{'Group':<8} {'N':>5} {'PESQ':>7} {'STOI':>7} {'SI-SDR':>8} {'ΔSNRi':>7}")
    print("=" * 60)
    for group in ["low", "mid", "high", "all"]:
        data = results[group]
        n    = len(data["pesq"])
        if n == 0:
            continue
        print(
            f"{group:<8} {n:>5} "
            f"{np.nanmean(data['pesq']):>7.3f} "
            f"{np.nanmean(data['stoi']):>7.3f} "
            f"{np.nanmean(data['si_sdr']):>8.2f} "
            f"{np.nanmean(data['delta_snri']):>7.2f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    print("Running both baselines...\n")
    evaluate_baseline("spectral_subtraction")
    print()
    evaluate_baseline("wiener")