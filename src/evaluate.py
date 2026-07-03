"""
evaluate.py

Evaluates the trained model on the test set and prints a results table

Metrics computed:
    PESQ   - Perceptual Evaluation of Speech Quality
    STOI   - Short-Time Objective Intelligibility
    SI-SDR - Scale-Invariant Signal-to-Distortion Ratio
    ΔSNRi  - SNR improvement (SNR_after - SNR_before)

Results are also broken down by SNR group (low / mid / high) to see
how the model performs on easy vs hard cases

Run:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
from pesq  import pesq
from pystoi import stoi

import config
import utils
from model    import UNetDenoiser
from train    import load_checkpoint, _batch_to_waveform


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """
    PESQ score in wideband mode (wb)
    Range: -0.5 to 4.5, higher = better
    """
    try:
        return pesq(sr, ref, deg, "wb")
    except Exception:
        return float("nan")


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """
    STOI intelligibility score
    Range: 0 to 1, higher = more intelligible
    """
    try:
        return stoi(ref, deg, sr, extended=False)
    except Exception:
        return float("nan")


def compute_si_sdr(ref: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SDR in dB on numpy arrays (single sample, not batch)"""
    ref  = ref  - ref.mean()
    pred = pred - pred.mean()
    dot        = np.dot(pred, ref)
    ref_pow    = np.dot(ref, ref) + eps
    s_target   = (dot / ref_pow) * ref
    e_noise    = pred - s_target
    si_sdr_val = 10 * np.log10((s_target ** 2).sum() / ((e_noise ** 2).sum() + eps) + eps)
    return float(si_sdr_val)


def compute_snr(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-8) -> float:
    """SNR in dB: 10*log10(||signal||^2 / ||noise||^2)"""
    return 10 * np.log10((signal ** 2).sum() / ((noise ** 2).sum() + eps) + eps)



@torch.no_grad()
def denoise_waveform(model: UNetDenoiser, noisy_wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Run the full denoising pipeline on a single 1-D waveform

    Steps:
      1. STFT -> magnitude + phase
      2. Log-compress and normalise
      3. Run model -> get mask
      4. Apply mask
      5. Denormalise -> undo log -> iSTFT

    Returns:
        clean_wav: 1-D tensor (float32, 16kHz)
    """
    from dataset import segment_waveform, waveform_to_input

    original_len = noisy_wav.shape[0]
    segments     = segment_waveform(noisy_wav)
    clean_segs   = []

    for seg in segments:
        # Compute STFT directly so we have the raw magnitude for masking
        raw_magnitude, phase = utils.compute_stft(seg)
        compressed           = utils.log_compress(raw_magnitude)
        spec_norm, mean, std = utils.normalise(compressed)

        # Add batch + channel dimensions for the model
        spec_in = spec_norm.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, F, T)
        mask    = model(spec_in).squeeze(0).squeeze(0).cpu()      # (F, T)

        # Apply mask to the RAW magnitude, not the normalised spectrogram
        # The mask M in [0,1] is a gain on the original amplitude values
        clean_magnitude = (mask * raw_magnitude).clamp(min=0)
        wav             = utils.reconstruct_waveform(clean_magnitude, phase)
        clean_segs.append(wav)

    # Overlap-add and trim to original length to remove padding
    result = _overlap_add(clean_segs, config.SEGMENT_SAMPLES, config.OVERLAP)
    return result[:original_len]


def _overlap_add(segments: list[torch.Tensor], seg_len: int, overlap: float) -> torch.Tensor:
    """
    Reconstruct a waveform from overlapping segments by simple overlap-add
    The step between consecutive segments is seg_len * (1 - overlap)
    """
    step   = int(seg_len * (1 - overlap))
    n_segs = len(segments)
    total  = step * (n_segs - 1) + seg_len
    output = torch.zeros(total)
    counts = torch.zeros(total)

    for i, seg in enumerate(segments):
        start = i * step
        end   = start + seg.shape[0]
        output[start:end] += seg
        counts[start:end] += 1

    counts = counts.clamp(min=1)
    return output / counts


def snr_group(noisy: np.ndarray, clean: np.ndarray) -> str:
    """
    Estimate the SNR of a noisy clip and return a group label:
        low  -> SNR < 5 dB   (hard cases)
        mid  -> 5 <= SNR < 15 dB
        high -> SNR >= 15 dB  (easy cases)
    """
    noise = noisy - clean
    snr   = compute_snr(clean, noise)
    if snr < 5:
        return "low"
    elif snr < 15:
        return "mid"
    else:
        return "high"


def evaluate(checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # Load model
    model = UNetDenoiser().to(device)
    epoch, val_si_sdr = load_checkpoint(model, checkpoint_path)
    model.eval()
    print(f"[eval] Loaded checkpoint from epoch {epoch}  (val SI-SDR={val_si_sdr:.2f} dB)")

    # Load .pt test segments (each file contains a paired noisy+clean segment)
    import glob
    test_dir   = os.path.join(config.PROCESSED_DIR, "segments", "test")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.pt")))
    if not test_files:
        raise RuntimeError(f"No .pt files found in {test_dir}")
    print(f"[eval] Test segments: {len(test_files)}")

    # Accumulate metrics per SNR group
    results = {g: {"pesq": [], "stoi": [], "si_sdr": [], "delta_snri": []}
               for g in ["low", "mid", "high", "all"]}

    for pt_path in test_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        # Reconstruct noisy and clean waveforms from the saved spectrograms
        mean = data["mean"]
        std  = data["std"]

        # Noisy waveform: denormalise -> undo log -> iSTFT with noisy phase
        noisy_spec = utils.denormalise(data["noisy_spec"].squeeze(0), mean, std)
        noisy_mag  = torch.expm1(noisy_spec).clamp(min=0)
        noisy_wav  = utils.reconstruct_waveform(noisy_mag, data["noisy_phase"])

        # Clean waveform: denormalise -> undo log -> iSTFT with clean phase
        clean_spec = utils.denormalise(data["clean_spec"].squeeze(0), mean, std)
        clean_mag  = torch.expm1(clean_spec).clamp(min=0)
        clean_wav  = utils.reconstruct_waveform(clean_mag, data["clean_phase"])

        # Denoise the noisy waveform
        pred_wav = denoise_waveform(model, noisy_wav, device)

        # Align lengths
        min_len  = min(noisy_wav.shape[0], clean_wav.shape[0], pred_wav.shape[0])
        noisy_np = noisy_wav[:min_len].numpy()
        clean_np = clean_wav[:min_len].numpy()
        pred_np  = pred_wav[:min_len].numpy()

        # Compute metrics
        stoi_score   = compute_stoi(clean_np, pred_np)
        si_sdr_score = compute_si_sdr(clean_np, pred_np)
        pesq_after   = compute_pesq(clean_np, pred_np)

        noise_before = noisy_np - clean_np
        noise_after  = pred_np  - clean_np
        snr_before   = compute_snr(clean_np, noise_before)
        snr_after    = compute_snr(clean_np, noise_after)
        delta_snri   = snr_after - snr_before

        group = snr_group(noisy_np, clean_np)

        for g in [group, "all"]:
            results[g]["pesq"].append(pesq_after)
            results[g]["stoi"].append(stoi_score)
            results[g]["si_sdr"].append(si_sdr_score)
            results[g]["delta_snri"].append(delta_snri)

    # Print results table
    print("\n" + "=" * 60)
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
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement model")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint)