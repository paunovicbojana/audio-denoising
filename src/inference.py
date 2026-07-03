"""
inference.py

Denoise a single WAV file using the trained model

Usage:
    python src/inference.py --input path/to/noisy.wav --output path/to/clean.wav
    python src/inference.py --input path/to/noisy.wav   # saves to outputs/ folder

The script loads the best checkpoint, runs the full pipeline, and writes
the denoised WAV to disk
"""

import argparse
import os

import torch
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

import config
import utils
from model    import UNetDenoiser
from train    import load_checkpoint
from evaluate import denoise_waveform


def plot_spectrograms(
    noisy_wav:   torch.Tensor,
    clean_wav:   torch.Tensor,
    output_path: str,
    sr:          int = config.SAMPLE_RATE,
) -> None:
    """
    Save a side-by-side spectrogram comparison (noisy vs denoised) as a PNG

    Uses mel-scale spectrograms (log amplitude) which match human hearing
    better than linear frequency spectrograms

    Args:
        noisy_wav:   original zashumljeni signal (1-D tensor)
        clean_wav:   denoised signal (1-D tensor)
        output_path: path to the output WAV - PNG is saved alongside it
    """
    noisy_np = noisy_wav.numpy()
    clean_np = clean_wav[:noisy_wav.shape[0]].numpy()   # align lengths

    # Mel spectrogram parameters
    n_fft   = config.N_FFT
    hop     = config.HOP_LENGTH
    n_mels  = 128

    mel_noisy = librosa.feature.melspectrogram(
        y=noisy_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    mel_clean = librosa.feature.melspectrogram(
        y=clean_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )

    # Convert to dB scale
    mel_noisy_db = librosa.power_to_db(mel_noisy, ref=np.max)
    mel_clean_db = librosa.power_to_db(mel_clean, ref=np.max)

    # Use the same color scale for both panels so they are comparable
    vmin = min(mel_noisy_db.min(), mel_clean_db.min())
    vmax = max(mel_noisy_db.max(), mel_clean_db.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    for ax, mel_db, title in zip(
        axes,
        [mel_noisy_db, mel_clean_db],
        ["Input (Noisy Signal)", "Output (Denoised Signal)"],
    ):
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=hop,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Mel)")

    fig.colorbar(img, ax=axes, format="%+2.0f dB", label="Amplitude (dB)")
    fig.suptitle(
        "Spectrogram Comparison Before and After Denoising",
        fontsize=14,
        fontweight="bold",
    )

    png_path = os.path.splitext(output_path)[0] + "_spectrogram.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inference] Spectrogram saved  -> {png_path}")


def postprocess(wav: torch.Tensor, reference: torch.Tensor, sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    Post-processing for inference (NOT used during evaluation so metrics are unaffected):
      1. Bandpass filter 80-7500 Hz to emphasize speech frequencies
      2. Peak normalization to match input loudness
    """
    wav_np = wav.numpy().astype(np.float64)
    b, a   = butter(2, 80   / (sr / 2), btype='high')
    wav_np = filtfilt(b, a, wav_np)
    b, a   = butter(2, 7500 / (sr / 2), btype='low')
    wav_np = filtfilt(b, a, wav_np)
    result = torch.from_numpy(wav_np.astype(np.float32))

    input_peak  = reference.abs().max().clamp(min=1e-8)
    output_peak = result.abs().max().clamp(min=1e-8)
    return result * (input_peak / output_peak)


def run_inference(input_path: str, output_path: str, checkpoint_path: str) -> None:
    """
    Load model + audio, denoise, save result and spectrogram comparison

    Args:
        input_path:      path to the noisy input WAV
        output_path:     where to write the denoised WAV
        checkpoint_path: path to model checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")

    # Load model
    model = UNetDenoiser().to(device)
    epoch, _ = load_checkpoint(model, checkpoint_path)
    model.eval()
    print(f"[inference] Loaded model from epoch {epoch}")

    # Load audio
    noisy_wav = utils.load_audio(input_path)
    duration  = noisy_wav.shape[0] / config.SAMPLE_RATE
    print(f"[inference] Input: {input_path}  ({duration:.2f}s)")

    # Denoise
    with torch.no_grad():
        clean_wav = denoise_waveform(model, noisy_wav, device)

    # Post-process: bandpass + loudness normalization
    clean_wav = postprocess(clean_wav, noisy_wav)

    # Save result
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    utils.save_audio(clean_wav, output_path)
    print(f"[inference] Saved denoised audio  -> {output_path}")

    # Save spectrogram comparison
    plot_spectrograms(noisy_wav, clean_wav, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a WAV file or a random batch")
    parser.add_argument("--input",  required=False, help="Path to noisy input WAV")
    parser.add_argument(
        "--output", required=False,
        help="Path for denoised output WAV (default: outputs/<input_filename>)",
    )
    parser.add_argument(
        "--checkpoint", required=False,
        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Process N random files from the test .pt segments instead of a single WAV",
    )
    args = parser.parse_args()

    if args.batch > 0:
        # Batch mode: pick N random .pt test segments
        import glob, random
        wav_dir  = config.DEV_NOISY_DIR
        wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
        if not wav_files:
            raise RuntimeError(f"No WAV files found in {wav_dir}")

        random.seed(None)   # truly random each run
        chosen = random.sample(wav_files, min(args.batch, len(wav_files)))
        print(f"[inference] Batch mode: processing {len(chosen)} random files from noisy_testclips")

        # Load model once, reuse for all files
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = UNetDenoiser().to(device)
        epoch, _ = load_checkpoint(model, args.checkpoint)
        model.eval()
        print(f"[inference] Loaded model from epoch {epoch}")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        for i, wav_path in enumerate(chosen, 1):
            noisy_wav = utils.load_audio(wav_path)

            with torch.no_grad():
                clean_wav = denoise_waveform(model, noisy_wav, device)
            clean_wav = postprocess(clean_wav, noisy_wav)

            stem      = os.path.splitext(os.path.basename(wav_path))[0]
            out_wav   = os.path.join(config.OUTPUT_DIR, f"{stem}_denoised.wav")
            noisy_out = os.path.join(config.OUTPUT_DIR, f"{stem}_noisy.wav")

            utils.save_audio(clean_wav, out_wav)
            utils.save_audio(noisy_wav, noisy_out)
            plot_spectrograms(noisy_wav, clean_wav, out_wav)
            print(f"  [{i}/{len(chosen)}] {stem}")

        print(f"[inference] Done. Results in {config.OUTPUT_DIR}")

    else:
        # Single file mode
        if not args.input:
            parser.error("Provide --input <wav> or --batch <N>")

        if not args.output:
            filename    = os.path.basename(args.input)
            args.output = os.path.join(config.OUTPUT_DIR, filename)

        run_inference(args.input, args.output, args.checkpoint)