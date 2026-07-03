"""
preprocess.py

One-time preprocessing script
Run this ONCE before training

What it does:
  1. Scans clean_speech and noise directories for WAV files
  2. Synthesizes noisy/clean pairs using noisyspeech_synthesizer logic
     (mixes clean speech with noise at a random SNR in [-5, 30] dB)
  3. Applies impulse response convolution if IR files are available
  4. Cuts every pair into 2-second segments with 50% overlap
  5. For each segment: STFT -> log-compress -> z-score normalise
  6. Saves (noisy_spec, clean_spec, noisy_phase, clean_phase, mean, std)
     as a .pt file in data/processed/

After running this script, train.py reads directly from .pt files -
no audio loading or STFT computation happens during training, which
makes each epoch significantly faster

Run:
    python src/preprocess.py
    python src/preprocess.py --clean_dir path/to/clean --noise_dir path/to/noise
    python src/preprocess.py --limit 500   # process only 500 pairs (for quick tests)
"""

import argparse
import os
import glob
import random

import torch

import config
import utils
from dataset import segment_waveform, waveform_to_input
from utils  import normalise_with_stats


def mix_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mix clean speech and noise at the requested SNR level (in dB)

    SNR = 10 * log10(P_speech / P_noise)
        -> we scale noise so that the power ratio matches the target SNR

    Args:
        clean:  1-D float32 tensor (speech signal)
        noise:  1-D float32 tensor (noise signal, will be cropped/looped to match)
        snr_db: target signal-to-noise ratio in dB

    Returns:
        noisy: clean + scaled noise, same length as clean
    """
    # Loop or trim noise to match clean length
    if noise.shape[0] < clean.shape[0]:
        repeats = (clean.shape[0] // noise.shape[0]) + 1
        noise   = noise.repeat(repeats)
    noise = noise[:clean.shape[0]]

    # Random offset so we don't always start from the same point in the noise file
    offset = random.randint(0, max(0, noise.shape[0] - clean.shape[0]))
    noise  = noise[offset:offset + clean.shape[0]]

    # Compute power of each signal
    clean_power = (clean ** 2).mean().clamp(min=eps)
    noise_power = (noise ** 2).mean().clamp(min=eps)

    # Scale noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale        = (target_noise_power / noise_power).sqrt()

    return clean + noise_scale * noise


def apply_impulse_response(
    waveform: torch.Tensor,
    ir: torch.Tensor,
) -> torch.Tensor:
    """
    Convolve a waveform with an impulse response to simulate room acoustics

    This makes the synthetic training data more realistic - the model learns
    to denoise speech that sounds like it was recorded in a real room, not
    just anechoic clean speech + additive noise

    Convolution is done in the frequency domain (FFT-based) for speed
    The output is trimmed to the original waveform length

    Args:
        waveform: 1-D float32 tensor (clean speech)
        ir:       1-D float32 tensor (impulse response)

    Returns:
        reverberant waveform, same length as input
    """
    n     = waveform.shape[0] + ir.shape[0] - 1
    # Next power of 2 for efficient FFT
    nfft  = 1 << (n - 1).bit_length()

    W = torch.fft.rfft(waveform, n=nfft)
    H = torch.fft.rfft(ir,       n=nfft)
    Y = torch.fft.irfft(W * H,   n=nfft)

    # Trim to original length and normalise to avoid clipping
    Y = Y[:waveform.shape[0]]
    peak = Y.abs().max()
    if peak > 1.0:
        Y = Y / peak

    return Y


def collect_wavs(directory: str) -> list[str]:
    """Recursively find all WAV files under a directory"""
    pattern = os.path.join(directory, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def preprocess(
    clean_dir:  str,
    noise_dir:  str,
    output_dir: str,
    ir_dir:     str,
    snr_min:    float = -5.0,
    snr_max:    float = 30.0,
    limit:      int   = 0,
    seed:       int   = 42,
) -> None:
    """
    Synthesize noisy/clean pairs, segment, and save as .pt files

    Args:
        clean_dir:  folder with clean speech WAVs
        noise_dir:  folder with noise WAVs
        output_dir: where to write .pt segment files
        snr_min:    minimum SNR for mixing (dB)
        snr_max:    maximum SNR for mixing (dB)
        limit:      if set, process at most this many clean files (quick test)
        seed:       random seed for reproducibility
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    clean_files = collect_wavs(clean_dir)
    noise_files = collect_wavs(noise_dir)

    if not clean_files:
        raise RuntimeError(f"No WAV files found in clean_dir: {clean_dir}")
    if not noise_files:
        raise RuntimeError(f"No WAV files found in noise_dir: {noise_dir}")

    ir_files = collect_wavs(ir_dir) if ir_dir and os.path.isdir(ir_dir) else []

    if limit:
        clean_files = clean_files[:limit]

    print(f"[preprocess] Clean files : {len(clean_files)}")
    print(f"[preprocess] Noise files : {len(noise_files)}")
    print(f"[preprocess] IR files    : {len(ir_files)} ({'enabled' if ir_files else 'disabled'})")
    print(f"[preprocess] Output dir  : {output_dir}")
    print(f"[preprocess] SNR range   : [{snr_min}, {snr_max}] dB")

    total_segments = 0
    skipped        = 0

    for file_idx, clean_path in enumerate(clean_files):
        # Load clean speech
        try:
            clean_wav = utils.load_audio(clean_path)
        except Exception as e:
            print(f"  [skip] {clean_path}: {e}")
            skipped += 1
            continue

        # Pick a random noise file and SNR
        noise_path = random.choice(noise_files)
        snr_db     = random.uniform(snr_min, snr_max)

        try:
            noise_wav = utils.load_audio(noise_path)
        except Exception as e:
            print(f"  [skip] noise {noise_path}: {e}")
            skipped += 1
            continue

        # Apply impulse response (room acoustics simulation) if available
        if ir_files:
            ir_path = random.choice(ir_files)
            try:
                ir_wav    = utils.load_audio(ir_path)
                clean_wav = apply_impulse_response(clean_wav, ir_wav)
            except Exception:
                pass

        # Synthesize noisy waveform
        noisy_wav = mix_at_snr(clean_wav, noise_wav, snr_db)

        # Segment both waveforms
        clean_segs = segment_waveform(clean_wav)
        noisy_segs = segment_waveform(noisy_wav)

        n_segs = min(len(clean_segs), len(noisy_segs))

        for seg_idx in range(n_segs):
            clean_seg = clean_segs[seg_idx]
            noisy_seg = noisy_segs[seg_idx]

            # STFT + log-compress + normalise with noisy stats
            try:
                noisy_spec, noisy_phase, mean, std = waveform_to_input(noisy_seg)
                raw_clean_mag, clean_phase = utils.compute_stft(clean_seg)
                compressed_clean           = utils.log_compress(raw_clean_mag)
                clean_spec_2d              = normalise_with_stats(compressed_clean, mean, std)
                clean_spec                 = clean_spec_2d.unsqueeze(0)  # (1, F, T)
            except Exception as e:
                print(f"  [warn] STFT failed for {clean_path} seg {seg_idx}: {e}")
                continue

            # Build output filename
            # Format: fileIDX_segIDX_snrVALUE.pt
            stem     = os.path.splitext(os.path.basename(clean_path))[0]
            filename = f"{stem}_seg{seg_idx:03d}_snr{snr_db:.1f}.pt"
            out_path = os.path.join(output_dir, filename)

            # Save as dict of tensors
            torch.save({
                "noisy_spec":  noisy_spec,   # (1, F, T)  float32
                "clean_spec":  clean_spec,   # (1, F, T)  float32
                "noisy_phase": noisy_phase,  # (F, T)     float32
                "clean_phase": clean_phase,  # (F, T)     float32
                "mean":        mean,         # scalar
                "std":         std,          # scalar
                "snr_db":      snr_db,       # float - useful for stratified eval
                "clean_path":  clean_path,   # str  - for debugging
            }, out_path)

            total_segments += 1

        # Progress reporting
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == len(clean_files):
            print(f"  [{file_idx + 1}/{len(clean_files)}] {total_segments} segments saved")

    print(f"\n[preprocess] Done.")
    print(f"  Total segments : {total_segments}")
    print(f"  Skipped files  : {skipped}")
    print(f"  Output dir     : {output_dir}")


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Faster alternative to DenoisingDataset - reads pre-saved .pt files
    instead of loading and processing WAVs on the fly

    Use this in train.py after running preprocess.py once:
        from preprocess import PreprocessedDataset
        ds = PreprocessedDataset("data/processed/train")

    Each item is a dict with keys:
        noisy_spec, clean_spec, noisy_phase, clean_phase, mean, std, snr_db
    """

    def __init__(self, pt_dir: str, augment: bool = False, max_files = None):
        files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))

        if not files:
            raise RuntimeError(f"No .pt files found in {pt_dir}. Run preprocess.py first.")

        if max_files is not None and max_files < len(files):
            import random
            random.seed(42)
            files = random.sample(files, max_files)

        self.files   = files
        self.augment = augment

        total = len(glob.glob(os.path.join(pt_dir, "*.pt")))
        if max_files is not None and max_files < total:
            print(f"[PreprocessedDataset] {len(self.files)}/{total} segments in {pt_dir} (limited)")
        else:
            print(f"[PreprocessedDataset] {len(self.files)} segments in {pt_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        if self.augment:
            # Light augmentation on the already-compressed spectrogram:
            #   add a tiny amount of Gaussian noise to the noisy spec
            #   so the model does not memorise exact spectrogram values
            noise = torch.randn_like(data["noisy_spec"]) * 0.01
            data["noisy_spec"] = data["noisy_spec"] + noise

        # Remap keys to match DenoisingDataset so train.py works with both
        #   .pt file saves:  noisy_spec, clean_spec
        #   train.py expects: noisy,      clean
        return {
            "noisy":       data["noisy_spec"],
            "clean":       data["clean_spec"],
            "noisy_phase": data["noisy_phase"],
            "clean_phase": data["clean_phase"],
            "mean":        data["mean"],
            "std":         data["std"],
        }


def split_preprocessed(
    pt_dir:      str,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = 42,
) -> None:
    """
    After preprocessing, call this to move .pt files into train/ val/ test/
    subfolders so PreprocessedDataset can load each split separately

    The split is random at the file level (not speaker-stratified, since
    speaker info is not always available after synthesis
    For a proper speaker-stratified split use dataset.py's build_dataloaders instead

    Args:
        pt_dir: directory containing all .pt files (output of preprocess())
    """
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    random.seed(seed)
    random.shuffle(files)

    n       = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(pt_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for src in split_files:
            dst = os.path.join(split_dir, os.path.basename(src))
            os.rename(src, dst)
        print(f"  {split_name}: {len(split_files)} segments -> {split_dir}")

    print("[split] Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DNS audio into .pt segment files")

    parser.add_argument("--clean_dir",  default=config.CLEAN_DIR,
                        help="Directory with clean speech WAVs")
    parser.add_argument("--noise_dir",  default=config.NOISE_DIR,
                        help="Directory with noise WAVs")
    parser.add_argument("--ir_dir",     default=None,
                        help="Directory with impulse response WAVs (optional)")
    parser.add_argument("--output_dir", default=config.PROCESSED_DIR,
                        help="Where to save .pt files")
    parser.add_argument("--snr_min",    type=float, default=-5.0,
                        help="Minimum SNR for mixing (dB)")
    parser.add_argument("--snr_max",    type=float, default=30.0,
                        help="Maximum SNR for mixing (dB)")
    parser.add_argument("--limit",      type=int,   default=None,
                        help="Process only this many clean files (for quick testing)")
    parser.add_argument("--split",      action="store_true",
                        help="After preprocessing, split into train/val/test subfolders")

    args = parser.parse_args()

    preprocess(
        clean_dir  = args.clean_dir,
        noise_dir  = args.noise_dir,
        output_dir = args.output_dir,
        ir_dir     = args.ir_dir,
        snr_min    = args.snr_min,
        snr_max    = args.snr_max,
        limit      = args.limit,
    )

    if args.split:
        print("\n[preprocess] Splitting into train/val/test...")
        split_preprocessed(args.output_dir)