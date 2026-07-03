"""
dataset.py

PyTorch Dataset that serves (noisy_spectrogram, clean_spectrogram) pairs
to the training loop

Pipeline for each audio pair:
  1. Load noisy + clean WAV
  2. Segment into 2-second chunks with 50% overlap
  3. STFT -> magnitude
  4. Log-compress the magnitude
  5. Z-score normalise (noisy mean/std used for BOTH noisy and clean)
  6. Return as (noisy_tensor, clean_tensor) where each is shape
     (1, n_freq_bins, n_frames)  ->  treated like a 1-channel image by the U-Net

The dataset also handles:
  - zero-padding of short clips
  - skipping near-empty trailing segments (< 25% real content)
  - SNR data augmentation (random ±3 dB shift at training time)
  - speaker-stratified train / val / test splitting
"""

import os
import glob
import random
from collections import defaultdict

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

import config
import utils

MIN_CONTENT_RATIO = 0.25  # drop trailing segments with < 25% real audio


def segment_waveform(waveform: torch.Tensor) -> list[torch.Tensor]:
    """
    Split a 1-D waveform into overlapping segments of SEGMENT_SAMPLES length

    - Overlap is 50% -> step = SEGMENT_SAMPLES // 2
    - If the last segment is shorter than SEGMENT_SAMPLES it is zero-padded,
      BUT only if it contains at least MIN_CONTENT_RATIO real samples
      Near-empty trailing chunks are dropped entirely
    - A waveform shorter than SEGMENT_SAMPLES is zero-padded to that length
      and returned as a single segment

    Returns:
        list of tensors, each of shape (SEGMENT_SAMPLES,)
    """
    seg_len = config.SEGMENT_SAMPLES
    step    = int(seg_len * (1 - config.OVERLAP))   # 50% overlap -> step = seg_len // 2
    total   = waveform.shape[0]

    if total < seg_len:
        # Pad the whole waveform and return it as one segment
        pad = torch.zeros(seg_len - total)
        return [torch.cat([waveform, pad])]

    segments = []
    start = 0
    while start < total:
        end        = start + seg_len
        real_len   = min(end, total) - start   # samples of actual audio in this chunk

        if real_len < seg_len * MIN_CONTENT_RATIO:
            break

        chunk = waveform[start:end]
        if chunk.shape[0] < seg_len:
            pad   = torch.zeros(seg_len - chunk.shape[0])
            chunk = torch.cat([chunk, pad])

        segments.append(chunk)
        start += step

    return segments


def _count_segments_from_info(path: str) -> int:
    """
    Estimate the number of segments a file will produce using only its
    duration metadata (no audio decoding)

    Uses soundfile.info() which reads only the file header - O(1) I/O
    regardless of file duration
    """
    try:
        info     = sf.info(path)
        # Resample frame count if needed (same logic as utils.load_audio)
        n_frames = info.frames
        if info.samplerate != config.SAMPLE_RATE:
            import math
            n_frames = int(n_frames * config.SAMPLE_RATE / info.samplerate)

        seg_len = config.SEGMENT_SAMPLES
        step    = int(seg_len * (1 - config.OVERLAP))

        if n_frames < seg_len:
            return 1

        count = 0
        start = 0
        while start < n_frames:
            real_len = min(start + seg_len, n_frames) - start
            if real_len < seg_len * MIN_CONTENT_RATIO:
                break
            count += 1
            start += step

        return max(count, 1)   # always at least one segment

    except Exception:
        return 1


def waveform_to_input(
    waveform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a waveform segment to a spectrogram tensor ready for the U-Net

    Returns:
        spec_norm:  (1, n_freq_bins, n_frames) - log-compressed, normalised
        phase:      (n_freq_bins, n_frames)    - original phase for iSTFT
        mean, std:  scalars used to undo normalisation during reconstruction
    """
    magnitude, phase = utils.compute_stft(waveform)
    compressed       = utils.log_compress(magnitude)
    spec_norm, mean, std = utils.normalise(compressed)
    # Add channel dimension so it looks like a 1-channel image: (1, F, T)
    return spec_norm.unsqueeze(0), phase, mean, std


def normalise_with_stats(
    waveform: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a waveform to a log-magnitude spectrogram and normalise it
    using EXTERNALLY provided mean and std (from the paired noisy signal)

    Returns:
        spec_norm:  (1, n_freq_bins, n_frames)
        phase:      (n_freq_bins, n_frames)
    """
    magnitude, phase = utils.compute_stft(waveform)
    compressed       = utils.log_compress(magnitude)
    spec_norm        = utils.normalise_with_stats(compressed, mean, std)
    return spec_norm.unsqueeze(0), phase



class DenoisingDataset(Dataset):
    """
    Pairs of (noisy_spectrogram, clean_spectrogram) segments from the
    DNS-Challenge dataset

    Each item is a dict:
        {
          "noisy":       (1, n_freq_bins, n_frames)  float32 tensor
          "clean":       (1, n_freq_bins, n_frames)  float32 tensor - same normalisation as noisy
          "noisy_phase": (n_freq_bins, n_frames)     for inference reconstruction
          "clean_phase": (n_freq_bins, n_frames)
          "mean":        scalar float  (from noisy, shared by clean)
          "std":         scalar float  (from noisy, shared by clean)
          "speaker_id":  str
        }

    Args:
        pairs:       list of (noisy_path, clean_path, speaker_id) tuples
        augment:     if True, apply random SNR shift (use only during training)
    """

    def __init__(self, pairs: list[tuple[str, str, str]], augment: bool = False):
        self.pairs   = pairs
        self.augment = augment
        self._items = self._build_index()


    def _build_index(self) -> list[tuple[tuple, int]]:
        """
        Walk through every pair and build a flat list of (pair, seg_idx)
        so __getitem__ has O(1) access
        """
        items = []
        for pair in self.pairs:
            noisy_path, clean_path, speaker_id = pair
            try:
                n_segs = _count_segments_from_info(noisy_path)
                for seg_idx in range(n_segs):
                    items.append((pair, seg_idx))
            except Exception as e:
                print(f"[dataset] Skipping {noisy_path}: {e}")
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        (noisy_path, clean_path, speaker_id), seg_idx = self._items[idx]

        # Load full waveforms
        noisy_wav = utils.load_audio(noisy_path)
        clean_wav = utils.load_audio(clean_path)

        # Trim both to their common length (DNS dev/test clean files can
        # differ slightly in length from the noisy counterpart)
        min_len   = min(noisy_wav.shape[0], clean_wav.shape[0])
        noisy_wav = noisy_wav[:min_len]
        clean_wav = clean_wav[:min_len]

        # Apply SNR augmentation: scale the noise component by ±SNR_AUGMENT_DB
        if self.augment:
            shift_db    = random.uniform(-config.SNR_AUGMENT_DB, config.SNR_AUGMENT_DB)
            noise       = noisy_wav - clean_wav
            noise_scale = 10 ** (shift_db / 20)
            noisy_wav   = clean_wav + noise * noise_scale

        # Extract the requested segment
        noisy_segs = segment_waveform(noisy_wav)
        clean_segs = segment_waveform(clean_wav)

        # Guard against index going out of range
        seg_idx = min(seg_idx, len(noisy_segs) - 1, len(clean_segs) - 1)

        noisy_seg = noisy_segs[seg_idx]
        clean_seg = clean_segs[seg_idx]

        # Compute noisy spectrogram first to get mean/std,
        # then normalise clean with the same statistics
        noisy_spec, noisy_phase, mean, std = waveform_to_input(noisy_seg)
        clean_spec, clean_phase            = normalise_with_stats(clean_seg, mean, std)

        return {
            "noisy":       noisy_spec,
            "clean":       clean_spec,
            "noisy_phase": noisy_phase,
            "clean_phase": clean_phase,
            "mean":        mean,
            "std":         std,
            "speaker_id":  speaker_id,
        }


def collect_pairs(noisy_dir: str, clean_dir: str) -> list[tuple[str, str, str]]:
    """
    Match every noisy WAV in noisy_dir with the corresponding clean WAV in
    clean_dir by filename stem

    DNS filenames follow the pattern:
        noisy_dir/fileid_<id>_snr<X>_tl<Y>_fileid_<Z>.wav
        clean_dir/fileid_<id>.wav

    Args:
        noisy_dir: path to folder of noisy WAV files
        clean_dir: path to folder of matching clean WAV files

    Returns:
        list of (noisy_path, clean_path, speaker_id) tuples
    """
    noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    pairs = []

    for noisy_path in noisy_paths:
        stem  = os.path.splitext(os.path.basename(noisy_path))[0]
        # DNS dev testset uses identical filenames for noisy and clean
        clean_path = os.path.join(clean_dir, os.path.basename(noisy_path))

        if not os.path.exists(clean_path):
            fileid = stem.split("_snr")[0]
            clean_path = os.path.join(clean_dir, fileid + ".wav")

        if not os.path.exists(clean_path):
            continue

        speaker_id = stem.split("_")[0]
        pairs.append((noisy_path, clean_path, speaker_id))

    return pairs


def split_by_speaker(
    pairs: list[tuple[str, str, str]],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """
    Group pairs by speaker_id, shuffle speakers, then assign whole
    speakers to train / val / test so no speaker appears in two splits

    Returns:
        train_pairs, val_pairs, test_pairs
    """
    by_speaker: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_speaker[pair[2]].append(pair)

    speakers = list(by_speaker.keys())
    random.seed(seed)
    random.shuffle(speakers)

    n          = len(speakers)
    n_train    = int(n * train_ratio)
    n_val      = int(n * val_ratio)

    train_spks = speakers[:n_train]
    val_spks   = speakers[n_train:n_train + n_val]
    test_spks  = speakers[n_train + n_val:]

    train_pairs = [p for s in train_spks for p in by_speaker[s]]
    val_pairs   = [p for s in val_spks   for p in by_speaker[s]]
    test_pairs  = [p for s in test_spks  for p in by_speaker[s]]

    print(f"[split] {len(train_spks)} train / {len(val_spks)} val / {len(test_spks)} test speakers")
    print(f"[split] {len(train_pairs)} / {len(val_pairs)} / {len(test_pairs)} file pairs")

    return train_pairs, val_pairs, test_pairs


def build_dataloaders(
    noisy_dir:   str = config.DEV_NOISY_DIR,
    clean_dir:   str = config.DEV_CLEAN_DIR,
    batch_size:  int = config.BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Full pipeline: scan directories -> split by speaker -> create DataLoaders

    Returns:
        train_loader, val_loader, test_loader
    """
    pairs = collect_pairs(noisy_dir, clean_dir)
    if not pairs:
        raise RuntimeError(f"No matching pairs found in {noisy_dir} / {clean_dir}")

    train_pairs, val_pairs, test_pairs = split_by_speaker(pairs)

    train_ds = DenoisingDataset(train_pairs, augment=True)
    val_ds   = DenoisingDataset(val_pairs,   augment=False)
    test_ds  = DenoisingDataset(test_pairs,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader