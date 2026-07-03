"""
train.py

Training loop for the U-Net speech enhancement model

What happens here:
  1. Build train / val DataLoaders from preprocessed .pt segment files
  2. Instantiate the model, loss function, AdamW optimiser
  3. Set up cosine LR schedule with warmup
  4. For each epoch:
       a. Train on all batches -> backprop -> update weights
       b. Evaluate on validation set -> log SI-SDR
       c. Save checkpoint if val SI-SDR improved
       d. Stop early if no improvement for EARLY_STOP_PATIENCE epochs
  5. Log everything to Weights & Biases (optional)

Run:
    python src/train.py
"""

import os
import math

import torch
import torch.optim as optim

import config
import utils
from preprocess import PreprocessedDataset
from model      import UNetDenoiser
from losses     import DenoisingLoss
from torch.utils.data import DataLoader


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def lr_lambda(epoch: int) -> float:
    """
    Returns the LR multiplier for a given epoch

    Schedule:
      epochs 0 … WARMUP_EPOCHS-1  -> linear ramp from 0 to 1
      epochs WARMUP_EPOCHS … END  -> cosine decay from 1 to MIN_LR/LR

    Using a warmup prevents large gradient updates at the start of training
    when the randomly-initialised model produces very noisy gradients
    """
    if epoch < config.WARMUP_EPOCHS:
        return (epoch + 1) / config.WARMUP_EPOCHS

    # Cosine decay phase
    progress = (epoch - config.WARMUP_EPOCHS) / max(config.MAX_EPOCHS - config.WARMUP_EPOCHS, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_frac = config.MIN_LR / config.LEARNING_RATE
    return min_frac + (1.0 - min_frac) * cosine


def train_one_epoch(
    model:      UNetDenoiser,
    loader,
    criterion:  DenoisingLoss,
    optimiser:  optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    """
    Iterate over all batches in the training loader, compute the loss,
    backpropagate, and update the model weights

    Returns a dict of averaged loss components for logging
    """
    model.train()
    totals    = {}
    n_batches = 0

    for batch in loader:
        noisy = batch["noisy"].to(device)   # (B, 1, F, T) noisy spectrogram
        clean = batch["clean"].to(device)   # (B, 1, F, T) clean spectrogram

        # Forward: predict soft mask
        mask = model(noisy)                 # (B, 1, F, T)  values in [0, 1]

        # Convert spectrograms to waveforms BEFORE applying mask
        estimated_wav = _batch_to_waveform_masked(noisy, mask, batch, phase_key="noisy_phase")
        clean_wav     = _batch_to_waveform(clean, batch, phase_key="clean_phase")

        # Compute loss
        loss, components = criterion(estimated_wav, clean_wav)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in the LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        # Accumulate for logging
        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(
    model:     UNetDenoiser,
    loader,
    criterion: DenoisingLoss,
    device:    torch.device,
) -> dict[str, float]:
    """
    Run the model on the validation set without computing gradients
    Returns averaged loss components
    """
    model.eval()
    totals    = {}
    n_batches = 0

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        mask = model(noisy)
        estimated_wav = _batch_to_waveform_masked(noisy, mask, batch, phase_key="noisy_phase")
        clean_wav     = _batch_to_waveform(clean, batch, phase_key="clean_phase")

        _, components = criterion(estimated_wav, clean_wav)

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def _batch_to_waveform(
    spec_batch: torch.Tensor,
    batch:      dict,
    phase_key:  str = "noisy_phase",
) -> torch.Tensor:
    """
    Convert a batch of log-compressed spectrograms back to waveforms

    Steps (inverse of dataset.waveform_to_input):
      1. Remove the channel dimension   (B, 1, F, T) -> (B, F, T)
      2. Denormalise using saved mean/std
      3. Undo log-compression: magnitude = exp(x) - 1
      4. iSTFT using the phase specified by phase_key
      5. Stack results into (B, samples)

    Args:
        spec_batch: (B, 1, F, T) log-compressed normalised spectrogram
        batch:      dict from DataLoader - contains mean, std, and phase tensors
        phase_key:  "noisy_phase" for estimated output (inference path),
                    "clean_phase" for clean reference (loss target)
    """
    mean  = batch["mean"].to(spec_batch.device)
    std   = batch["std"].to(spec_batch.device)
    phase = batch[phase_key].to(spec_batch.device)   # (B, F, T)

    B = spec_batch.shape[0]
    waveforms = []

    for i in range(B):
        s = spec_batch[i, 0]                        # (F, T)
        # Undo normalisation
        s = utils.denormalise(s, mean[i], std[i])
        # Undo log-compression: log1p inverse is expm1
        magnitude = torch.expm1(s).clamp(min=0)
        # Reconstruct waveform using the specified phase
        wav = utils.reconstruct_waveform(magnitude, phase[i])
        waveforms.append(wav)

    # Pad to equal length and stack
    max_len = max(w.shape[0] for w in waveforms)
    padded  = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    return torch.stack(padded, dim=0)   # (B, samples)


def _batch_to_waveform_masked(
    noisy_batch: torch.Tensor,
    mask_batch:  torch.Tensor,
    batch:       dict,
    phase_key:   str = "noisy_phase",
) -> torch.Tensor:
    """
    Apply the predicted mask to raw noisy magnitudes and reconstruct waveforms

    The correct order is:
      1. Denormalise noisy spectrogram -> log-compressed magnitude
      2. expm1 -> raw magnitude
      3. Multiply by mask  ← mask applied here on raw magnitudes
      4. iSTFT with noisy phase -> waveform

    This is the same pipeline as inference (evaluate.py / inference.py),
    so train and inference are now consistent.
    """
    mean  = batch["mean"].to(noisy_batch.device)
    std   = batch["std"].to(noisy_batch.device)
    phase = batch[phase_key].to(noisy_batch.device)   # (B, F, T)

    B = noisy_batch.shape[0]
    waveforms = []

    for i in range(B):
        # Denormalise and undo log-compression to get raw magnitude
        s             = noisy_batch[i, 0]                        # (F, T)
        s             = utils.denormalise(s, mean[i], std[i])
        raw_magnitude = torch.expm1(s).clamp(min=0)

        # Apply mask on raw magnitude
        m                = mask_batch[i, 0]                      # (F, T)
        clean_magnitude  = (m * raw_magnitude).clamp(min=0)

        wav = utils.reconstruct_waveform(clean_magnitude, phase[i])
        waveforms.append(wav)

    max_len = max(w.shape[0] for w in waveforms)
    padded  = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    return torch.stack(padded, dim=0)


def save_checkpoint(model: UNetDenoiser, epoch: int, val_si_sdr: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_si_sdr":  val_si_sdr,
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(model: UNetDenoiser, path: str) -> tuple[int, float]:
    """
    Load weights into model
    Return (epoch, val_si_sdr)
    """
    if model._bottleneck is None:
        with torch.no_grad():
            model_device = next(model.parameters()).device
            dummy = torch.zeros(
                1, 1, config.N_FREQ_BINS, config.SEGMENT_SAMPLES // config.HOP_LENGTH + 1,
                device=model_device,
            )
            model(dummy)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["epoch"], ckpt["val_si_sdr"]


def train() -> None:
    utils.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # Data
    _seg_root    = os.path.join(config.PROCESSED_DIR, "segments")
    train_ds     = PreprocessedDataset(os.path.join(_seg_root, "train"), augment=True,  max_files=config.MAX_TRAIN_FILES)
    val_ds       = PreprocessedDataset(os.path.join(_seg_root, "val"),   augment=False, max_files=config.MAX_VAL_FILES)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model
    model = UNetDenoiser().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}")

    # Optimiser + schedule
    optimiser = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
    criterion = DenoisingLoss()

    # WANDB
    if WANDB_AVAILABLE and config.WANDB_PROJECT:
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY, config={
            "lr": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
            "epochs": config.MAX_EPOCHS, "lstm_hidden": config.LSTM_HIDDEN,
        })

    # Training loop
    best_val_si_sdr  = float("-inf")
    patience_counter = 0
    best_ckpt_path   = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(config.MAX_EPOCHS):
        current_lr = optimiser.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}  (lr={current_lr:.2e})")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimiser, device, epoch)
        # Validate
        val_metrics   = validate(model, val_loader, criterion, device)

        scheduler.step()

        # SI-SDR is stored as negative in the loss -> recover the positive value
        val_si_sdr = -val_metrics["loss_si_sdr"]

        print(
            f"  train loss={train_metrics['loss_total']:.4f} "
            f"| val loss={val_metrics['loss_total']:.4f} "
            f"| val SI-SDR={val_si_sdr:.2f} dB"
        )

        # WANDB logging
        if WANDB_AVAILABLE and config.WANDB_PROJECT:
            wandb.log({"epoch": epoch + 1, "lr": current_lr,
                       **{f"train/{k}": v for k, v in train_metrics.items()},
                       **{f"val/{k}":   v for k, v in val_metrics.items()}})

        # Save best checkpoint
        if val_si_sdr > best_val_si_sdr:
            best_val_si_sdr = val_si_sdr
            patience_counter = 0
            save_checkpoint(model, epoch + 1, val_si_sdr, best_ckpt_path)
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{config.EARLY_STOP_PATIENCE})")

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n[train] Early stopping at epoch {epoch + 1}")
            break

    print(f"\n[train] Done. Best val SI-SDR: {best_val_si_sdr:.2f} dB")
    if WANDB_AVAILABLE and config.WANDB_PROJECT:
        wandb.finish()


if __name__ == "__main__":
    train()