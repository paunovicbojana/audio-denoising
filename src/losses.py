"""
losses.py

Loss functions used during training

Combined loss:
    L = -SI_SDR + 0.1 * MultiResolutionSTFTLoss

SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):
    Measures similarity between predicted and clean waveform regardless of
    their absolute loudness -> higher SI-SDR = better -> we minimise -SI-SDR

Multi-Resolution STFT Loss:
    Compares spectrograms at several time-frequency resolutions so the model
    learns to preserve both fast transients (fine resolution) and slow phonetic
    structure (coarse resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio loss

    Works on raw waveforms (not spectrograms)

    Formula:
        s_target = (<s_hat, s> / ||s||^2) * s          (projection of pred onto target)
        e_noise  = s_hat - s_target                    (distortion component)
        SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
        loss     = -mean(SI-SDR)                       (minimise the negative)

    Args:
        pred:   (batch, samples) - reconstructed waveform
        target: (batch, samples) - clean reference waveform

    Returns:
        scalar loss tensor
    """
    # Zero-mean both signals (SI-SDR is defined for zero-mean signals)
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Compute the projection of pred onto target
    dot        = (pred * target).sum(dim=-1, keepdim=True)
    target_pow = (target * target).sum(dim=-1, keepdim=True).clamp(min=eps)
    s_target   = (dot / target_pow) * target

    # Distortion = everything that is NOT the projected signal
    e_noise = pred - s_target

    # SI-SDR in dB
    si_sdr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1).clamp(min=eps) /
        (e_noise  ** 2).sum(dim=-1).clamp(min=eps)
    )

    # Return negative mean (we minimise, so lower loss = higher SI-SDR)
    return -si_sdr.mean()


def _stft_loss_single(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    n_fft:      int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """
    Compute STFT loss at one resolution

    Two components:
        spectral_convergence = ||M_target - M_pred||_F / ||M_target||_F
        log_magnitude        = ||log(M_target + eps) - log(M_pred + eps)||_1 / N

    Both components penalise differences in the magnitude spectrogram.
    Spectral convergence focuses on large-energy regions; log-magnitude
    focuses more on quiet details

    Args:
        pred / target: (batch, samples) - waveforms

    Returns:
        scalar loss tensor
    """
    eps    = 1e-7
    window = torch.hann_window(win_length, device=pred.device)

    def magnitude(x):
        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True,
        )
        return stft.abs()

    M_pred   = magnitude(pred)
    M_target = magnitude(target)

    # Spectral convergence loss (Frobenius norm ratio)
    sc_loss = torch.norm(M_target - M_pred, p="fro") / torch.norm(M_target, p="fro").clamp(min=eps)

    # Log-magnitude loss (L1)
    log_pred   = torch.log(M_pred   + eps)
    log_target = torch.log(M_target + eps)
    lm_loss    = F.l1_loss(log_pred, log_target)

    return sc_loss + lm_loss


def multi_resolution_stft_loss(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    resolutions: list[tuple[int, int, int]] = config.STFT_RESOLUTIONS,
) -> torch.Tensor:
    """
    Average STFT loss across multiple (n_fft, hop_length, win_length) settings

    Using multiple resolutions forces the model to be accurate at both:
        - fine time resolution (small n_fft -> captures fast transients)
        - fine frequency resolution (large n_fft -> captures tonal structure)

    Args:
        pred / target: (batch, samples)
        resolutions:   list of (n_fft, hop, win) tuples

    Returns:
        scalar loss tensor
    """
    total = torch.tensor(0.0, device=pred.device)
    for n_fft, hop, win in resolutions:
        total = total + _stft_loss_single(pred, target, n_fft, hop, win)
    return total / len(resolutions)


class DenoisingLoss(nn.Module):
    """
    Combined loss: L = -SI_SDR + w * MultiResSTFTLoss

    The STFT loss provides a perceptually grounded spectral penalty while
    SI-SDR ensures overall waveform fidelity.  The weight (default 0.1) keeps
    both terms in a similar numeric range

    Both pred and target must be raw waveforms of shape (batch, samples)
    """

    def __init__(self, stft_weight: float = config.STFT_LOSS_WEIGHT):
        super().__init__()
        self.stft_weight = stft_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor (used for backpropagation)
            components: dict with individual loss values for logging
        """
        loss_si_sdr  = si_sdr_loss(pred, target)
        loss_stft    = multi_resolution_stft_loss(pred, target)
        total        = loss_si_sdr + self.stft_weight * loss_stft

        components = {
            "loss_si_sdr": loss_si_sdr.item(),
            "loss_stft":   loss_stft.item(),
            "loss_total":  total.item(),
        }
        return total, components