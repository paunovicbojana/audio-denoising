"""
model.py

U-Net architecture with a Bidirectional LSTM bottleneck for speech enhancement

Architecture overview:
  Encoder  (5 conv blocks)   ->  compress spectrogram, extract features
  Bottleneck (BiLSTM)        ->  model long-range temporal context
  Decoder  (5 transpose-conv blocks) ->  reconstruct clean spectrogram
  Skip connections           ->  transfer fine-grained detail from encoder to decoder
  Output layer               ->  sigmoid mask M ∈ [0, 1]

Input:  (batch, 1, n_freq_bins, n_frames)   - 1-channel log-magnitude spectrogram
Output: (batch, 1, n_freq_bins, n_frames)   - soft mask, same shape as input
"""

import torch
import torch.nn as nn

import config


class EncoderBlock(nn.Module):
    """
    Single encoder stage that halves the spatial dimensions (H, W) while
    doubling the number of feature channels

    stride=2 does the downsampling inline - no separate pooling layer needed
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1,   # stride=2 -> halves H and W
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Single decoder stage that doubles the spatial dimensions and halves
    the number of feature channels

    It also concatenates the skip connection from the matching encoder block
    before applying the convolution
    Because of the concatenation the actual input channel count is 
    in_channels + skip_channels
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels + skip_channels, out_channels,
                kernel_size=4, stride=2, padding=1,   # stride=2 -> doubles H and W
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class BiLSTMBottleneck(nn.Module):
    """
    Processes the spatially compressed feature map with a Bidirectional LSTM
    to capture long-range temporal dependencies that local convolutions miss

    The feature map is reshaped so that the time axis aligns with the LSTM
    sequence dimension.  The frequency and channel axes are flattened into
    the feature vector

    Input / output shape:  (batch, channels, freq, time)
    """

    def __init__(self, n_channels: int, n_freq: int, hidden: int, n_layers: int):
        super().__init__()
        input_size = n_channels * n_freq   # flattened freq+channel per time step
        self.lstm  = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,   # looks both forward and backward in time
        )
        # Project BiLSTM output (2 * hidden) back to the original feature size
        self.proj = nn.Linear(2 * hidden, input_size)

        self._n_channels = n_channels
        self._n_freq     = n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape   # batch, channels, freq-bins, time-frames

        # Reshape: (B, C, F, T) -> (B, T, C*F) - time is now the sequence axis
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        out, _ = self.lstm(x_seq)           # (B, T, 2*hidden)
        out    = self.proj(out)             # (B, T, C*F)

        # Reshape back: (B, T, C*F) -> (B, C, F, T)
        out = out.view(B, T, C, F).permute(0, 2, 3, 1).contiguous()
        return out


class SpectralAttention(nn.Module):
    """
    Lightweight channel + frequency attention applied after the BiLSTM bottleneck

    Two complementary attention mechanisms:
      - Channel attention  ("which feature maps matter?")
        Squeeze global (freq, time) info into a per-channel descriptor,
        then learn a recalibration vector via two FC layers (SE-Net style)
      - Frequency attention ("which frequency bands matter?")
        Average across channels and time, then learn per-frequency-bin weights

    Both produce sigmoid gates in [0, 1] that are multiplied with the input
    This lets the model suppress noise-dominated frequency bands and
    down-weight less-informative feature channels before decoding

    Input / output shape: (batch, channels, freq, time)  -- unchanged
    """

    def __init__(self, n_channels: int, n_freq: int, reduction: int = 8):
        super().__init__()

        # Channel attention (SE-Net style)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(n_channels, max(n_channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(n_channels // reduction, 4), n_channels),
            nn.Sigmoid(),
        )

        # Frequency attention
        self.freq_att = nn.Sequential(
            nn.Linear(n_freq, max(n_freq // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(n_freq // reduction, 4), n_freq),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape

        # Channel attention: gate each channel map
        ch_gate = self.channel_att(x)          # (B, C)
        x = x * ch_gate.view(B, C, 1, 1)

        # Frequency attention: average over channels and time -> per-freq weight
        freq_desc = x.mean(dim=(1, 3))         # (B, F)
        freq_gate = self.freq_att(freq_desc)   # (B, F)
        x = x * freq_gate.view(B, 1, F, 1)

        return x


class UNetDenoiser(nn.Module):
    """
    U-Net speech enhancement model

    Predicts a soft mask M ∈ [0, 1] from the noisy log-magnitude spectrogram
    The caller multiplies the mask by the noisy magnitude to obtain the
    estimated clean magnitude:
        clean_mag ≈ mask * noisy_mag

    Args:
        encoder_channels:  list of output channel counts for each encoder block
                           e.g. [16, 32, 64, 128, 256]
        lstm_hidden:       hidden units per direction in the BiLSTM
        lstm_layers:       number of stacked LSTM layers
    """

    def __init__(
        self,
        encoder_channels: list[int] = config.ENCODER_CHANNELS,
        lstm_hidden:      int       = config.LSTM_HIDDEN,
        lstm_layers:      int       = config.LSTM_LAYERS,
    ):
        super().__init__()

        # Encoder
        channels = [1] + encoder_channels   # include input channel count
        self.encoders = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1])
            for i in range(len(encoder_channels))
        ])

        # Bottleneck (BiLSTM) + Attention
        #   After 5 encoder blocks with stride=2 each, spatial dims are /32
        #   We don't know exact freq/time dims statically so we compute them
        #   lazily in the first forward pass
        self._lstm_hidden  = lstm_hidden
        self._lstm_layers  = lstm_layers
        self._bottleneck   = None   # created lazily in forward()
        self._attention    = None   # created lazily in forward()

        # Decoder - reverse the encoder channel list; each decoder block gets a skip
        #           from its mirror encoder block
        dec_in     = encoder_channels[-1]   # input channels to first decoder block
        dec_skips  = list(reversed(encoder_channels[:-1]))   # skip channel counts
        dec_out    = list(reversed(encoder_channels[:-1]))   # output channel counts

        self.decoders = nn.ModuleList()
        for skip_ch, out_ch in zip(dec_skips, dec_out):
            self.decoders.append(DecoderBlock(dec_in, skip_ch, out_ch))
            dec_in = out_ch

        # Final projection to mask
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dec_in, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # constrain output to [0, 1]
        )

    def _init_bottleneck(self, C: int, F: int, device: torch.device) -> None:
        """Create the BiLSTM and SpectralAttention once we know the compressed spatial dimensions"""
        self._bottleneck = BiLSTMBottleneck(
            n_channels=C,
            n_freq=F,
            hidden=self._lstm_hidden,
            n_layers=self._lstm_layers,
        ).to(device)
        self._attention = SpectralAttention(
            n_channels=C,
            n_freq=F,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_freq_bins, n_frames)  - noisy log-magnitude spectrogram

        Returns:
            mask: (batch, 1, n_freq_bins, n_frames)  - values in [0, 1]
        """
        skips = []

        # Encoder
        for encoder in self.encoders:
            skips.append(x)   # save feature map for skip connection
            x = encoder(x)

        # Bottleneck
        B, C, F, T = x.shape
        if self._bottleneck is None:
            self._init_bottleneck(C, F, x.device)

        assert self._bottleneck is not None
        assert self._attention is not None
        x = self._bottleneck(x)
        x = self._attention(x)   # channel + frequency attention gates

        # Decoder (reverse skip order)
        for decoder, skip in zip(self.decoders, reversed(skips[1:])):
            # Crop skip to match x if sizes differ (can happen with odd dims)
            skip = _crop_to_match(skip, x)
            x    = decoder(x, skip)

        # Final mask
        mask = self.mask_conv(x)

        # Resize mask to exactly match input spatial dims in case of rounding
        mask = _interpolate_to(mask, skips[0].shape[-2], skips[0].shape[-1])

        return mask



def _crop_to_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Centre-crop source so its (H, W) matches target
    Needed because transposed convolutions can produce one extra row/column
    """
    _, _, Hs, Ws = source.shape
    _, _, Ht, Wt = target.shape
    dh = (Hs - Ht) // 2
    dw = (Ws - Wt) // 2
    return source[:, :, dh:dh + Ht, dw:dw + Wt]


def _interpolate_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear resize to exactly (h, w) - final safety clamp"""
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model  = UNetDenoiser()
    dummy  = torch.randn(2, 1, 257, 128)   # batch=2, 257 freq bins, 128 frames
    mask   = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {mask.shape}")
    print(f"Mask range  : [{mask.min():.3f}, {mask.max():.3f}]")
    total  = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total:,}")