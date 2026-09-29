"""Neural building blocks for the frozen Tiny U-Net V1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return fixed sinusoidal embeddings for a batch of DDPM timesteps."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional")
    if embedding_dim <= 0 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be a positive even integer")

    output_dtype = torch.get_default_dtype() if dtype is None else dtype
    half = embedding_dim // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0, device=timesteps.device, dtype=output_dtype))
        * (torch.arange(half, device=timesteps.device, dtype=output_dtype) / half)
    )
    arguments = timesteps.to(dtype=output_dtype)[:, None] * frequencies[None]
    return torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=-1)


class TimeEmbedding(nn.Module):
    """Map a DDPM timestep batch to the Tiny U-Net V1 global time embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Produce a processed ``[B, 256]`` embedding from timesteps ``[B]``."""
        embedding = sinusoidal_timestep_embedding(
            timesteps,
            embedding_dim=64,
            dtype=self.mlp[0].weight.dtype,
        )
        return self.mlp(embedding)


class ResBlock(nn.Module):
    """Time-conditioned residual block for the frozen Tiny U-Net V1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embedding_dim: int = 256,
        num_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError("channels must be divisible by num_groups")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.in_channels = in_channels
        self.time_embedding_dim = time_embedding_dim
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Apply the residual path while preserving the spatial resolution."""
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W]")
        if x.shape[1] != self.in_channels:
            raise ValueError("x has an incompatible channel count")
        if time_emb.ndim != 2:
            raise ValueError("time_emb must have shape [B, time_embedding_dim]")
        if time_emb.shape[0] != x.shape[0]:
            raise ValueError("x and time_emb must have the same batch size")
        if time_emb.shape[1] != self.time_embedding_dim:
            raise ValueError("time_emb has an incompatible embedding dimension")

        h = self.conv1(self.activation1(self.norm1(x)))
        projected_time = self.time_projection(time_emb)[:, :, None, None]
        h = h + projected_time
        h = self.conv2(self.dropout(self.activation2(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Halve spatial resolution while preserving the channel count."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen stride-two convolution."""
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W]")
        if x.shape[1] != self.channels:
            raise ValueError("x has an incompatible channel count")
        return self.conv(x)


class Upsample(nn.Module):
    """Double spatial resolution by nearest interpolation followed by convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nearest-neighbor resize followed by the frozen convolution."""
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W]")
        if x.shape[1] != self.channels:
            raise ValueError("x has an incompatible channel count")
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class TinyUNet(nn.Module):
    """Frozen unconditional Tiny U-Net V1 for pixel-space epsilon prediction."""

    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding()
        self.input_projection = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.encoder32_block1 = ResBlock(64, 64, dropout=dropout)
        self.encoder32_block2 = ResBlock(64, 64, dropout=dropout)
        self.downsample32 = Downsample(64)

        self.encoder16_block1 = ResBlock(64, 128, dropout=dropout)
        self.encoder16_block2 = ResBlock(128, 128, dropout=dropout)
        self.downsample16 = Downsample(128)

        self.encoder8_block1 = ResBlock(128, 128, dropout=dropout)
        self.encoder8_block2 = ResBlock(128, 128, dropout=dropout)

        self.bottleneck_block1 = ResBlock(128, 128, dropout=dropout)
        self.bottleneck_block2 = ResBlock(128, 128, dropout=dropout)

        self.decoder8_block1 = ResBlock(256, 128, dropout=dropout)
        self.decoder8_block2 = ResBlock(128, 128, dropout=dropout)
        self.upsample8 = Upsample(128)

        self.decoder16_block1 = ResBlock(256, 128, dropout=dropout)
        self.decoder16_block2 = ResBlock(128, 128, dropout=dropout)
        self.upsample16 = Upsample(128)

        self.decoder32_block1 = ResBlock(192, 64, dropout=dropout)
        self.decoder32_block2 = ResBlock(64, 64, dropout=dropout)

        self.output_norm = nn.GroupNorm(32, 64)
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict epsilon for a batch of noisy RGB images."""
        if x_t.ndim != 4:
            raise ValueError("x_t must have shape [B, 3, H, W]")
        if x_t.shape[1] != 3:
            raise ValueError("x_t must have three channels")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be one-dimensional")
        if timesteps.shape[0] != x_t.shape[0]:
            raise ValueError("x_t and timesteps must have the same batch size")
        if x_t.shape[2] % 4 != 0 or x_t.shape[3] % 4 != 0:
            raise ValueError("spatial dimensions must be divisible by four")

        time_emb = self.time_embedding(timesteps)
        h = self.input_projection(x_t)

        h = self.encoder32_block1(h, time_emb)
        s32 = self.encoder32_block2(h, time_emb)
        h = self.downsample32(s32)

        h = self.encoder16_block1(h, time_emb)
        s16 = self.encoder16_block2(h, time_emb)
        h = self.downsample16(s16)

        h = self.encoder8_block1(h, time_emb)
        s8 = self.encoder8_block2(h, time_emb)

        h = self.bottleneck_block1(s8, time_emb)
        h = self.bottleneck_block2(h, time_emb)

        h = torch.cat((h, s8), dim=1)
        h = self.decoder8_block1(h, time_emb)
        h = self.decoder8_block2(h, time_emb)
        h = self.upsample8(h)

        h = torch.cat((h, s16), dim=1)
        h = self.decoder16_block1(h, time_emb)
        h = self.decoder16_block2(h, time_emb)
        h = self.upsample16(h)

        h = torch.cat((h, s32), dim=1)
        h = self.decoder32_block1(h, time_emb)
        h = self.decoder32_block2(h, time_emb)

        return self.output_projection(self.output_activation(self.output_norm(h)))
