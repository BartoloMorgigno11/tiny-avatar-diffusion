"""Neural building blocks for the frozen Tiny U-Net V1."""

from __future__ import annotations

import torch
from torch import nn


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
