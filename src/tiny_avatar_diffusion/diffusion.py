"""Precomputed coefficients for the frozen linear DDPM forward process.

Python indices ``t = 0, ..., T - 1`` represent mathematical DDPM steps
``1, ..., T``.  Accordingly, ``alphas_cumprod_prev[0]`` is exactly one.
"""

from __future__ import annotations

import torch
from torch import nn


def _extract(coefficients: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Select per-sample coefficients and reshape them for batch-first broadcasting."""
    if x.ndim < 1:
        raise ValueError("x must have a batch dimension")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional")
    if timesteps.shape[0] != x.shape[0]:
        raise ValueError("timesteps must have one value per batch element")
    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must have dtype torch.long")
    if torch.any(timesteps < 0) or torch.any(timesteps >= coefficients.shape[0]):
        raise ValueError("timesteps must be within the schedule range")

    values = coefficients.to(device=x.device, dtype=x.dtype).gather(
        0, timesteps.to(device=x.device)
    )
    return values.reshape(x.shape[0], *((1,) * (x.ndim - 1)))


class LinearDDPMSchedule(nn.Module):
    """Store the fixed linear DDPM schedule and its derived coefficients.

    The tensors are registered buffers, so they do not require gradients and
    move with this module when it is moved to a model device.
    """

    def __init__(
        self,
        num_timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")

        betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1])
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.rsqrt(alphas))

        denominator = 1.0 - alphas_cumprod
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / denominator,
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / denominator,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / denominator,
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` from ``q(x_t | x_0)`` for one timestep per sample."""
        if noise is None:
            noise = torch.randn_like(x_start)
        elif noise.shape != x_start.shape:
            raise ValueError("noise must have the same shape as x_start")

        sqrt_alphas_cumprod = _extract(self.sqrt_alphas_cumprod, timesteps, x_start)
        sqrt_one_minus_alphas_cumprod = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start
        )
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
