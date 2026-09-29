"""Precomputed coefficients for the frozen linear DDPM forward process.

Python indices ``t = 0, ..., T - 1`` represent mathematical DDPM steps
``1, ..., T``.  Accordingly, ``alphas_cumprod_prev[0]`` is exactly one.
"""

from __future__ import annotations

import torch
from torch import nn


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
