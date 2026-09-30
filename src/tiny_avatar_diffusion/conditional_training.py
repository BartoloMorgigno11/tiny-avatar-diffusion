"""Minimal joint conditional DDPM epsilon-prediction training helpers for DR03."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet


@dataclass(frozen=True)
class ConditionalDiffusionLoss:
    """Inspectable outputs from one conditional DDPM epsilon-loss evaluation."""

    loss: torch.Tensor
    predicted_noise: torch.Tensor
    noise: torch.Tensor
    noisy_images: torch.Tensor
    timesteps: torch.Tensor


def conditional_diffusion_loss(
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    schedule: LinearDDPMSchedule,
    x_start: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalDiffusionLoss:
    """Compute the sole DR03 objective: DDPM epsilon-prediction MSE.

    Explicit ``timesteps`` and ``noise`` are honored unchanged for deterministic
    tests and debug probes. Otherwise they use the existing DR02 conventions.
    """
    if x_start.ndim != 4:
        raise ValueError("x_start must have shape [B, C, H, W]")
    if input_ids.ndim != 2 or input_ids.shape[0] != x_start.shape[0]:
        raise ValueError("input_ids must have shape [B, L] matching x_start")
    if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    batch_size = x_start.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            0,
            schedule.betas.shape[0],
            (batch_size,),
            device=x_start.device,
            generator=generator,
        )
    if noise is None:
        noise = torch.randn(
            x_start.shape,
            device=x_start.device,
            dtype=x_start.dtype,
            generator=generator,
        )

    noisy_images = schedule.q_sample(x_start, timesteps, noise=noise)
    context = text_encoder(input_ids, attention_mask)
    predicted_noise = denoiser(noisy_images, timesteps, context, attention_mask)
    loss = F.mse_loss(predicted_noise, noise)
    return ConditionalDiffusionLoss(
        loss=loss,
        predicted_noise=predicted_noise,
        noise=noise,
        noisy_images=noisy_images,
        timesteps=timesteps,
    )


def conditional_training_step(
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    schedule: LinearDDPMSchedule,
    optimizer: torch.optim.Optimizer,
    x_start: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalDiffusionLoss:
    """Run one DEBUG-only joint optimizer step using the conditional DDPM loss."""
    text_encoder.train()
    denoiser.train()
    optimizer.zero_grad(set_to_none=True)
    result = conditional_diffusion_loss(
        text_encoder,
        denoiser,
        schedule,
        x_start,
        input_ids,
        attention_mask,
        timesteps=timesteps,
        noise=noise,
        generator=generator,
    )
    if not torch.isfinite(result.predicted_noise).all() or not torch.isfinite(result.loss):
        raise FloatingPointError("Conditional DDPM prediction or loss contains NaN or Inf")
    result.loss.backward()
    optimizer.step()
    return result
