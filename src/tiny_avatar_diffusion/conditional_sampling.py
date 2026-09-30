"""Conditional DR03 wrapper around the frozen DR02 DDPM reverse primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise ValueError("module must have parameters") from error


def conditional_sample(
    schedule: LinearDDPMSchedule,
    text_encoder: TextConditioningEncoder,
    conditional_unet: ConditionalTinyUNet,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    x_T: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    image_shape: tuple[int, int, int, int] | None = None,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """Sample once per prompt using the unchanged DR02 reverse-step mathematics.

    The text encoder runs once, then its contextual tokens remain fixed for the
    whole chain. Reverse noise is supplied explicitly to the existing
    ``p_sample_from_noise`` primitive, allowing a caller-provided generator.
    The main-facing default clips only reconstructed ``x_0`` through the DR02
    reverse primitive; it never clips epsilon, ``x_t``, or reverse noise.
    """
    if input_ids is None:
        raise ValueError("input_ids are required for conditional sampling")
    if attention_mask is None:
        raise ValueError("attention_mask is required for conditional sampling")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    text_device = _module_device(text_encoder)
    denoiser_device = _module_device(conditional_unet)
    if text_device != denoiser_device:
        raise ValueError("text_encoder and conditional_unet must be on the same device")
    batch_size = input_ids.shape[0]
    if x_T is None:
        shape = image_shape or (batch_size, 3, 32, 32)
        if shape != (batch_size, 3, 32, 32):
            raise ValueError("image_shape must be [B, 3, 32, 32] matching input_ids")
        x_t = torch.randn(shape, device=denoiser_device, dtype=torch.float32, generator=generator)
    else:
        if x_T.shape != (batch_size, 3, 32, 32):
            raise ValueError("x_T must have shape [B, 3, 32, 32] matching input_ids")
        if x_T.device != denoiser_device:
            raise ValueError("x_T must be on the conditional_unet device")
        x_t = x_T

    text_was_training = text_encoder.training
    denoiser_was_training = conditional_unet.training
    text_encoder.eval()
    conditional_unet.eval()
    try:
        with torch.no_grad():
            input_ids = input_ids.to(denoiser_device)
            attention_mask = attention_mask.to(denoiser_device)
            context = text_encoder(input_ids, attention_mask)
            for timestep in range(schedule.betas.shape[0] - 1, -1, -1):
                timesteps = torch.full(
                    (batch_size,), timestep, dtype=torch.long, device=denoiser_device
                )
                predicted_noise = conditional_unet(x_t, timesteps, context, attention_mask)
                reverse_noise = torch.randn(
                    x_t.shape, device=denoiser_device, dtype=x_t.dtype, generator=generator
                )
                x_t = schedule.p_sample_from_noise(
                    x_t,
                    timesteps,
                    predicted_noise,
                    noise=reverse_noise,
                    clip_denoised=clip_denoised,
                )["sample"]
    finally:
        text_encoder.train(text_was_training)
        conditional_unet.train(denoiser_was_training)
    return x_t


def load_conditional_checkpoint(
    checkpoint_path: Path,
    text_encoder: TextConditioningEncoder,
    conditional_unet: ConditionalTinyUNet,
    *,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load the existing DR03 LR-pilot final-state format without altering it."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Conditional checkpoint must be a dictionary")
    try:
        text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        conditional_unet.load_state_dict(checkpoint["conditional_unet_state_dict"])
    except KeyError as error:
        raise ValueError("Checkpoint does not use the DR03 LR-pilot state format") from error
    return checkpoint
