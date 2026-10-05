"""Precomputed coefficients for fixed DDPM forward processes.

Python indices ``t = 0, ..., T - 1`` represent mathematical DDPM steps
``1, ..., T``.  Accordingly, ``alphas_cumprod_prev[0]`` is exactly one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

COSINE_OFFSET = 0.008
COSINE_MAX_BETA = 0.999


@dataclass(frozen=True)
class SamplingSnapshot:
    """Detached CPU copy of a predicted ``x_0`` estimate at one timestep."""

    timestep: int
    predicted_x_start: torch.Tensor


@dataclass(frozen=True)
class ObservedSample:
    """Final sample plus requested observational predicted-``x_0`` snapshots."""

    sample: torch.Tensor
    snapshots: tuple[SamplingSnapshot, ...]


def normalize_snapshot_timesteps(
    timesteps: Iterable[int] | None, num_timesteps: int
) -> tuple[int, ...]:
    """Validate and normalize requested observations into sampling order."""
    if timesteps is None:
        return ()
    values = tuple(timesteps)
    if any(type(value) is not int for value in values):
        raise TypeError("snapshot timesteps must be integers")
    if any(value < 0 or value >= num_timesteps for value in values):
        raise ValueError(f"snapshot timesteps must be in [0, {num_timesteps})")
    return tuple(sorted(set(values), reverse=True))


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


def _register_ddpm_buffers(schedule: nn.Module, betas: torch.Tensor) -> None:
    """Derive and register the DDPM coefficients shared by all fixed schedules."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        (torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1])
    )

    schedule.register_buffer("betas", betas)
    schedule.register_buffer("alphas", alphas)
    schedule.register_buffer("alphas_cumprod", alphas_cumprod)
    schedule.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
    schedule.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    schedule.register_buffer(
        "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )
    schedule.register_buffer("sqrt_recip_alphas", torch.rsqrt(alphas))

    denominator = 1.0 - alphas_cumprod
    schedule.register_buffer(
        "posterior_variance",
        betas * (1.0 - alphas_cumprod_prev) / denominator,
    )
    schedule.register_buffer(
        "posterior_mean_coef1",
        betas * torch.sqrt(alphas_cumprod_prev) / denominator,
    )
    schedule.register_buffer(
        "posterior_mean_coef2",
        torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / denominator,
    )


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
        _register_ddpm_buffers(self, betas)

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

    def predict_x_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct ``x_0`` from ``x_t`` and the forward-process noise."""
        if noise.shape != x_t.shape:
            raise ValueError("noise must have the same shape as x_t")

        sqrt_alphas_cumprod = _extract(self.sqrt_alphas_cumprod, timesteps, x_t)
        sqrt_one_minus_alphas_cumprod = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t
        )
        return (x_t - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod

    def q_posterior_mean(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the DDPM posterior mean ``q(x_{t-1} | x_t, x_0)``."""
        if x_start.shape != x_t.shape:
            raise ValueError("x_start and x_t must have the same shape")

        coef1 = _extract(self.posterior_mean_coef1, timesteps, x_t)
        coef2 = _extract(self.posterior_mean_coef2, timesteps, x_t)
        return coef1 * x_start + coef2 * x_t

    def p_mean_variance_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
        *,
        clip_denoised: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compose reverse-process quantities from predicted noise.

        When requested, clip only the reconstructed ``x_0`` before using it
        in the posterior mean. The predicted epsilon remains unchanged.
        """
        predicted_x_start = self.predict_x_start_from_noise(
            x_t, timesteps, predicted_noise
        )
        if clip_denoised:
            predicted_x_start = predicted_x_start.clamp(-1.0, 1.0)
        model_mean = self.q_posterior_mean(predicted_x_start, x_t, timesteps)
        posterior_variance = _extract(self.posterior_variance, timesteps, x_t)
        return {
            "model_mean": model_mean,
            "posterior_variance": posterior_variance,
            "predicted_x_start": predicted_x_start,
        }

    def p_sample_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Take one DDPM reverse step from an already predicted noise tensor."""
        if noise is None:
            noise = torch.randn_like(x_t)
        elif noise.shape != x_t.shape:
            raise ValueError("noise must have the same shape as x_t")

        mean_variance = self.p_mean_variance_from_noise(
            x_t, timesteps, predicted_noise, clip_denoised=clip_denoised
        )
        nonzero_mask = (timesteps != 0).to(device=x_t.device, dtype=x_t.dtype)
        nonzero_mask = nonzero_mask.reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        sample = (
            mean_variance["model_mean"]
            + nonzero_mask * torch.sqrt(mean_variance["posterior_variance"]) * noise
        )
        return {"sample": sample, **mean_variance}

    def sample_loop_from_noise_predictor(
        self,
        x_T: torch.Tensor,
        noise_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """Run the complete DDPM reverse process from a provided ``x_T`` tensor."""
        if not isinstance(x_T, torch.Tensor):
            raise TypeError("x_T must be a torch.Tensor")
        if x_T.ndim < 1:
            raise ValueError("x_T must have a batch dimension")
        if x_T.shape[0] == 0:
            raise ValueError("x_T must have a non-empty batch dimension")
        if not callable(noise_predictor):
            raise TypeError("noise_predictor must be callable")

        x_t = x_T
        with torch.no_grad():
            for timestep in range(self.betas.shape[0] - 1, -1, -1):
                timesteps = torch.full(
                    (x_t.shape[0],),
                    timestep,
                    dtype=torch.long,
                    device=x_t.device,
                )
                predicted_noise = noise_predictor(x_t, timesteps)
                if not isinstance(predicted_noise, torch.Tensor):
                    raise TypeError("noise_predictor must return a torch.Tensor")
                if predicted_noise.shape != x_t.shape:
                    raise ValueError(
                        "noise_predictor output must have the same shape as x_t"
                    )

                x_t = self.p_sample_from_noise(
                    x_t,
                    timesteps,
                    predicted_noise,
                    clip_denoised=clip_denoised,
                )["sample"]

        return x_t


def unconditional_sample(
    schedule: LinearDDPMSchedule,
    model: nn.Module,
    *,
    x_T: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    image_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    clip_denoised: bool = True,
    snapshot_timesteps: Iterable[int] | None = None,
) -> torch.Tensor | ObservedSample:
    """Run the canonical unconditional ancestral DDPM sampler.

    This is the reusable form of the unconditional reverse loop used by the
    frozen DR05 evaluation scripts. Initial and reverse noise are drawn in
    FP32 from the caller-provided generator, and only reconstructed ``x_0`` is
    clipped by the existing reverse-step primitive.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have parameters") from error
    if len(image_shape) != 4 or image_shape[0] <= 0 or image_shape[1:] != (3, 32, 32):
        raise ValueError("image_shape must be [B, 3, 32, 32] with a non-empty batch")
    if x_T is None:
        x_t = torch.randn(
            image_shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    else:
        if x_T.shape != image_shape:
            raise ValueError("x_T must match image_shape")
        if x_T.device != device:
            raise ValueError("x_T must be on the model device")
        if x_T.dtype != torch.float32:
            raise ValueError("x_T must have dtype torch.float32")
        x_t = x_T

    requested_snapshots = normalize_snapshot_timesteps(
        snapshot_timesteps, schedule.betas.shape[0]
    )
    requested_set = set(requested_snapshots)
    snapshots: list[SamplingSnapshot] = []
    with torch.inference_mode():
        for timestep in range(schedule.betas.shape[0] - 1, -1, -1):
            timesteps = torch.full(
                (x_t.shape[0],), timestep, dtype=torch.long, device=device
            )
            predicted_noise = model(x_t, timesteps)
            reverse_noise = torch.randn(
                x_t.shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            step = schedule.p_sample_from_noise(
                x_t,
                timesteps,
                predicted_noise,
                noise=reverse_noise,
                clip_denoised=clip_denoised,
            )
            if timestep in requested_set:
                with torch.inference_mode(False):
                    predicted_x_start = (
                        step["predicted_x_start"].detach().clone().cpu()
                    )
                snapshots.append(
                    SamplingSnapshot(
                        timestep=timestep,
                        predicted_x_start=predicted_x_start,
                    )
                )
            x_t = step["sample"]
    if not requested_snapshots:
        return x_t
    return ObservedSample(sample=x_t, snapshots=tuple(snapshots))


class CosineDDPMSchedule(LinearDDPMSchedule):
    """Use the frozen Improved DDPM cosine beta construction.

    The Python timestep convention and every derived DDPM operation are
    inherited unchanged. Only the initial one-dimensional ``betas`` vector
    differs from :class:`LinearDDPMSchedule`.
    """

    def __init__(self, num_timesteps: int = 1_000) -> None:
        nn.Module.__init__(self)
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

        def alpha_bar(time: float) -> float:
            return math.cos(
                ((time + COSINE_OFFSET) / (1.0 + COSINE_OFFSET)) * math.pi / 2.0
            ) ** 2

        betas = torch.tensor(
            [
                min(
                    1.0
                    - alpha_bar((index + 1) / num_timesteps)
                    / alpha_bar(index / num_timesteps),
                    COSINE_MAX_BETA,
                )
                for index in range(num_timesteps)
            ],
            dtype=torch.float32,
        )
        _register_ddpm_buffers(self, betas)
