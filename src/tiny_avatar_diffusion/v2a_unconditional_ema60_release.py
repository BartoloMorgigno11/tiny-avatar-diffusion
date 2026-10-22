"""Strict CPU-first adapter for the packaged V2-A unconditional EMA-60k."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.inference import resolve_device, sha256_file
from tiny_avatar_diffusion.v2a_unconditional_native64 import EXPECTED_PARAMETERS, V2AUnconditionalNative64UNet
from tiny_avatar_diffusion.release_paths import release_root

ROOT = release_root()
CHECKPOINT = ROOT / "artifacts/checkpoints/ddpm-u64-ema-60k.pt"
SOURCE_EMA_SHA256 = "0404caf40f5317d04a9972fbeb8759086e40ff9497a0596772c82b8031673015"


@dataclass(frozen=True)
class Bundle:
    model: V2AUnconditionalNative64UNet
    schedule: LinearDDPMSchedule
    device: torch.device
    requested_device: str
    checkpoint_sha256: str


def load_v2a_unconditional_ema60(device: str = "auto", *, expected_sha256: str | None = None) -> Bundle:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"DDPM-U64 EMA-60k package is missing: {CHECKPOINT}")
    digest = sha256_file(CHECKPOINT)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("DDPM-U64 EMA-60k package SHA-256 mismatch")
    artifact = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    required = {"artifact_format_version", "artifact_type", "model_role", "weight_variant", "selected_step", "parameter_count", "architecture", "source_ema_package_sha256", "model_state_dict"}
    if (not isinstance(artifact, dict) or not required.issubset(artifact) or artifact["artifact_format_version"] != 1 or artifact["artifact_type"] != "canonical_inference_checkpoint" or artifact["model_role"] != "v2a_unconditional_native64_60k" or artifact["weight_variant"] != "EMA" or artifact["selected_step"] != 60_000 or artifact["parameter_count"] != EXPECTED_PARAMETERS or artifact["architecture"] != "V2-A image backbone without text encoder/CA8/CA16" or artifact["source_ema_package_sha256"] != SOURCE_EMA_SHA256):
        raise ValueError("V2-A unconditional EMA-60k package identity mismatch")
    model = V2AUnconditionalNative64UNet()
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != EXPECTED_PARAMETERS:
        raise ValueError("V2-A unconditional EMA-60k parameter identity mismatch")
    resolved = resolve_device(device)
    model.to(resolved).eval()
    return Bundle(model, LinearDDPMSchedule().to(resolved).eval(), resolved, device, digest)


def generate_v2a_unconditional(bundle: Bundle, *, seed: int) -> Image.Image:
    """Native 64px ancestral DDPM, called only by explicit UI generation."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    with torch.inference_mode():
        x_t = torch.randn((1, 3, 64, 64), device=bundle.device, dtype=torch.float32, generator=generator)
        for value in range(bundle.schedule.betas.shape[0] - 1, -1, -1):
            timestep = torch.full((1,), value, dtype=torch.long, device=bundle.device)
            noise = torch.randn(x_t.shape, device=bundle.device, dtype=torch.float32, generator=generator)
            x_t = bundle.schedule.p_sample_from_noise(x_t, timestep, bundle.model(x_t, timestep), noise=noise, clip_denoised=True)["sample"]
    pixels = ((x_t[0].cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(pixels, mode="RGB")
