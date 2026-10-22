"""Strict external-artifact inference support for authentic V2 native64 models.

The large V2 training checkpoints are intentionally not package artifacts.  This
module accepts only their authenticated local paths and refuses a mismatched,
resumed, or architecture-incompatible checkpoint before it reaches a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer
from tiny_avatar_diffusion.v2_native64 import V2Native64ConditionalUNet, make_v2_model, parameter_count

V2_CHECKPOINT_SCHEMA = "tiny_avatar_diffusion.v2_authentic64_training_v1"
V2_IMAGE_SIZE = 64
V2_TIMESTEPS = 1_000
V2_VARIANTS = ("v2_a", "v2_b")
V2_CHECKPOINT_SHA256 = {
    "v2_a": "2e181d4bb23908ea544eacb291b80189a069dccfe20626e4505220c3ed9e3ff8",
    "v2_b": "b146e29df6d669e53bbb9e0ea9b20fdec9c75949160c6ffe06a6f65519c185fd",
}
V2_PARAMETER_COUNTS = {"v2_a": 5_230_403, "v2_b": 5_247_171}
V2_DENOISER_COUNTS = {"v2_a": 5_127_427, "v2_b": 5_144_195}
V2_TEXT_COUNT = 102_976
V2_TRAINING_CONFIG = {
    "batch_size": 8,
    "cache_sha256": "7776ca0bc4c14b3789f9f0a6491fefae1a30c8801637b65eaafeebdd9e4a25fd",
    "ema_decay": 0.9999,
    "lr": 0.0002,
    "resolution": 64,
    "seed": 20261017,
    "steps": 40_000,
    "train_manifest_sha256": "295f7c93766294243871ebb2bf742a9fdae2f5a716ac773b0ead16dc8e6eb014",
    "weight_decay": 0.0,
}


@dataclass(frozen=True)
class V2Native64InferenceBundle:
    variant: Literal["v2_a", "v2_b"]
    checkpoint_path: Path
    checkpoint_sha256: str
    model: V2Native64ConditionalUNet
    text_encoder: TextConditioningEncoder
    tokenizer: WhitespaceTokenizer
    schedule: LinearDDPMSchedule
    device: torch.device
    requested_device: str


def _validate_checkpoint_artifact(artifact: Any, variant: str) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError("V2 checkpoint must be a dictionary")
    if artifact.get("schema") != V2_CHECKPOINT_SCHEMA or artifact.get("step") != 40_000:
        raise ValueError("V2 checkpoint schema or selected training step mismatch")
    config = artifact.get("config")
    expected_config = {**V2_TRAINING_CONFIG, "variant": variant}
    if config != expected_config:
        raise ValueError("V2 checkpoint training configuration/provenance mismatch")
    ema = artifact.get("ema")
    if not isinstance(ema, dict) or set(ema) != {"model", "text"}:
        raise ValueError("V2 checkpoint must contain exactly EMA model and text states")
    for section in ("model", "text", "ema"):
        value = artifact.get(section)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"V2 checkpoint {section} state is missing or malformed")
    return artifact


def _strict_state_validation(artifact: dict[str, Any], variant: str) -> tuple[V2Native64ConditionalUNet, TextConditioningEncoder]:
    _config, model = make_v2_model(variant)
    text = TextConditioningEncoder(TextConditioningConfig())
    try:
        # Validate both resumable and selected EMA state shapes/keys.  Only EMA is served.
        model.load_state_dict(artifact["model"], strict=True)
        text.load_state_dict(artifact["text"], strict=True)
        model.load_state_dict(artifact["ema"]["model"], strict=True)
        text.load_state_dict(artifact["ema"]["text"], strict=True)
    except (RuntimeError, KeyError) as error:
        raise ValueError("V2 checkpoint state_dict is incompatible with its declared architecture") from error
    counts = parameter_count(model, text)
    if counts != {"denoiser": V2_DENOISER_COUNTS[variant], "text_encoder": V2_TEXT_COUNT, "total": V2_PARAMETER_COUNTS[variant]}:
        raise ValueError("V2 checkpoint architecture parameter-count identity mismatch")
    return model, text


def load_v2_native64_bundle(
    variant: Literal["v2_a", "v2_b"] | str,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> V2Native64InferenceBundle:
    """Load one exact V2 EMA checkpoint and retain its native 64px contract."""
    if variant not in V2_VARIANTS:
        raise ValueError("variant must be v2_a or v2_b")
    from tiny_avatar_diffusion.h7_release import H7_VOCABULARY_SHA256
    from tiny_avatar_diffusion.inference import VOCABULARY_PATH, _load_tokenizer, resolve_device, sha256_file

    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"V2 checkpoint is missing: {path}")
    digest = sha256_file(path)
    if digest != V2_CHECKPOINT_SHA256[variant]:
        raise ValueError("V2 checkpoint SHA256 mismatch for declared variant")
    # The digest above authenticates this local training checkpoint before unsafe
    # Python metadata is deserialized; it intentionally is not a package artifact.
    artifact = _validate_checkpoint_artifact(torch.load(path, map_location="cpu", weights_only=False), variant)
    model, text = _strict_state_validation(artifact, variant)
    if sha256_file(VOCABULARY_PATH) != H7_VOCABULARY_SHA256:
        raise ValueError("V2 tokenizer vocabulary SHA256 mismatch")
    schedule = LinearDDPMSchedule()
    expected_betas = torch.linspace(1e-4, 0.02, V2_TIMESTEPS, dtype=torch.float32)
    if schedule.betas.shape != (V2_TIMESTEPS,) or not torch.equal(schedule.betas.cpu(), expected_betas):
        raise ValueError("V2 DDPM scheduler configuration mismatch")
    resolved = resolve_device(device)
    model.to(resolved).eval()
    text.to(resolved).eval()
    schedule.to(resolved).eval()
    return V2Native64InferenceBundle(variant, path, digest, model, text, _load_tokenizer(), schedule, resolved, device)


def v2_tensor_to_rgb_image(sample: torch.Tensor) -> Image.Image:
    """Convert a native V2 tensor directly to native 64px RGB pixels."""
    if sample.shape != (3, V2_IMAGE_SIZE, V2_IMAGE_SIZE):
        raise ValueError("V2 sample must have shape [3,64,64]")
    pixels = (((sample.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy())
    return Image.fromarray(pixels, mode="RGB")


def generate_v2_native64(
    bundle: V2Native64InferenceBundle,
    *,
    caption: str,
    seed: int,
) -> Image.Image:
    """Run the authentic 1,000-step ancestral DDPM V2 native64 sampler."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    encoded = bundle.tokenizer.encode(caption)
    ids = torch.tensor([encoded.token_ids], device=bundle.device, dtype=torch.long)
    mask = torch.tensor([encoded.attention_mask], device=bundle.device, dtype=torch.long)
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    x_t = torch.randn((1, 3, 64, 64), device=bundle.device, dtype=torch.float32, generator=generator)
    with torch.inference_mode():
        context = bundle.text_encoder(ids, mask)
        for timestep in range(V2_TIMESTEPS - 1, -1, -1):
            steps = torch.full((1,), timestep, device=bundle.device, dtype=torch.long)
            epsilon = bundle.model(x_t, steps, context, mask)
            noise = torch.randn(x_t.shape, device=bundle.device, dtype=torch.float32, generator=generator)
            x_t = bundle.schedule.p_sample_from_noise(x_t, steps, epsilon, noise=noise, clip_denoised=True)["sample"]
    return v2_tensor_to_rgb_image(x_t[0])
