"""Strict CPU-first loader for the packaged V2-A conditional EMA-60k endpoint."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.inference import _load_tokenizer, resolve_device, sha256_file
from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
from tiny_avatar_diffusion.v2_native64 import make_v2_model, parameter_count
from tiny_avatar_diffusion.release_paths import release_root

ROOT = release_root()
CHECKPOINT = ROOT / "artifacts/checkpoints/ddpm-c64-ema-60k.pt"
SOURCE_SHA256 = "f47132dc9e247861c6697e41b5d352ccfa347ba6a19e48dcee7f061206cb0f14"
PARAMETERS = {"denoiser": 5127427, "text_encoder": 102976, "total": 5230403}

@dataclass(frozen=True)
class Bundle:
    model: torch.nn.Module; text_encoder: TextConditioningEncoder; tokenizer: object
    schedule: LinearDDPMSchedule; device: torch.device; requested_device: str; checkpoint_sha256: str

def load_v2a_conditional_ema60(device: str = "auto", *, expected_sha256: str | None = None) -> Bundle:
    if not CHECKPOINT.is_file(): raise FileNotFoundError(f"DDPM-C64 EMA-60k package is missing: {CHECKPOINT}")
    digest = sha256_file(CHECKPOINT)
    if expected_sha256 is not None and digest != expected_sha256: raise ValueError("DDPM-C64 EMA-60k package SHA-256 mismatch")
    artifact = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    required = {"artifact_format_version", "artifact_type", "model_role", "weight_variant", "selected_step", "source_checkpoint_path", "source_checkpoint_sha256", "architecture", "expected_parameter_count", "expected_denoiser_parameter_count", "expected_text_parameter_count", "model_state_dict", "text_state_dict"}
    if not isinstance(artifact, dict) or not required.issubset(artifact) or artifact["artifact_format_version"] != 1 or artifact["artifact_type"] != "v2a_conditional_ema_inference" or artifact["model_role"] != "v2a_conditional_60k" or artifact["weight_variant"] != "EMA" or artifact["selected_step"] != 60000 or artifact["source_checkpoint_sha256"] != SOURCE_SHA256 or artifact["architecture"] != {"variant":"v2_a","resolution":64,"text_cross_attention":["CA8","CA16"],"text_encoder":"TextConditioningEncoder"}:
        raise ValueError("V2-A EMA-60k package identity mismatch")
    _, model = make_v2_model("v2_a"); text = TextConditioningEncoder(TextConditioningConfig())
    model.load_state_dict(artifact["model_state_dict"], strict=True); text.load_state_dict(artifact["text_state_dict"], strict=True)
    if parameter_count(model, text) != PARAMETERS: raise ValueError("V2-A EMA-60k parameter identity mismatch")
    resolved = resolve_device(device); model.to(resolved).eval(); text.to(resolved).eval(); schedule = LinearDDPMSchedule().to(resolved).eval()
    return Bundle(model, text, _load_tokenizer(), schedule, resolved, device, digest)
