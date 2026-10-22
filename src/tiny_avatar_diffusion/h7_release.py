"""Packaging and strict identity checks for the selected H7 inference artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig

H7_SOURCE_SHA256 = "80eedea866a8f73b728b0d8bdc184910c814e93857d051f0cbb5a680ac6f10dd"
H7_SOURCE_RELATIVE_PATH = (
    "results/h7_ca16_40k_v1/h7_ca16_40k_v1_seed2026/terminal_40000.pt"
)
H7_VOCABULARY_RELATIVE_PATH = "data/processed/dr01_v1/text/10k/vocab.json"
H7_VOCABULARY_SHA256 = "15f2feb538e6e13389bf6ff113c48ed7a9663e0aefd8416dcd4c6fc760d7041a"
H7_PARAMETER_COUNT = 4_749_059
H7_DENOISER_PARAMETER_COUNT = 4_646_083
H7_TEXT_PARAMETER_COUNT = 102_976


def sha256_file(path: Path) -> str:
    """Return the complete SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h7_architecture() -> dict[str, Any]:
    """Immutable architecture identity expected by the public H7 loader."""
    return {
        "text_encoder": {
            "class": "TextConditioningEncoder",
            "config": asdict(TextConditioningConfig()),
        },
        "conditional_unet": {
            "class": "ConditionalTinyUNet",
            "dropout": 0.0,
            "use_decoder16_cross_attention": True,
        },
        "diffusion": {
            "image_size": [32, 32],
            "timesteps": 1000,
            "schedule": "linear",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "prediction_type": "epsilon",
            "sampler": "ancestral_ddpm",
            "clip_denoised": True,
        },
    }


def _load_vocab(path: Path) -> dict[str, Any]:
    if sha256_file(path) != H7_VOCABULARY_SHA256:
        raise ValueError("H7 vocabulary SHA256 mismatch")
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "tiny_avatar_diffusion.dr01_train_vocabulary",
        "version": "dr01_v1",
        "dataset_variant": "10k",
        "source_split": "TRAIN",
        "vocab_size": 45,
    }
    if any(vocabulary.get(key) != value for key, value in required.items()):
        raise ValueError("H7 vocabulary identity is incompatible")
    if not isinstance(vocabulary.get("token_to_id"), dict):
        raise ValueError("H7 vocabulary lacks token_to_id")
    return vocabulary


def package_h7_ema(
    *, source_checkpoint: Path, vocabulary_path: Path, destination: Path, source_git_commit: str
) -> dict[str, Any]:
    """Extract only verified H7 EMA inference state into a portable artifact."""
    if sha256_file(source_checkpoint) != H7_SOURCE_SHA256:
        raise ValueError("H7 source checkpoint SHA256 mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise TypeError("H7 source checkpoint must be a dictionary")
    if (
        source.get("schema") != "tiny_avatar_diffusion.h7_extended_training_v1"
        or source.get("run_id") != "h7_ca16_40k_v1_seed2026"
        or source.get("progress", {}).get("step") != 40_000
    ):
        raise ValueError("H7 source checkpoint identity is incompatible")
    ema = source.get("ema_state_dict")
    if not isinstance(ema, dict) or set(ema) != {"text_encoder", "conditional_unet"}:
        raise ValueError("H7 source checkpoint lacks complete EMA state")
    if not all(isinstance(value, dict) and value for value in ema.values()):
        raise ValueError("H7 EMA components must be non-empty state dictionaries")
    vocabulary = _load_vocab(vocabulary_path)
    artifact = {
        "artifact_format_version": 1,
        "artifact_type": "canonical_inference_checkpoint",
        "model_role": "h7_ca8_ca16_40k",
        "model_state_dict": {
            "text_encoder": ema["text_encoder"],
            "conditional_unet": ema["conditional_unet"],
        },
        "architecture": h7_architecture(),
        "source_worktree": "DL_Project_H7",
        "source_checkpoint_path": H7_SOURCE_RELATIVE_PATH,
        "source_checkpoint_sha256": H7_SOURCE_SHA256,
        "source_git_commit": source_git_commit,
        "expected_parameter_count": H7_PARAMETER_COUNT,
        "expected_denoiser_parameter_count": H7_DENOISER_PARAMETER_COUNT,
        "expected_text_parameter_count": H7_TEXT_PARAMETER_COUNT,
        "selected_step": 40_000,
        "weight_variant": "EMA",
        "vocabulary": vocabulary,
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "vocabulary_relative_path": H7_VOCABULARY_RELATIVE_PATH,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return {
        "source_checkpoint_sha256": H7_SOURCE_SHA256,
        "package_checkpoint_sha256": sha256_file(destination),
        "source_checkpoint_path": H7_SOURCE_RELATIVE_PATH,
        "package_checkpoint_path": destination.name,
        "selected_step": 40_000,
        "weight_variant": "EMA",
    }
