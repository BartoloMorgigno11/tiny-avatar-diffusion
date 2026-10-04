"""Canonical DR05-D11 parameter-count evidence helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

try:
    from scripts.dr05_materialize_canonical_checkpoints import (
        _conditional_models_from_artifact,
        _evaluator_from_artifact,
        _unconditional_model_from_artifact,
    )
except ModuleNotFoundError:
    from dr05_materialize_canonical_checkpoints import (  # type: ignore[no-redef]
        _conditional_models_from_artifact,
        _evaluator_from_artifact,
        _unconditional_model_from_artifact,
    )

PROTOCOL_RELATIVE_PATH = "data/processed/dr05_final_eval_v1/dr05_final_protocol.json"
OUTPUT_RELATIVE_PATH = "results/runs/dr05_final_efficiency_v1/parameter_count_summary.json"
EXPECTED_HASHES = {
    PROTOCOL_RELATIVE_PATH: "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f",
    "artifacts/checkpoints/conditional_main_ema_step20000.pt": "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb",
    "artifacts/checkpoints/unconditional_main_ema_step20000.pt": "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e",
    "artifacts/checkpoints/dr04_evaluator_epoch41.pt": "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc",
}
EXPECTED_COUNTS = {
    "unconditional_backbone": 4_595_779,
    "cross_attention": 25_152,
    "conditional_unet": 4_620_931,
    "text_encoder": 102_976,
    "full_conditional": 4_723_907,
    "unconditional": 4_595_779,
    "evaluator": 293_064,
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest for a required frozen input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_expected_hash(path: Path, expected_sha256: str) -> str:
    """Fail closed unless ``path`` has the exact expected digest."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"unexpected SHA256 for frozen input: {path}")
    return actual_sha256


def parameter_counts(modules: Iterable[nn.Module]) -> dict[str, int]:
    """Count unique parameter objects across modules, preserving requires-grad state."""
    seen: set[int] = set()
    total = 0
    trainable = 0
    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return {"total_parameters": total, "trainable_parameters": trainable}


def _conditional_backbone_counts(model: nn.Module) -> dict[str, int]:
    """Count the conditional U-Net excluding its direct cross-attention child."""
    attention_prefix = "image_text_cross_attention."
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(attention_prefix)
    ]
    return parameter_counts_from_parameters(parameters)


def parameter_counts_from_parameters(parameters: Iterable[nn.Parameter]) -> dict[str, int]:
    """Count a parameter iterable once per object for component-level audits."""
    seen: set[int] = set()
    total = 0
    trainable = 0
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"total_parameters": total, "trainable_parameters": trainable}


def _assert_counts(actual: dict[str, int], *, expected: int, label: str) -> None:
    if actual["total_parameters"] != expected or actual["trainable_parameters"] != expected:
        raise ValueError(f"unexpected {label} parameter counts: {actual}")


def build_parameter_summary(root: Path) -> dict[str, Any]:
    """Reconstruct frozen architectures on CPU and return canonical D11 evidence."""
    hashes = {
        relative: validate_expected_hash(root / relative, expected)
        for relative, expected in EXPECTED_HASHES.items()
    }
    conditional_path = root / "artifacts/checkpoints/conditional_main_ema_step20000.pt"
    unconditional_path = root / "artifacts/checkpoints/unconditional_main_ema_step20000.pt"
    evaluator_path = root / "artifacts/checkpoints/dr04_evaluator_epoch41.pt"

    conditional_artifact = torch.load(conditional_path, map_location="cpu", weights_only=False)
    unconditional_artifact = torch.load(unconditional_path, map_location="cpu", weights_only=False)
    evaluator_artifact = torch.load(evaluator_path, map_location="cpu", weights_only=False)
    text_encoder, conditional_unet = _conditional_models_from_artifact(conditional_artifact)
    unconditional_unet = _unconditional_model_from_artifact(unconditional_artifact)
    evaluator = _evaluator_from_artifact(evaluator_artifact)

    unconditional_backbone = _conditional_backbone_counts(conditional_unet)
    cross_attention = parameter_counts([conditional_unet.image_text_cross_attention])
    conditional_total = parameter_counts([conditional_unet])
    text_total = parameter_counts([text_encoder])
    full_conditional = parameter_counts([conditional_unet, text_encoder])
    unconditional_total = parameter_counts([unconditional_unet])
    evaluator_total = parameter_counts([evaluator])

    for label, count, expected_key in (
        ("unconditional backbone", unconditional_backbone, "unconditional_backbone"),
        ("cross attention", cross_attention, "cross_attention"),
        ("conditional U-Net", conditional_total, "conditional_unet"),
        ("text encoder", text_total, "text_encoder"),
        ("full conditional generator", full_conditional, "full_conditional"),
        ("unconditional generator", unconditional_total, "unconditional"),
        ("evaluator", evaluator_total, "evaluator"),
    ):
        _assert_counts(count, expected=EXPECTED_COUNTS[expected_key], label=label)

    equations = {
        "unconditional_backbone_plus_cross_attention_equals_conditional_unet": {
            "left": unconditional_backbone["total_parameters"] + cross_attention["total_parameters"],
            "right": conditional_total["total_parameters"],
        },
        "conditional_unet_plus_text_encoder_equals_full_conditional": {
            "left": conditional_total["total_parameters"] + text_total["total_parameters"],
            "right": full_conditional["total_parameters"],
        },
        "unconditional_backbone_plus_cross_attention_plus_text_encoder_equals_full_conditional": {
            "left": unconditional_backbone["total_parameters"] + cross_attention["total_parameters"] + text_total["total_parameters"],
            "right": full_conditional["total_parameters"],
        },
        "conditional_backbone_matches_unconditional_checkpoint": {
            "left": unconditional_backbone["total_parameters"],
            "right": unconditional_total["total_parameters"],
        },
    }
    for equation in equations.values():
        equation["residual"] = equation["left"] - equation["right"]
        equation["pass"] = equation["residual"] == 0
    if not all(item["pass"] for item in equations.values()):
        raise ValueError("anti-double-count equation mismatch")

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "schema": "tiny_avatar_diffusion.dr05_parameter_count_summary",
        "version": 1,
        "classification": "D11 parameter-count evidence",
        "git_head": git_head,
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": hashes[PROTOCOL_RELATIVE_PATH]},
        "checkpoint_provenance": {
            "conditional_generator": {"path": conditional_path.relative_to(root).as_posix(), "sha256": hashes[conditional_path.relative_to(root).as_posix()]},
            "unconditional_generator": {"path": unconditional_path.relative_to(root).as_posix(), "sha256": hashes[unconditional_path.relative_to(root).as_posix()]},
            "evaluation_instrument": {"path": evaluator_path.relative_to(root).as_posix(), "sha256": hashes[evaluator_path.relative_to(root).as_posix()]},
        },
        "counting_method": {
            "total_parameters": "sum(p.numel() for p in module.parameters())",
            "trainable_parameters": "sum(p.numel() for p in module.parameters() if p.requires_grad)",
            "shared_parameter_policy": "unique parameter objects are counted once across an explicit component set",
            "architecture_loading": "existing repository reconstruction helpers with strict PyTorch load_state_dict compatibility",
            "device": "cpu",
            "forward_inference_performed": False,
        },
        "conditional": {
            "unconditional_backbone": unconditional_backbone,
            "cross_attention": cross_attention,
            "conditional_unet": conditional_total,
            "text_encoder": text_total,
            "full_system": full_conditional,
        },
        "unconditional": unconditional_total,
        "evaluation_instrument": {**evaluator_total, "explicitly_excluded_from_generator_counts": True},
        "anti_double_count": {
            "rule": "The conditional generator is conditional U-Net plus text encoder; the conditional U-Net already contains the unconditional backbone and cross-attention, so those components must not be added again.",
            "equations": equations,
            "evaluator_excluded_from_generator_counts": True,
            "pass": True,
        },
    }


def write_parameter_summary(root: Path, summary: dict[str, Any]) -> Path:
    """Write the one canonical D11 artifact without ever replacing an existing file."""
    output = root / OUTPUT_RELATIVE_PATH
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canonical parameter artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return output
