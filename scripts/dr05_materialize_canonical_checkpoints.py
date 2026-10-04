"""Materialize the frozen DR05 inference-only checkpoint artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.checkpoint_selection import load_ema_checkpoint
from tiny_avatar_diffusion.evaluator import (
    ARCHITECTURE_ID,
    CLASS_COUNTS,
    PREPROCESSING_ID,
    AttributeEvaluator,
    parameter_count,
)
from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
from tiny_avatar_diffusion.unconditional_main_training import (
    EXPECTED_PARAMETERS,
    UnconditionalMainConfig,
    canonical_checkpoint_ready,
)
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, TinyUNet
from tiny_avatar_diffusion.conditional_main_training import TrainingProgress

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "checkpoints"
CONDITIONAL_SOURCE = ROOT / "reports" / "dr03_main_training" / "dr03_conditional_main_10k_seed2026_r2" / "milestone_20000.pt"
EVALUATOR_SOURCE = ROOT / "results" / "runs" / "dr04_evaluator_main_100k_seed2026_r1" / "best.pt"
CONDITIONAL_OUTPUT = ARTIFACT_ROOT / "conditional_main_ema_step20000.pt"
EVALUATOR_OUTPUT = ARTIFACT_ROOT / "dr04_evaluator_epoch41.pt"
UNCONDITIONAL_SOURCE = ROOT / "results" / "runs" / "dr05_unconditional_main_10k_seed2026_r1" / "milestone_20000.pt"
UNCONDITIONAL_OUTPUT = ARTIFACT_ROOT / "unconditional_main_ema_step20000.pt"
EXECUTION_BINDINGS = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json"
CONDITIONAL_SOURCE_SHA256 = "0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884"
EVALUATOR_SOURCE_SHA256 = "3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f"
CONDITIONAL_PARAMETER_COUNT = 4_723_907
EVALUATOR_PARAMETER_COUNT = 293_064
UNCONDITIONAL_SOURCE_SHA256 = "a23436b8f3713d87374ef3ad847aadeffd38469fff89bcb169424539c9842129"
UNCONDITIONAL_RUN_ID = "dr05_unconditional_main_10k_seed2026_r1"
UNCONDITIONAL_PARAMETER_COUNT = EXPECTED_PARAMETERS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parameter_total(*models: torch.nn.Module) -> int:
    return sum(parameter.numel() for model in models for parameter in model.parameters())


def assert_state_dict_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    if expected.keys() != actual.keys():
        raise ValueError("state-dict key mismatch")
    for key in expected:
        left, right = expected[key], actual[key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"state-dict tensor mismatch: {key}")


def _conditional_models_from_artifact(artifact: dict[str, Any]) -> tuple[TextConditioningEncoder, ConditionalTinyUNet]:
    architecture = artifact["architecture"]
    text = TextConditioningEncoder(TextConditioningConfig(**architecture["text_encoder"]["config"]))
    unet = ConditionalTinyUNet(dropout=architecture["conditional_unet"]["dropout"])
    state = artifact["model_state_dict"]
    text.load_state_dict(state["text_encoder"])
    unet.load_state_dict(state["conditional_unet"])
    return text.eval(), unet.eval()


def _evaluator_from_artifact(artifact: dict[str, Any]) -> AttributeEvaluator:
    architecture = artifact["architecture"]
    if architecture["architecture_id"] != ARCHITECTURE_ID or architecture["class_counts"] != CLASS_COUNTS:
        raise ValueError("unexpected evaluator architecture metadata")
    model = AttributeEvaluator()
    model.load_state_dict(artifact["model_state_dict"])
    return model.eval()


def _unconditional_model_from_artifact(artifact: dict[str, Any]) -> TinyUNet:
    architecture = artifact["architecture"]
    if architecture != {"tiny_unet": {"class": "TinyUNet", "dropout": 0.0}}:
        raise ValueError("unexpected unconditional architecture metadata")
    model = TinyUNet(dropout=architecture["tiny_unet"]["dropout"])
    model.load_state_dict(artifact["model_state_dict"])
    return model.eval()


def _conditional_forward_equal(
    source_text: TextConditioningEncoder,
    source_unet: ConditionalTinyUNet,
    artifact_text: TextConditioningEncoder,
    artifact_unet: ConditionalTinyUNet,
) -> None:
    input_ids = torch.tensor([[1, 4, 5, 2, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    image = torch.linspace(-1.0, 1.0, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    timestep = torch.tensor([137], dtype=torch.long)
    with torch.no_grad():
        source_output = source_unet(image, timestep, source_text(input_ids, attention_mask), attention_mask)
        artifact_output = artifact_unet(image, timestep, artifact_text(input_ids, attention_mask), attention_mask)
    if not torch.equal(source_output, artifact_output):
        raise ValueError("conditional deterministic forward mismatch")


def _evaluator_forward_equal(source: AttributeEvaluator, artifact: AttributeEvaluator) -> None:
    image = torch.linspace(-1.0, 1.0, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        source_output, artifact_output = source(image), artifact(image)
    if source_output.keys() != artifact_output.keys() or any(
        not torch.equal(source_output[key], artifact_output[key]) for key in source_output
    ):
        raise ValueError("evaluator deterministic forward mismatch")


def _validate_common(artifact: dict[str, Any], *, expected_type: str, expected_count: int) -> None:
    required = {
        "artifact_format_version",
        "artifact_type",
        "model_role",
        "model_state_dict",
        "architecture",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_git_commit",
        "expected_parameter_count",
    }
    if not required.issubset(artifact) or artifact["artifact_format_version"] != 1:
        raise ValueError("invalid inference-artifact schema")
    if artifact["artifact_type"] != expected_type or artifact["expected_parameter_count"] != expected_count:
        raise ValueError("inference-artifact identity mismatch")
    if Path(artifact["source_checkpoint_path"]).is_absolute():
        raise ValueError("source checkpoint path must be repository-relative")
    forbidden = ("optimizer", "scheduler", "scaler", "rng", "progress", "history", "dataloader")
    if any(any(token in key.lower() for token in forbidden) for key in artifact):
        raise ValueError("resume-only metadata found in inference artifact")


def build_conditional() -> tuple[dict[str, Any], TextConditioningEncoder, ConditionalTinyUNet]:
    if sha256_file(CONDITIONAL_SOURCE) != CONDITIONAL_SOURCE_SHA256:
        raise ValueError("conditional source SHA256 mismatch")
    source = torch.load(CONDITIONAL_SOURCE, map_location="cpu", weights_only=False)
    if source.get("progress", {}).get("step") != 20_000 or not isinstance(source.get("ema_state_dict"), dict):
        raise ValueError("unexpected conditional source checkpoint structure")
    source_text, source_unet = load_ema_checkpoint(CONDITIONAL_SOURCE, expected_step=20_000, device=torch.device("cpu"))
    if parameter_total(source_text, source_unet) != CONDITIONAL_PARAMETER_COUNT:
        raise ValueError("conditional parameter-count mismatch")
    artifact = {
        "artifact_format_version": 1,
        "artifact_type": "canonical_inference_checkpoint",
        "model_role": "conditional_main",
        "selected_step": 20_000,
        "weight_variant": "EMA",
        "model_state_dict": {"text_encoder": source["ema_state_dict"]["text_encoder"], "conditional_unet": source["ema_state_dict"]["conditional_unet"]},
        "architecture": {
            "text_encoder": {"class": "TextConditioningEncoder", "config": asdict(TextConditioningConfig())},
            "conditional_unet": {"class": "ConditionalTinyUNet", "dropout": 0.0},
        },
        "source_checkpoint_path": relative(CONDITIONAL_SOURCE),
        "source_checkpoint_sha256": CONDITIONAL_SOURCE_SHA256,
        "source_git_commit": "2d38da843f792e9f91c687a7d051984134e87a0a",
        "expected_parameter_count": CONDITIONAL_PARAMETER_COUNT,
    }
    _validate_common(artifact, expected_type="canonical_inference_checkpoint", expected_count=CONDITIONAL_PARAMETER_COUNT)
    return artifact, source_text, source_unet


def build_evaluator() -> tuple[dict[str, Any], AttributeEvaluator]:
    if sha256_file(EVALUATOR_SOURCE) != EVALUATOR_SOURCE_SHA256:
        raise ValueError("evaluator source SHA256 mismatch")
    source = torch.load(EVALUATOR_SOURCE, map_location="cpu", weights_only=False)
    if source.get("completed_epoch") != 41 or source.get("best_epoch") != 41 or not isinstance(source.get("model_state_dict"), dict):
        raise ValueError("unexpected evaluator source checkpoint structure")
    source_model = AttributeEvaluator()
    source_model.load_state_dict(source["model_state_dict"])
    source_model.eval()
    if parameter_count() != EVALUATOR_PARAMETER_COUNT:
        raise ValueError("evaluator parameter-count mismatch")
    artifact = {
        "artifact_format_version": 1,
        "artifact_type": "canonical_evaluator_inference_checkpoint",
        "model_role": "dr04_evaluator",
        "selected_epoch": 41,
        "model_state_dict": source["model_state_dict"],
        "architecture": {
            "class": "AttributeEvaluator",
            "architecture_id": ARCHITECTURE_ID,
            "preprocessing_id": PREPROCESSING_ID,
            "class_counts": CLASS_COUNTS,
        },
        "source_checkpoint_path": relative(EVALUATOR_SOURCE),
        "source_checkpoint_sha256": EVALUATOR_SOURCE_SHA256,
        "source_git_commit": "a43ad8e43f35899052b0145e3cbd8ed31cac5517",
        "expected_parameter_count": EVALUATOR_PARAMETER_COUNT,
    }
    _validate_common(artifact, expected_type="canonical_evaluator_inference_checkpoint", expected_count=EVALUATOR_PARAMETER_COUNT)
    return artifact, source_model


def build_unconditional() -> tuple[dict[str, Any], TinyUNet, dict[str, torch.Tensor]]:
    if sha256_file(UNCONDITIONAL_SOURCE) != UNCONDITIONAL_SOURCE_SHA256:
        raise ValueError("unconditional source SHA256 mismatch")
    source = torch.load(UNCONDITIONAL_SOURCE, map_location="cpu", weights_only=False)
    progress_data = source.get("progress", {})
    progress = TrainingProgress(
        progress_data.get("step", -1), progress_data.get("images_seen", -1),
        progress_data.get("batch_offset", 0), progress_data.get("permutation"),
        progress_data.get("train_losses"), progress_data.get("skipped_step_count", 0),
    )
    required = {
        "tiny_unet_state_dict", "ema_tiny_unet_state_dict", "optimizer_state_dict",
        "grad_scaler_state_dict", "progress", "rng_state", "config", "provenance",
    }
    expected_config = asdict(UnconditionalMainConfig(run_id=UNCONDITIONAL_RUN_ID))
    if (
        source.get("run_id") != UNCONDITIONAL_RUN_ID
        or not required.issubset(source)
        or progress.step != 20_000
        or progress.images_seen != 640_000
        or not canonical_checkpoint_ready(progress)
        or source.get("config") != expected_config
    ):
        raise ValueError("unexpected unconditional source checkpoint structure")
    ema_state = source["ema_tiny_unet_state_dict"]
    source_model = TinyUNet(dropout=0.0)
    source_model.load_state_dict(ema_state)
    source_model.eval()
    if parameter_total(source_model) != UNCONDITIONAL_PARAMETER_COUNT:
        raise ValueError("unconditional parameter-count mismatch")
    if not all(torch.isfinite(tensor).all() for tensor in ema_state.values()):
        raise ValueError("unconditional EMA contains non-finite tensors")
    artifact = {
        "artifact_format_version": 1,
        "artifact_type": "canonical_inference_checkpoint",
        "model_role": "unconditional_main",
        "selected_step": 20_000,
        "weight_variant": "EMA",
        "model_state_dict": ema_state,
        "architecture": {"tiny_unet": {"class": "TinyUNet", "dropout": 0.0}},
        "source_checkpoint_path": relative(UNCONDITIONAL_SOURCE),
        "source_checkpoint_sha256": UNCONDITIONAL_SOURCE_SHA256,
        "source_git_commit": source["provenance"]["git_commit"],
        "source_run_id": UNCONDITIONAL_RUN_ID,
        "source_ema_field": "ema_tiny_unet_state_dict",
        "expected_parameter_count": UNCONDITIONAL_PARAMETER_COUNT,
    }
    _validate_common(artifact, expected_type="canonical_inference_checkpoint", expected_count=UNCONDITIONAL_PARAMETER_COUNT)
    return artifact, source_model, ema_state


def _unconditional_forward_smoke(model: TinyUNet) -> None:
    image = torch.linspace(-1.0, 1.0, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    timestep = torch.tensor([137], dtype=torch.long)
    with torch.no_grad():
        output = model(image, timestep)
    if output.shape != image.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("unconditional synthetic forward smoke failed")


def materialize_unconditional() -> dict[str, Any]:
    if UNCONDITIONAL_OUTPUT.exists():
        raise FileExistsError("unconditional canonical inference artifact already exists; refusing overwrite")
    artifact, source_model, source_ema = build_unconditional()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, UNCONDITIONAL_OUTPUT)
    loaded = torch.load(UNCONDITIONAL_OUTPUT, map_location="cpu", weights_only=False)
    _validate_common(loaded, expected_type="canonical_inference_checkpoint", expected_count=UNCONDITIONAL_PARAMETER_COUNT)
    artifact_model = _unconditional_model_from_artifact(loaded)
    assert_state_dict_equal(source_ema, loaded["model_state_dict"])
    assert_state_dict_equal(source_model.state_dict(), artifact_model.state_dict())
    _unconditional_forward_smoke(artifact_model)
    return {
        "path": relative(UNCONDITIONAL_OUTPUT), "size_bytes": UNCONDITIONAL_OUTPUT.stat().st_size,
        "sha256": sha256_file(UNCONDITIONAL_OUTPUT), "state_dict_tensor_count": len(artifact_model.state_dict()),
        "parameter_count": parameter_total(artifact_model), "exact_tensor_equality": True,
        "forward_smoke": True,
    }


def update_unconditional_bindings(*, artifact_sha256: str) -> str:
    """Resolve only the two D23-authorized unconditional binding entries."""
    before = json.loads(EXECUTION_BINDINGS.read_text(encoding="utf-8"))
    after = json.loads(json.dumps(before))
    after["bindings"]["unconditional_checkpoint_path"] = {
        "status": "resolved", "value": relative(UNCONDITIONAL_OUTPUT),
    }
    after["bindings"]["unconditional_checkpoint_sha256"] = {
        "status": "resolved", "value": artifact_sha256,
    }
    changed = {
        name for name in before["bindings"]
        if before["bindings"][name] != after["bindings"][name]
    }
    if changed != {"unconditional_checkpoint_path", "unconditional_checkpoint_sha256"}:
        raise ValueError(f"unexpected execution-binding changes: {changed}")
    EXECUTION_BINDINGS.write_bytes((json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return sha256_file(EXECUTION_BINDINGS)


def materialize() -> dict[str, dict[str, Any]]:
    if CONDITIONAL_OUTPUT.exists() or EVALUATOR_OUTPUT.exists():
        raise FileExistsError("canonical inference artifact already exists; refusing overwrite")
    conditional, source_text, source_unet = build_conditional()
    evaluator, source_evaluator = build_evaluator()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(conditional, CONDITIONAL_OUTPUT)
    torch.save(evaluator, EVALUATOR_OUTPUT)
    loaded_conditional = torch.load(CONDITIONAL_OUTPUT, map_location="cpu", weights_only=False)
    loaded_evaluator = torch.load(EVALUATOR_OUTPUT, map_location="cpu", weights_only=False)
    _validate_common(loaded_conditional, expected_type="canonical_inference_checkpoint", expected_count=CONDITIONAL_PARAMETER_COUNT)
    _validate_common(loaded_evaluator, expected_type="canonical_evaluator_inference_checkpoint", expected_count=EVALUATOR_PARAMETER_COUNT)
    artifact_text, artifact_unet = _conditional_models_from_artifact(loaded_conditional)
    artifact_evaluator = _evaluator_from_artifact(loaded_evaluator)
    assert_state_dict_equal(source_text.state_dict(), artifact_text.state_dict())
    assert_state_dict_equal(source_unet.state_dict(), artifact_unet.state_dict())
    assert_state_dict_equal(source_evaluator.state_dict(), artifact_evaluator.state_dict())
    _conditional_forward_equal(source_text, source_unet, artifact_text, artifact_unet)
    _evaluator_forward_equal(source_evaluator, artifact_evaluator)
    return {
        "conditional": {
            "path": relative(CONDITIONAL_OUTPUT), "size_bytes": CONDITIONAL_OUTPUT.stat().st_size,
            "sha256": sha256_file(CONDITIONAL_OUTPUT), "state_dict_tensor_count": len(artifact_text.state_dict()) + len(artifact_unet.state_dict()),
            "parameter_count": parameter_total(artifact_text, artifact_unet), "exact_tensor_equality": True, "forward_equivalence": True,
        },
        "evaluator": {
            "path": relative(EVALUATOR_OUTPUT), "size_bytes": EVALUATOR_OUTPUT.stat().st_size,
            "sha256": sha256_file(EVALUATOR_OUTPUT), "state_dict_tensor_count": len(artifact_evaluator.state_dict()),
            "parameter_count": parameter_total(artifact_evaluator), "exact_tensor_equality": True, "forward_equivalence": True,
        },
    }


if __name__ == "__main__":
    print(json.dumps(materialize(), indent=2, sort_keys=True))
