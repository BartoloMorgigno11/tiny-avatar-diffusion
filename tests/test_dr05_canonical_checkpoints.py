from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import scripts.dr05_materialize_canonical_checkpoints as materializer
from tiny_avatar_diffusion.evaluator import ARCHITECTURE_ID, AttributeEvaluator, parameter_count
from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ROOT = Path(__file__).parents[1]
CONDITIONAL = ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt"
EVALUATOR = ROOT / "artifacts" / "checkpoints" / "dr04_evaluator_epoch41.pt"
UNCONDITIONAL = ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"checked-in canonical artifact unavailable: {path.name}")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_conditional_artifact_schema_and_repository_native_loading():
    artifact = _artifact(CONDITIONAL)
    assert artifact["artifact_format_version"] == 1
    assert artifact["artifact_type"] == "canonical_inference_checkpoint"
    assert artifact["model_role"] == "conditional_main"
    assert artifact["selected_step"] == 20_000
    assert artifact["weight_variant"] == "EMA"
    assert artifact["expected_parameter_count"] == 4_723_907
    assert artifact["source_checkpoint_sha256"] == "0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884"
    assert not Path(artifact["source_checkpoint_path"]).is_absolute()
    assert not any(token in key.lower() for key in artifact for token in ("optimizer", "scheduler", "scaler", "rng", "progress", "history", "dataloader"))
    text = TextConditioningEncoder(TextConditioningConfig(**artifact["architecture"]["text_encoder"]["config"]))
    unet = ConditionalTinyUNet(dropout=artifact["architecture"]["conditional_unet"]["dropout"])
    text.load_state_dict(artifact["model_state_dict"]["text_encoder"])
    unet.load_state_dict(artifact["model_state_dict"]["conditional_unet"])
    assert sum(p.numel() for p in text.parameters()) + sum(p.numel() for p in unet.parameters()) == 4_723_907


def test_evaluator_artifact_schema_and_repository_native_loading():
    artifact = _artifact(EVALUATOR)
    assert artifact["artifact_format_version"] == 1
    assert artifact["artifact_type"] == "canonical_evaluator_inference_checkpoint"
    assert artifact["model_role"] == "dr04_evaluator"
    assert artifact["selected_epoch"] == 41
    assert artifact["expected_parameter_count"] == 293_064
    assert artifact["source_checkpoint_sha256"] == "3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f"
    assert artifact["architecture"]["architecture_id"] == ARCHITECTURE_ID
    assert not Path(artifact["source_checkpoint_path"]).is_absolute()
    assert not any(token in key.lower() for key in artifact for token in ("optimizer", "scheduler", "scaler", "rng", "progress", "history", "dataloader"))
    model = AttributeEvaluator()
    model.load_state_dict(artifact["model_state_dict"])
    assert parameter_count() == 293_064


def test_unconditional_artifact_is_exact_ema_inference_only_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class SyntheticTinyUNet(torch.nn.Module):
        def __init__(self, *, dropout: float) -> None:
            super().__init__()
            assert dropout == 0.0
            self.weight = torch.nn.Parameter(torch.zeros(1))

    canonical_run_id = materializer.UNCONDITIONAL_RUN_ID
    root = tmp_path / "repository"
    source = root / "results" / "synthetic" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    raw_state = {"weight": torch.tensor([1.0])}
    ema_state = {"weight": torch.tensor([2.0])}
    run_id = "synthetic_unconditional_run"
    torch.save(
        {
            "run_id": run_id,
            "tiny_unet_state_dict": raw_state,
            "ema_tiny_unet_state_dict": ema_state,
            "optimizer_state_dict": {"state": {}},
            "grad_scaler_state_dict": {},
            "progress": {"step": 20_000, "images_seen": 640_000},
            "rng_state": {"torch": torch.tensor([7], dtype=torch.uint8)},
            "config": asdict(materializer.UnconditionalMainConfig(run_id=run_id)),
            "provenance": {"git_commit": "synthetic-commit"},
        },
        source,
    )
    monkeypatch.setattr(materializer, "ROOT", root)
    monkeypatch.setattr(materializer, "UNCONDITIONAL_SOURCE", source)
    monkeypatch.setattr(materializer, "UNCONDITIONAL_SOURCE_SHA256", materializer.sha256_file(source))
    monkeypatch.setattr(materializer, "UNCONDITIONAL_RUN_ID", run_id)
    monkeypatch.setattr(materializer, "UNCONDITIONAL_PARAMETER_COUNT", 1)
    monkeypatch.setattr(materializer, "TinyUNet", SyntheticTinyUNet)

    synthetic_first, _, _ = materializer.build_unconditional()
    synthetic_second, _, _ = materializer.build_unconditional()
    assert synthetic_first["model_state_dict"]["weight"].item() == 2.0
    assert not torch.equal(synthetic_first["model_state_dict"]["weight"], raw_state["weight"])
    assert set(synthetic_first) == {
        "artifact_format_version", "artifact_type", "model_role", "selected_step", "weight_variant",
        "model_state_dict", "architecture", "source_checkpoint_path", "source_checkpoint_sha256",
        "source_git_commit", "source_run_id", "source_ema_field", "expected_parameter_count",
    }
    assert synthetic_first["source_checkpoint_path"] == "results/synthetic/checkpoint.pt"
    assert synthetic_first["source_git_commit"] == "synthetic-commit"
    assert synthetic_first["source_run_id"] == run_id
    assert synthetic_first["source_ema_field"] == "ema_tiny_unet_state_dict"
    assert not any(token in key.lower() for key in synthetic_first for token in ("optimizer", "scaler", "rng", "progress", "history", "dataloader"))
    assert synthetic_first.keys() == synthetic_second.keys()
    assert all(
        value == synthetic_second[key] if not isinstance(value, dict) else value.keys() == synthetic_second[key].keys()
        for key, value in synthetic_first.items()
    )
    assert torch.equal(synthetic_first["model_state_dict"]["weight"], synthetic_second["model_state_dict"]["weight"])

    artifact = _artifact(UNCONDITIONAL)
    assert _sha(UNCONDITIONAL) == "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e"
    assert artifact["artifact_format_version"] == 1
    assert artifact["artifact_type"] == "canonical_inference_checkpoint"
    assert artifact["model_role"] == "unconditional_main"
    assert artifact["selected_step"] == 20_000
    assert artifact["weight_variant"] == "EMA"
    assert artifact["source_checkpoint_sha256"] == "a23436b8f3713d87374ef3ad847aadeffd38469fff89bcb169424539c9842129"
    assert artifact["source_run_id"] == canonical_run_id
    assert artifact["source_ema_field"] == "ema_tiny_unet_state_dict"
    assert artifact["expected_parameter_count"] == 4_595_779
    assert not Path(artifact["source_checkpoint_path"]).is_absolute()
    assert not any(token in key.lower() for key in artifact for token in ("optimizer", "scaler", "rng", "progress", "history", "dataloader"))
    assert all(bool(torch.isfinite(tensor).all()) for tensor in artifact["model_state_dict"].values())
    assert not any(token in key.lower() for key in artifact["model_state_dict"] for token in ("text_encoder", "transformer", "cross_attention", "conditioning"))


def test_artifacts_do_not_depend_on_local_source_checkpoints():
    conditional = _artifact(CONDITIONAL)
    evaluator = _artifact(EVALUATOR)
    unconditional = _artifact(UNCONDITIONAL)
    assert conditional["source_checkpoint_path"].startswith("reports/")
    assert evaluator["source_checkpoint_path"].startswith("results/")
    assert unconditional["source_checkpoint_path"].startswith("results/")
    assert _sha(CONDITIONAL) != conditional["source_checkpoint_sha256"]
    assert _sha(EVALUATOR) != evaluator["source_checkpoint_sha256"]
    assert _sha(UNCONDITIONAL) != unconditional["source_checkpoint_sha256"]
