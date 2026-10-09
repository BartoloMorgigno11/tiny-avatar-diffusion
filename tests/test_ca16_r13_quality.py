from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import tiny_avatar_diffusion.ca16_r13_quality as r13


def _record(index: int) -> dict[str, object]:
    role = "COMP_VAL" if index < 338 else "ID_VAL"
    return {
        "canonical_manifest_index": index,
        "generation_seed": 2026 + index,
        "stable_record_key": f"10k|{role}|image{index:04d}",
        "dataset_variant": "10k",
        "split_role": role,
        "image_id": f"image{index:04d}",
        "caption": "synthetic caption",
        "source_image_path": "synthetic",
    }


def _plan(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            **_record(index),
            "planned_sample_index": index,
            "sample_id": f"quality_{index:04d}",
            "token_ids": [1, 2],
            "attention_mask": [1, 1],
        }
        for index in range(count)
    ]


def test_quality_plan_enforces_frozen_geometry_and_canonical_seeds(monkeypatch):
    entries = [_record(index) for index in range(804)]
    tokens = {
        str(row["stable_record_key"]): {
            "dataset_variant": row["dataset_variant"],
            "split_role": row["split_role"],
            "image_id": row["image_id"],
            "caption": row["caption"],
            "token_ids": [1, 2],
            "attention_mask": [1, 1],
        }
        for row in entries
    }
    monkeypatch.setattr(r13, "_read_json", lambda _path: {"entries": entries})
    monkeypatch.setattr(r13, "_token_index", lambda _protocol: tokens)
    plan = r13.build_execution_plan({})
    assert len(plan) == 804
    assert {row["split_role"] for row in plan} == {"ID_VAL", "COMP_VAL"}
    assert [row["generation_seed"] for row in plan] == list(range(2026, 2830))


def test_quality_plan_rejects_token_identity_mismatch(monkeypatch):
    entries = [_record(index) for index in range(804)]
    tokens = {
        str(row["stable_record_key"]): {
            "dataset_variant": row["dataset_variant"],
            "split_role": row["split_role"],
            "image_id": row["image_id"],
            "caption": row["caption"],
            "token_ids": [1],
            "attention_mask": [1],
        }
        for row in entries
    }
    tokens[str(entries[0]["stable_record_key"])]["caption"] = "wrong"
    monkeypatch.setattr(r13, "_read_json", lambda _path: {"entries": entries})
    monkeypatch.setattr(r13, "_token_index", lambda _protocol: tokens)
    with pytest.raises(ValueError, match="caption"):
        r13.build_execution_plan({})


def test_b8_generator_reuses_original_lanes_on_partial_resume(monkeypatch):
    calls: list[int] = []

    def fake_sample(_schedule, _encoder, _denoiser, ids, _masks, **_kwargs):
        calls.append(int(ids.shape[0]))
        return torch.zeros((ids.shape[0], 3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(r13, "conditional_sample", fake_sample)
    plan = _plan(8)
    generator = r13.BatchedQualityGenerator(plan, None, None, torch.device("cpu"))
    assert generator(plan[3]).shape == (32, 32, 3)
    assert generator(plan[4]).shape == (32, 32, 3)
    assert calls == [8]


def test_b8_tail_padding_is_exactly_eight_and_never_published(monkeypatch):
    calls: list[int] = []

    def fake_sample(_schedule, _encoder, _denoiser, ids, _masks, **_kwargs):
        calls.append(int(ids.shape[0]))
        return torch.zeros((ids.shape[0], 3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(r13, "conditional_sample", fake_sample)
    plan = _plan(12)
    generator = r13.BatchedQualityGenerator(plan, None, None, torch.device("cpu"))
    generator(plan[10])
    assert calls == [8]
    assert set(generator.cache) == {"quality_0008", "quality_0009", "quality_0011"}


def test_executor_resume_preserves_existing_png_and_rejects_corruption(tmp_path):
    plan = _plan()
    identities = {"checkpoint_path": "synthetic.pt", "checkpoint_sha256": "hash"}
    executor = r13.R13Executor(tmp_path, plan, identities)
    executor.generate(lambda _row: np.zeros((32, 32, 3), dtype=np.uint8))
    first = tmp_path / "images" / "comp_val" / "0000.png"
    before = first.read_bytes()
    executor.generate(lambda _row: pytest.fail("valid output must be reused"))
    assert first.read_bytes() == before
    executor.audit()
    state = executor.state()
    state["metrics"] = "complete"
    state["phase"] = 3
    executor._write(state)
    metrics = {
        name: {"status": "COMPUTED"} for name in ("KID_ID", "KID_COMP", "KID_GLOBAL")
    }
    r13.atomic_json(tmp_path / "quality_metrics.json", metrics)
    summary = executor.finalize(metrics, {"synthetic": True})
    assert r13.validate_completed_run(executor) == summary
    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="PNG identity"):
        executor.audit()


def test_atomic_jsonl_retries_transient_windows_access_denied(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    calls = 0

    def transient(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            error = PermissionError("access denied")
            error.winerror = 5
            raise error
        os.replace(source, destination)

    monkeypatch.setattr(r13.time, "sleep", lambda _delay: None)
    r13.atomic_jsonl(path, [{"planned_sample_index": 0, "value": 1}], replace=transient)
    assert calls == 3
    assert r13._read_jsonl(path) == [{"planned_sample_index": 0, "value": 1}]


def test_atomic_json_preserves_existing_destination_on_permanent_access_denied(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"old": True}), encoding="utf-8")

    def permanent(_source: Path, _destination: Path) -> None:
        error = PermissionError("access denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(r13.time, "sleep", lambda _delay: None)
    with pytest.raises(PermissionError, match="access denied"):
        r13.atomic_json(path, {"new": True}, replace=permanent)
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_preflight_is_identity_only_and_never_builds_quality_plan(
    monkeypatch, tmp_path
):
    protocol = {
        "quality_manifest": {
            "sha256": "q",
            "geometry": {"ID_VAL": 466, "COMP_VAL": 338, "GLOBAL": 804},
            "generation_seed": "2026+canonical_manifest_index",
        },
        "sampling": {
            "weight_variant": "EMA",
            "precision": "FP32",
            "sampler": "ancestral DDPM",
            "timesteps": 1000,
            "sampling_batch_size": 8,
        },
        "quality_metrics": {
            "primary": ["KID_ID", "KID_COMP", "KID_GLOBAL"],
            "implementation": {
                "version": r13.CLEAN_FID_VERSION,
                "mode": "clean",
                "feature_extractor": "standard Inception-v3",
            },
        },
        "token_sources": {
            role: {"path": f"{role}.jsonl", "sha256": role}
            for role in ("ID_VAL", "COMP_VAL")
        },
        "r11_bindings": {
            "complete_path": "complete.json",
            "complete_sha256": "c",
            "inventory_path": "inventory.json",
            "inventory_sha256": "i",
        },
    }
    monkeypatch.setattr(r13, "load_protocol", lambda: protocol)
    monkeypatch.setattr(
        r13, "_git_state", lambda: ("frozen", "exp/ca16-ablation", True)
    )
    monkeypatch.setattr(r13, "_require_hash", lambda _path, expected: expected)
    monkeypatch.setattr(
        r13,
        "resolve_selected_checkpoints",
        lambda: tuple(
            {"selection_path": f"s{index}", "selection_sha256": f"h{index}"}
            for index in range(6)
        ),
    )
    result = r13.validate_preflight(
        output_root=tmp_path / "new", expected_eval_sha="frozen"
    )
    assert result["status"] == "preflight_pass"
    assert result["geometry"]["total_slots"] == 4824
