from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tiny_avatar_diffusion.ca16_r12_orchestration as r12

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")


def _slot(index: int) -> dict[str, object]:
    return {
        "planned_sample_index": index,
        "sample_id": f"s{index}",
        "prompt_id": f"p{index // 5}",
        "prompt": "synthetic prompt",
        "panel_split": "ID",
        "ood_family": None,
        "control_role": None,
        "matched_ood_prompt_id": None,
        "seed": 2026 + index % 5,
        "checkpoint_path": "synthetic.pt",
        "checkpoint_sha256": "synthetic",
        "targets": {name: 1 for name in ATTRIBUTES},
    }


def test_r12_resolves_exactly_the_six_r11_winners_without_reselection():
    winners = r12.resolve_selected_checkpoints()
    assert len(winners) == 6
    assert {winner["milestone_step"] for winner in winners} == {20_000}
    assert {winner["training_seed"] for winner in winners} == {2026, 2027, 2028}
    assert {winner["architecture_arm"] for winner in winners} == {"ca8", "ca8_ca16"}
    assert all(
        Path(str(winner["checkpoint_path"])).name == "milestone_20000.pt"
        for winner in winners
    )


def test_preflight_hashes_sealed_inputs_without_opening_prompt_records(
    monkeypatch, tmp_path
):
    protocol = {
        "schema": "tiny_avatar_diffusion.ca16_r12_execution_protocol_v1",
        "protocol_id": r12.PROTOCOL_ID,
        "status": "PRE_FREEZE",
        "dr04_reuse": {
            "panel_manifest_sha256": "p",
            "matched_controls_sha256": "c",
            "protocol_sha256": "d",
            "geometry": {
                "prompt_count": 252,
                "id_prompt_count": 152,
                "final_ood_prompt_count": 100,
                "seeds": [2026, 2027, 2028, 2029, 2030],
                "slots_per_model": 1260,
            },
        },
        "bindings": {
            "r11_complete_sha256": "r",
            "r11_inventory_sha256": "i",
            "vocabulary_sha256": "v",
            "evaluator_freeze_path": "data/processed/dr04_evaluator_v1/evaluator_checkpoint_freeze.json",
            "evaluator_freeze_sha256": "f",
            "evaluator_checkpoint_sha256": "e",
        },
        "sampling": {
            "weight_variant": "EMA",
            "precision": "FP32",
            "sampler": "ancestral DDPM",
            "timesteps": 1000,
            "sampling_batch_size": 8,
            "batch_partitioning": "canonical contiguous plan groups; incomplete resumed groups are regenerated in memory with their original lanes and only missing outputs are published",
            "final_batch_padding": "the final four canonical records are repeated in canonical order as four non-published padding lanes, so every sampler invocation has exactly eight lanes",
        },
    }
    monkeypatch.setattr(r12, "load_protocol", lambda: protocol)
    monkeypatch.setattr(
        r12, "_git_state", lambda: ("frozen", "exp/ca16-ablation", True)
    )
    checked: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        r12,
        "_require_hash",
        lambda path, expected: checked.append((path, expected)) or expected,
    )
    monkeypatch.setattr(
        r12,
        "resolve_selected_checkpoints",
        lambda: ({"run_id": "r", "selection_path": "s", "selection_sha256": "s"},) * 6,
    )
    opened: list[Path] = []
    monkeypatch.setattr(
        r12,
        "_read_json",
        lambda path: (
            opened.append(path)
            or pytest.fail("sealed content must not be parsed in preflight")
        ),
    )
    result = r12.validate_preflight(
        output_root=tmp_path / "r12", expected_eval_sha="frozen"
    )
    assert result["status"] == "preflight_pass"
    assert result["sealed_content_opened"] is False
    assert opened == []
    assert {expected for _, expected in checked} >= {
        "p",
        "c",
        "d",
        "r",
        "i",
        "v",
        "f",
        "e",
    }


def test_execution_plan_is_namespaced_after_confirmed_panel_load(monkeypatch):
    rows = [{"synthetic": True}]
    base = [_slot(0)]
    monkeypatch.setattr(r12, "_read_json", lambda _path: {"prompt_records": rows})
    monkeypatch.setattr(r12, "validate_panel", lambda panel, controls: rows)
    monkeypatch.setattr(
        r12, "expand_plan", lambda loaded, path, digest: [dict(row) for row in base]
    )
    monkeypatch.setattr(r12, "validate_execution_plan", lambda _plan: None)
    plan = r12.build_execution_plan(
        checkpoint={
            "run_id": "ca16_ablation_ca8_seed2026",
            "architecture_arm": "ca8",
            "training_seed": 2026,
            "milestone_step": 20_000,
            "checkpoint_path": "c.pt",
            "checkpoint_sha256": "h",
        }
    )
    assert plan[0]["sample_id"] == "ca16_ablation_ca8_seed2026__s0"
    assert plan[0]["r12_run_id"] == "ca16_ablation_ca8_seed2026"


def test_b8_resume_regenerates_the_original_eight_lanes_but_returns_missing_lane(
    monkeypatch,
):
    calls: list[int] = []

    def fake_sample(_schedule, _text, _denoiser, ids, _masks, **_kwargs):
        calls.append(int(ids.shape[0]))
        return torch.zeros((ids.shape[0], 3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(r12, "conditional_sample", fake_sample)
    tokenizer = SimpleNamespace(
        encode=lambda _prompt: SimpleNamespace(token_ids=[1, 2], attention_mask=[1, 1])
    )
    plan = [_slot(index) for index in range(8)]
    generator = r12.BatchedPlanGenerator(
        plan=plan,
        text_encoder=None,
        denoiser=None,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    assert generator(plan[3]).shape == (32, 32, 3)
    assert calls == [8]
    generator(plan[4])
    assert calls == [8]


def test_b8_tail_is_padded_to_eight_and_never_caches_padding_records(monkeypatch):
    calls: list[int] = []

    def fake_sample(_schedule, _text, _denoiser, ids, _masks, **_kwargs):
        calls.append(int(ids.shape[0]))
        return torch.zeros((ids.shape[0], 3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(r12, "conditional_sample", fake_sample)
    tokenizer = SimpleNamespace(
        encode=lambda _prompt: SimpleNamespace(token_ids=[1, 2], attention_mask=[1, 1])
    )
    plan = [_slot(index) for index in range(12)]
    generator = r12.BatchedPlanGenerator(
        plan=plan,
        text_encoder=None,
        denoiser=None,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    real, lanes = generator._lanes(8)
    assert [row["sample_id"] for row in real] == ["s8", "s9", "s10", "s11"]
    assert [row["sample_id"] for row in lanes] == [
        "s8",
        "s9",
        "s10",
        "s11",
        "s8",
        "s9",
        "s10",
        "s11",
    ]
    generator(plan[10])
    assert calls == [8]
    assert set(generator.cache) == {"s8", "s9", "s11"}


def test_r12_executor_synthetic_resume_and_metric_lifecycle(tmp_path):
    plan = [_slot(index) for index in range(5)]
    executor = r12.R12Executor(
        tmp_path,
        plan,
        {
            "protocol_id": r12.PROTOCOL_ID,
            "evaluator_checkpoint_path": "e.pt",
            "evaluator_checkpoint_sha256": "e",
        },
    )
    executor.record_parameter_counts(
        {"text_encoder": 1, "denoiser": 2, "total": 3, "evaluator": 4}
    )
    executor.generate(lambda _slot: np.zeros((32, 32, 3), dtype=np.uint8))
    executor.audit()
    executor.predict(lambda _pixels: {name: 1 for name in ATTRIBUTES})
    executor.metrics()
    summary = executor.finalize({"synthetic": True})
    assert summary["schema"] == "tiny_avatar_diffusion.ca16_r12_summary_v1"
    assert summary["generation"]["completed"] == 5


def test_completed_summary_is_validated_and_corruption_preserves_existing_pngs(
    tmp_path,
):
    plan = [_slot(index) for index in range(5)]
    identities = {
        "protocol_id": r12.PROTOCOL_ID,
        "evaluator_checkpoint_path": "e.pt",
        "evaluator_checkpoint_sha256": "e",
    }
    executor = r12.R12Executor(tmp_path, plan, identities)
    executor.record_parameter_counts(
        {"text_encoder": 1, "denoiser": 2, "total": 3, "evaluator": 4}
    )
    executor.generate(lambda _slot: np.zeros((32, 32, 3), dtype=np.uint8))
    executor.audit()
    executor.predict(lambda _pixels: {name: 1 for name in ATTRIBUTES})
    executor.metrics()
    executor.finalize({"synthetic": True})
    before = {path: path.read_bytes() for path in (tmp_path / "images").glob("*.png")}
    assert r12.validate_completed_run(executor)["status"] == "COMPLETE"
    summary_path = tmp_path / "r12_summary.json"
    summary = r12._read_json(summary_path)
    summary["status"] = "INCOMPLETE"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="identity or status"):
        r12.validate_completed_run(executor)
    assert {path: path.read_bytes() for path in before} == before


def test_execution_geometry_rejects_split_seed_and_identity_drift():
    plan = []
    for split, prompt_count in (("ID", 152), ("OOD", 100)):
        for prompt in range(prompt_count):
            for seed in range(2026, 2031):
                index = len(plan)
                row = _slot(index)
                row.update(
                    sample_id=f"{split}-{prompt}-{seed}",
                    prompt_id=f"{split}-{prompt}",
                    panel_split=split,
                    seed=seed,
                )
                plan.append(row)
    r12.validate_execution_plan(plan)
    plan[-1]["seed"] = 2031
    with pytest.raises(ValueError, match="seed"):
        r12.validate_execution_plan(plan)


def test_selected_checkpoint_resolves_typed_frozen_spec_for_both_architectures():
    winners = r12.resolve_selected_checkpoints()
    specs = {r12.resolve_run_spec(winner) for winner in winners}
    assert {spec.architecture_arm.value for spec in specs} == {"ca8", "ca8_ca16"}
    assert all(isinstance(spec, r12.CA16RunSpec) for spec in specs)


def test_orchestration_refuses_execute_without_an_expected_tag(tmp_path):
    with pytest.raises(ValueError, match="annotated tag"):
        r12.execute_r12(output_root=tmp_path / "r12", expected_eval_sha="x")


def test_preflight_propagates_an_evaluator_or_vocabulary_hash_mismatch(
    monkeypatch, tmp_path
):
    protocol = {
        "schema": "tiny_avatar_diffusion.ca16_r12_execution_protocol_v1",
        "protocol_id": r12.PROTOCOL_ID,
        "status": "PRE_FREEZE",
        "dr04_reuse": {
            "panel_manifest_sha256": "p",
            "matched_controls_sha256": "c",
            "protocol_sha256": "d",
            "geometry": {
                "prompt_count": 252,
                "id_prompt_count": 152,
                "final_ood_prompt_count": 100,
                "seeds": [2026, 2027, 2028, 2029, 2030],
                "slots_per_model": 1260,
            },
        },
        "bindings": {
            "r11_complete_sha256": "r",
            "r11_inventory_sha256": "i",
            "vocabulary_sha256": "bad-vocab",
            "evaluator_freeze_path": "data/processed/dr04_evaluator_v1/evaluator_checkpoint_freeze.json",
            "evaluator_freeze_sha256": "f",
            "evaluator_checkpoint_sha256": "e",
        },
        "sampling": {
            "weight_variant": "EMA",
            "precision": "FP32",
            "sampler": "ancestral DDPM",
            "timesteps": 1000,
            "sampling_batch_size": 8,
            "batch_partitioning": "canonical contiguous plan groups; incomplete resumed groups are regenerated in memory with their original lanes and only missing outputs are published",
            "final_batch_padding": "the final four canonical records are repeated in canonical order as four non-published padding lanes, so every sampler invocation has exactly eight lanes",
        },
    }
    monkeypatch.setattr(r12, "load_protocol", lambda: protocol)
    monkeypatch.setattr(
        r12, "_git_state", lambda: ("frozen", "exp/ca16-ablation", True)
    )
    monkeypatch.setattr(
        r12,
        "_require_hash",
        lambda _path, expected: (
            (_ for _ in ()).throw(ValueError("hash mismatch"))
            if expected == "bad-vocab"
            else expected
        ),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        r12.validate_preflight(output_root=tmp_path / "r12", expected_eval_sha="frozen")


def test_cli_requires_explicit_heldout_confirmation(monkeypatch):
    path = ROOT / "scripts" / "run_ca16_r12_evaluation.py"
    spec = importlib.util.spec_from_file_location("r12_cli_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "load_protocol", lambda: {"output_root": "results/synthetic"}
    )
    monkeypatch.setattr(
        module, "validate_preflight", lambda **_kwargs: {"status": "ok"}
    )
    with pytest.raises(SystemExit):
        module.main(
            [
                "--expected-eval-sha",
                "x",
                "--execute",
                "--confirm-r12-heldout-evaluation",
            ]
        )
    with pytest.raises(SystemExit):
        module.main(
            [
                "--expected-eval-sha",
                "x",
                "--expected-eval-tag",
                "tag",
                "--execute",
            ]
        )
