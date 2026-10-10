from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tiny_avatar_diffusion import ca16_r12_orchestration as r12
from tiny_avatar_diffusion import ca16_r14_orchestration as r14
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer


@pytest.fixture(scope="module")
def protocol() -> dict:
    return r14.load_protocol()


@pytest.fixture(scope="module")
def plan(protocol: dict) -> list[dict]:
    return r14.build_execution_plan(protocol)


def test_protocol_binds_the_approved_panel(protocol: dict):
    assert protocol["status"] == "FROZEN"
    assert protocol["diversity_manifest"]["geometry"] == {
        "ID_VAL": 100,
        "COMP_VAL": 100,
        "COMP_family": {"CV1": 25, "CV2": 25, "CV3": 25, "CV4": 25},
    }
    assert protocol["scientific_scope"]["forbidden_roles"] == [
        "TRAIN",
        "ID_TEST",
        "FINAL_OOD",
    ]


def test_plan_has_exact_population_seed_and_batch_geometry(plan: list[dict]):
    assert len(plan) == 1000
    assert len(plan) % 8 == 0
    assert {row["split_role"] for row in plan} == {"ID_VAL", "COMP_VAL"}
    assert sum(row["split_role"] == "ID_VAL" for row in plan) == 500
    assert sum(row["split_role"] == "COMP_VAL" for row in plan) == 500
    assert {row["generation_seed"] for row in plan} == {2026, 2027, 2028, 2029, 2030}
    assert {row["comp_family"] for row in plan if row["split_role"] == "COMP_VAL"} == {
        "CV1",
        "CV2",
        "CV3",
        "CV4",
    }


def test_r14_adapter_reproduces_the_legacy_generator_contract(plan: list[dict]):
    executor = r14.R14Executor(Path("unused"), plan, {"protocol_sha256": "frozen"})
    plan_sha = executor._plan_sha()
    projected = r14.validate_generator_adapter(plan)
    assert len(projected) == len(plan)
    assert all("prompt" not in row and "seed" not in row for row in plan)
    assert all(
        row["prompt"] == original["caption"]
        and row["seed"] == original["generation_seed"]
        and row["sample_id"] == original["sample_id"]
        for original, row in zip(plan, projected, strict=True)
    )
    _plan, checkpoints = r14.resolve_inputs(r14.load_protocol())
    assert len(checkpoints) == 6
    assert all(len(r14.adapt_plan_for_r12_generator(plan)) == 1000 for _ in checkpoints)
    assert executor._plan_sha() == plan_sha


def test_legacy_raw_r14_rows_reproduce_missing_prompt_before_adapter(
    plan: list[dict],
):
    generator = r12.BatchedPlanGenerator(
        plan=plan[:8],
        text_encoder=object(),
        denoiser=object(),
        tokenizer=SimpleNamespace(encode=lambda _prompt: None),
        device=torch.device("cpu"),
    )
    with pytest.raises(KeyError, match="prompt"):
        generator._generate_group(0)


def test_adapted_first_group_tokenizes_and_samples_without_ddpm(
    monkeypatch: pytest.MonkeyPatch, plan: list[dict]
):
    projected = r14.validate_generator_adapter(plan)[:8]
    vocab = r14._read_json(r14.ROOT / "data/processed/dr01_v1/text/10k/vocab.json")
    tokenizer = WhitespaceTokenizer(vocab["token_to_id"])
    captured: dict[str, torch.Tensor] = {}

    def fake_sample(_schedule, _encoder, _denoiser, ids, masks, **_kwargs):
        captured["ids"], captured["masks"] = ids.cpu(), masks.cpu()
        return torch.zeros((8, 3, 32, 32), dtype=torch.float32)

    monkeypatch.setattr(r12, "conditional_sample", fake_sample)
    generator = r12.BatchedPlanGenerator(
        plan=projected,
        text_encoder=object(),
        denoiser=object(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    generator._generate_group(0)
    assert set(generator.cache) == {row["sample_id"] for row in projected}
    assert captured["ids"].tolist() == [row["token_ids"] for row in projected]
    assert captured["masks"].tolist() == [row["attention_mask"] for row in projected]


def test_generator_adapter_rejects_missing_or_incompatible_prompt_fields(
    plan: list[dict],
):
    missing_caption = [dict(plan[0])]
    missing_caption[0].pop("caption")
    with pytest.raises(ValueError, match="generator schema"):
        r14.adapt_plan_for_r12_generator(missing_caption)
    incompatible_tokens = [dict(plan[0])]
    incompatible_tokens[0]["token_ids"] = [0] * 7
    with pytest.raises(ValueError, match="conditioning mismatch"):
        r14.validate_generator_adapter(incompatible_tokens)


def test_pairs_are_ten_per_prompt_and_two_thousand_total(plan: list[dict]):
    artifacts = {
        row["sample_id"]: {"png_relative_path": f"images/{row['sample_id']}.png"}
        for row in plan
    }
    pairs = r14.enumerate_r14_pairs(plan, artifacts)
    assert len(pairs) == 2000
    assert len({row["pair_id"] for row in pairs}) == 2000
    assert all(row["seed_a"] < row["seed_b"] for row in pairs)


def test_prompt_and_equal_family_aggregation(plan: list[dict]):
    artifacts = {
        row["sample_id"]: {"png_relative_path": f"images/{row['sample_id']}.png"}
        for row in plan
    }
    pairs = r14.enumerate_r14_pairs(plan, artifacts)
    measured = [
        {**row, "lpips": 0.1 if row["panel_split"] == "ID" else 0.2} for row in pairs
    ]
    prompts, aggregate = r14.aggregate_r14_pairs(measured)
    assert len(prompts) == 200
    assert aggregate["ID_VAL_prompt_macro_lpips"] == pytest.approx(0.1)
    assert aggregate["COMP_VAL_equal_family_macro_lpips"] == pytest.approx(0.2)


def test_nonfinite_lpips_fails_closed(plan: list[dict]):
    artifacts = {
        row["sample_id"]: {"png_relative_path": f"images/{row['sample_id']}.png"}
        for row in plan
    }
    pairs = r14.enumerate_r14_pairs(plan, artifacts)
    with pytest.raises(ValueError, match="non-finite"):
        r14.aggregate_r14_pairs([{**row, "lpips": np.nan} for row in pairs])


def test_six_checkpoint_inputs_and_balanced_efficiency_order(protocol: dict):
    _plan, checkpoints = r14.resolve_inputs(protocol)
    order = r14.efficiency_execution_order(checkpoints)
    assert len(checkpoints) == 6
    assert len(order) == 12
    assert {(row["architecture_arm"], row["batch_size"]) for row in order} == {
        (arm, batch) for arm in ("ca8", "ca8_ca16") for batch in (1, 8)
    }


def test_parameter_count_expectations_are_fail_closed(protocol: dict):
    ca8 = {"architecture_arm": "ca8"}
    r14.validate_parameter_counts(
        ca8, {"text_encoder": 102976, "total": 4723907}, protocol
    )
    with pytest.raises(ValueError, match="parameter count"):
        r14.validate_parameter_counts(
            ca8, {"text_encoder": 102976, "total": 1}, protocol
        )


def test_executor_rejects_corrupt_existing_png(tmp_path: Path, plan: list[dict]):
    small = plan[:1]
    identities = {
        "protocol_sha256": "protocol",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "abc",
        "architecture_arm": "ca8",
        "training_seed": 2026,
    }
    executor = r14.R14Executor(tmp_path, small, identities)
    executor.generate(lambda _row: np.zeros((32, 32, 3), dtype=np.uint8))
    png = next(tmp_path.glob("images/**/*.png"))
    png.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="PNG identity"):
        executor.generate(lambda _row: np.zeros((32, 32, 3), dtype=np.uint8))


def test_executor_rejects_incorrect_stored_pixel_hash(tmp_path: Path, plan: list[dict]):
    identities = {
        "protocol_sha256": "protocol",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "abc",
        "architecture_arm": "ca8",
        "training_seed": 2026,
    }
    executor = r14.R14Executor(tmp_path, plan[:1], identities)
    executor.generate(lambda _row: np.zeros((32, 32, 3), dtype=np.uint8))
    row = r14._read_jsonl(executor.manifest_path)[0]
    row["pixel_sha256"] = "0" * 64
    r14.atomic_jsonl(executor.manifest_path, [row])
    with pytest.raises(ValueError, match="pixel identity"):
        executor.generate(lambda _row: np.zeros((32, 32, 3), dtype=np.uint8))


def test_executor_rejects_incompatible_resume_identity(
    tmp_path: Path, plan: list[dict]
):
    base = {
        "protocol_sha256": "p1",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "abc",
        "architecture_arm": "ca8",
        "training_seed": 2026,
    }
    r14.R14Executor(tmp_path, plan[:1], base).generate(
        lambda _row: np.zeros((32, 32, 3), dtype=np.uint8)
    )
    changed = {**base, "protocol_sha256": "p2"}
    with pytest.raises(ValueError, match="incompatible"):
        r14.R14Executor(tmp_path, plan[:1], changed).generate(
            lambda _row: np.zeros((32, 32, 3), dtype=np.uint8)
        )


def test_efficiency_summary_rejects_incomplete_or_nonfinite_records():
    with pytest.raises(ValueError, match="six checkpoints"):
        r14.summarize_efficiency([])
    record = {
        "batch_size": 1,
        "raw_measured_seconds_per_batch": [1.0] * 10,
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 1,
    }
    with pytest.raises(ValueError, match="six checkpoints"):
        r14.summarize_efficiency([record] * 11)


def test_efficiency_executor_resumes_seven_valid_conditions(
    tmp_path: Path, protocol: dict
):
    _plan, checkpoints = r14.resolve_inputs(protocol)
    order = r14.efficiency_execution_order(checkpoints)
    executor = r14.EfficiencyExecutor(tmp_path, {"protocol_sha256": "p"}, order)
    payload = {
        "raw_measured_seconds_per_batch": [1.0] * 10,
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 1,
    }
    for condition in order[:7]:
        executor.publish(condition, payload)
    assert set(executor.completed()) == set(range(7))
    assert not (tmp_path / "r14_efficiency_summary.json").exists()


def test_efficiency_executor_rejects_corrupted_completed_record(
    tmp_path: Path, protocol: dict
):
    _plan, checkpoints = r14.resolve_inputs(protocol)
    order = r14.efficiency_execution_order(checkpoints)
    executor = r14.EfficiencyExecutor(tmp_path, {"protocol_sha256": "p"}, order)
    executor.publish(
        order[0],
        {
            "raw_measured_seconds_per_batch": [1.0] * 10,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 1,
        },
    )
    row = r14._read_json(executor.path(0))
    row["peak_reserved_bytes"] = 99
    r14.atomic_json(executor.path(0), row)
    with pytest.raises(ValueError, match="integrity"):
        executor.completed()


def test_efficiency_atomic_publish_failure_has_no_false_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, protocol: dict
):
    _plan, checkpoints = r14.resolve_inputs(protocol)
    order = r14.efficiency_execution_order(checkpoints)
    executor = r14.EfficiencyExecutor(tmp_path, {"protocol_sha256": "p"}, order)
    executor.state()
    original = r14.atomic_json
    monkeypatch.setattr(
        r14,
        "atomic_json",
        lambda path, payload: (
            (_ for _ in ()).throw(OSError("publication failed"))
            if path.name == "efficiency_00.json"
            else original(path, payload)
        ),
    )
    with pytest.raises(OSError, match="publication failed"):
        executor.publish(
            order[0],
            {
                "raw_measured_seconds_per_batch": [1.0] * 10,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
            },
        )
    assert executor.completed() == {}
    assert not executor.path(0).exists()


def test_frozen_protocol_hash_is_bound():
    assert r14.load_protocol(require_bound_hash=True)["status"] == "FROZEN"


def _conditions() -> list[dict]:
    return [
        {
            "ordinal": n,
            "training_seed": 2026 + n // 4,
            "architecture_arm": "ca8" if n % 2 == 0 else "ca8_ca16",
            "batch_size": 1 if n % 4 < 2 else 8,
            "checkpoint_sha256": f"sha-{n}",
        }
        for n in range(12)
    ]


def _parent_mocks(monkeypatch: pytest.MonkeyPatch, root: Path, fail: int | None = None):
    order, launched = _conditions(), []
    monkeypatch.setattr(r14, "PROTOCOL_SHA256", "frozen")
    monkeypatch.setattr(r14, "load_protocol", lambda **_kwargs: {"status": "FROZEN"})
    monkeypatch.setattr(r14, "_git_state", lambda: ("head", "exp/ca16-ablation", True))
    monkeypatch.setattr(r14, "resolve_inputs", lambda _protocol: ([], ()))
    monkeypatch.setattr(r14, "efficiency_execution_order", lambda _checkpoints: order)

    def run(args, **kwargs):
        if args[0] == "git":
            return SimpleNamespace(stdout="head\n")
        ordinal = int(args[args.index("--worker-ordinal") + 1])
        launched.append(ordinal)
        if ordinal == fail:
            raise subprocess.CalledProcessError(1, args)
        r14.EfficiencyExecutor(root, {"protocol_sha256": "frozen"}, order).publish(
            order[ordinal],
            {
                "raw_measured_seconds_per_batch": [1.0] * 10,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
            },
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(r14.subprocess, "run", run)
    return launched


def test_parent_worker_fresh_run_failure_resume_and_reentry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    launched = _parent_mocks(monkeypatch, tmp_path, fail=7)
    with pytest.raises(subprocess.CalledProcessError):
        r14.execute_efficiency(
            output_root=tmp_path, expected_eval_sha="head", expected_tag="tag"
        )
    assert (
        launched == list(range(8))
        and not (tmp_path / "r14_efficiency_summary.json").exists()
    )
    resumed = _parent_mocks(monkeypatch, tmp_path)
    assert (
        r14.execute_efficiency(
            output_root=tmp_path, expected_eval_sha="head", expected_tag="tag"
        )["status"]
        == "COMPLETE"
    )
    assert resumed == list(range(7, 12))
    repeated = _parent_mocks(monkeypatch, tmp_path)
    r14.execute_efficiency(
        output_root=tmp_path, expected_eval_sha="head", expected_tag="tag"
    )
    assert repeated == []


def test_parent_rejects_corruption_before_worker_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    launched = _parent_mocks(monkeypatch, tmp_path)
    r14.execute_efficiency(
        output_root=tmp_path, expected_eval_sha="head", expected_tag="tag"
    )
    row = r14._read_json(tmp_path / "efficiency_00.json")
    row["checkpoint_sha256"] = "bad"
    r14.atomic_json(tmp_path / "efficiency_00.json", row)
    blocked = _parent_mocks(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        r14.execute_efficiency(
            output_root=tmp_path, expected_eval_sha="head", expected_tag="tag"
        )
    assert launched and blocked == []
