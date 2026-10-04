from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiny_avatar_diffusion.dr05_final_executor import ExecutionIdentity, FinalQualityExecutor, load_jsonl


def _plan() -> list[dict[str, object]]:
    rows = []
    for index in range(32):
        split = "ID" if index < 16 else "OOD"
        rows.append({"planned_sample_index": index, "split": split, "reference_sample_id": f"{split}-{index % 2}", "prompt": f"prompt {index % 2}", "generation_namespace": "quality_generation", "batch_index": index // 16, "batch_seed": 502026 + index // 16, "sample_index_in_batch": index % 16})
    return rows


def _executor(tmp_path: Path, generator, metric=lambda *_args: {"kid": 0.1, "fid": 1.0}) -> FinalQualityExecutor:
    references = {"ID": [{"reference_sample_id": "ID-0"}, {"reference_sample_id": "ID-1"}], "OOD": [{"reference_sample_id": "OOD-0"}, {"reference_sample_id": "OOD-1"}]}
    return FinalQualityExecutor(root=tmp_path / "synthetic", plan=_plan(), identity=ExecutionIdentity("synthetic-plan", "plan-sha", "synthetic-meta", "meta-sha"), references=references,
        reference_loader=lambda item: np.full((32, 32, 3), len(item["reference_sample_id"]), dtype=np.uint8), generator=generator, metric=metric,
        checkpoint_sha={"conditional": "c", "unconditional": "u"}, checkpoint_path={"conditional": "c.pt", "unconditional": "u.pt"},
        expected_counts={"real_id": 2, "real_ood": 2, "conditional_id": 16, "conditional_ood": 16, "unconditional_id": 16, "unconditional_ood": 16})


def test_synthetic_lifecycle_is_resumable_and_does_not_touch_project_images(tmp_path: Path) -> None:
    calls: list[tuple[str, int, int]] = []
    def generator(model, rows, seed):
        calls.append((model, rows[0]["batch_index"], seed))
        return [np.full((32, 32, 3), row["planned_sample_index"] % 256, dtype=np.uint8) for row in rows]
    executor = _executor(tmp_path, generator)
    executor.canonicalize_references(); executor.generate_model("conditional"); executor.generate_model("unconditional")
    inventories = executor.audit(); results = executor.metrics(); summary = executor.summary({"synthetic": True})
    assert len(calls) == 4 and len(inventories) == 6 and len(results) == 6
    assert summary["FINAL_SCIENTIFIC_EVALUATION"] is True
    assert len(load_jsonl(executor.root / "manifests" / "conditional_id.jsonl")) == 16
    assert not Path("data/processed/dr01_v1/text/10k/id_test.jsonl").resolve().is_relative_to(executor.root.resolve())


def test_failed_batch_reuses_seed_and_never_regenerates_completed_batch(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []; fail = {1}
    def generator(_model, rows, seed):
        batch = rows[0]["batch_index"]; calls.append((batch, seed))
        if batch in fail: fail.remove(batch); raise RuntimeError("synthetic interruption")
        return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in rows]
    executor = _executor(tmp_path, generator)
    with pytest.raises(RuntimeError): executor.generate_model("conditional")
    assert calls == [(0, 502026), (1, 502027)]
    executor.generate_model("conditional")
    assert calls == [(0, 502026), (1, 502027), (1, 502027)]
    assert executor.state()["models"]["conditional"]["completed_batches"] == [0, 1]
    assert len(load_jsonl(executor.root / "manifests" / "conditional_id.jsonl")) == 16


def test_metrics_resume_without_generation_and_summary_requires_prerequisites(tmp_path: Path) -> None:
    generated = []
    def generator(model, rows, seed):
        generated.append((model, seed)); return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in rows]
    metric_calls: list[str] = []; fail = {"conditional:OOD"}
    def metric(model, split, _real, _generated):
        key = f"{model}:{split}"; metric_calls.append(key)
        if key in fail: fail.remove(key); raise RuntimeError("metric interruption")
        return {"kid": 0.1, "fid": 1.0}
    executor = _executor(tmp_path, generator, metric)
    with pytest.raises(RuntimeError): executor.summary({})
    executor.canonicalize_references(); executor.generate_model("conditional"); executor.generate_model("unconditional"); executor.audit()
    with pytest.raises(RuntimeError): executor.metrics()
    before = list(generated); executor.metrics()
    assert generated == before
    assert metric_calls.count("conditional:ID") == 1 and metric_calls.count("conditional:OOD") == 2


def test_complete_summary_and_identity_mismatch_are_protected(tmp_path: Path) -> None:
    generator = lambda _m, rows, _s: [np.zeros((32, 32, 3), dtype=np.uint8) for _ in rows]
    executor = _executor(tmp_path, generator); executor.canonicalize_references(); executor.generate_model("conditional"); executor.generate_model("unconditional"); executor.audit(); executor.metrics(); executor.summary({})
    with pytest.raises(FileExistsError): executor.state()
    mismatch = _executor(tmp_path, generator)
    mismatch.identity = ExecutionIdentity("different", "x", "y", "z")  # type: ignore[misc]
    with pytest.raises(ValueError): mismatch.state()
