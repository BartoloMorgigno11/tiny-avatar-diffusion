from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from tiny_avatar_diffusion.dr05_latency import CANONICAL_CONDITIONS, LATENCY_FILENAME, benchmark_condition, condition_record, measure_once, summarize, validate_condition, write_latency_summary
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule


def test_warmups_are_excluded_and_measurement_geometry(monkeypatch):
    condition = CANONICAL_CONDITIONS[0]
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: calls.append("sync"))
    values = iter(range(100, 110))
    monkeypatch.setattr("tiny_avatar_diffusion.dr05_latency.measure_once", lambda sample, **_: (sample(), float(next(values)))[1])
    result = benchmark_condition(lambda seed: (calls.append(seed), torch.zeros(1, 3, 32, 32))[1], condition=condition, device=torch.device("cpu"))
    assert calls[:3] == [602026, 602027, 602028]
    assert result["measured_seeds"] == list(range(602029, 602039))
    assert result["raw_measured_seconds_per_batch"] == list(range(100, 110))


def test_measurement_synchronizes_around_sample(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("sync"))
    ticks = iter([1.0, 3.5])
    assert measure_once(lambda: (events.append("sample"), torch.zeros(1, 3, 32, 32))[1], device=torch.device("cpu"), clock=lambda: next(ticks)) == 2.5
    assert events == ["sync", "sample", "sync"]


def test_statistics_and_b16_math_are_exact():
    summary = summarize([float(value) for value in range(1, 11)], 16)
    assert summary["median_seconds_per_batch"] == 5.5
    assert summary["median_seconds_per_image"] == 5.5 / 16
    assert summary["images_per_second"] == 16 / 5.5
    assert summary["mean_seconds_per_image"] == 5.5 / 16


def test_canonical_guards_reject_noncanonical_values():
    with pytest.raises(ValueError): validate_condition(CANONICAL_CONDITIONS[0], diffusion_steps=10)
    with pytest.raises(ValueError): validate_condition(CANONICAL_CONDITIONS[0], precision="AMP")
    with pytest.raises(ValueError): validate_condition(CANONICAL_CONDITIONS[0], resolution=(64, 64))


def test_result_writer_is_fail_closed_and_preserves_siblings():
    root = Path(tempfile.mkdtemp())
    sibling = root / "conditional_b1.json"
    sibling.write_text("frozen")
    artifact = {"conditions": [condition_record(item, {"warmup_seeds": [], "measured_seeds": [], "raw_measured_seconds_per_batch": [], "statistics": {}}, checkpoint_path="x", checkpoint_sha256="y") for item in CANONICAL_CONDITIONS]}
    output = write_latency_summary(root, artifact)
    assert output.name == LATENCY_FILENAME and sibling.read_text() == "frozen"
    assert len(json.loads(output.read_text())["conditions"]) == 4
    with pytest.raises(FileExistsError): write_latency_summary(root, artifact)


def test_cuda_required_has_no_cpu_fallback(monkeypatch):
    from scripts import dr05_final_latency_benchmark as runner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        runner.run()


def test_conditional_context_is_encoded_once_for_a_complete_batch():
    class Text(nn.Module):
        def __init__(self): super().__init__(); self.anchor = nn.Parameter(torch.zeros(1)); self.calls = 0
        def forward(self, ids, mask): self.calls += 1; return torch.zeros(ids.shape[0], 1, 1)
    class UNet(nn.Module):
        def __init__(self): super().__init__(); self.anchor = nn.Parameter(torch.zeros(1)); self.calls = 0
        def forward(self, x, timestep, context, mask): self.calls += 1; return torch.zeros_like(x)
    text, model = Text(), UNet()
    output = conditional_sample(LinearDDPMSchedule(num_timesteps=3), text, model, torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long), generator=torch.Generator().manual_seed(1))
    assert output.shape == (1, 3, 32, 32)
    assert text.calls == 1 and model.calls == 3


def test_complete_four_condition_schema_and_binding_sentinel(tmp_path: Path):
    bindings = tmp_path / "data/processed/dr05_final_eval_v1/dr05_execution_bindings.json"
    bindings.parent.mkdir(parents=True)
    bindings.write_text('{"sentinel": true}')
    before = bindings.read_bytes()
    measured = {"warmup_seeds": [1, 2, 3], "measured_seeds": list(range(4, 14)), "raw_measured_seconds_per_batch": [1.0] * 10, "statistics": summarize([1.0] * 10, 1)}
    records = [condition_record(item, {**measured, "statistics": summarize([1.0] * 10, item.batch_size)}, checkpoint_path="checkpoint.pt", checkpoint_sha256="sha") for item in CANONICAL_CONDITIONS]
    assert {(row["model_type"], row["batch_size"]) for row in records} == {("conditional", 1), ("unconditional", 1), ("conditional", 16), ("unconditional", 16)}
    for row in records:
        assert {"model_type", "batch_size", "precision", "resolution", "diffusion_steps", "sampler", "checkpoint_path", "checkpoint_sha256", "raw_measured_seconds_per_batch"} <= row.keys()
        assert row["role"] == ("PRIMARY_HEADLINE" if row["batch_size"] == 1 else "SECONDARY_ENGINEERING_THROUGHPUT")
        assert "median_seconds_per_image" in row["statistics"]
        if row["batch_size"] == 16: assert {"median_seconds_per_batch", "images_per_second"} <= row["statistics"].keys()
    assert bindings.read_bytes() == before


def test_timed_helper_has_no_non_sampling_dependencies():
    import inspect
    import tiny_avatar_diffusion.dr05_latency as latency
    source = inspect.getsource(latency.measure_once).lower()
    for forbidden in ("png", "pil", "save", "evaluator", "kid", "fid", "lpips"):
        assert forbidden not in source
