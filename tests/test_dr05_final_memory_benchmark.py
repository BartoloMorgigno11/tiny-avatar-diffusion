from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from scripts import dr05_final_memory_benchmark as cli
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.dr05_efficiency import (
    CANONICAL_CONDITIONS,
    MemoryCondition,
    bytes_to_mib,
    measure_sampling_memory,
    result_record,
    validate_canonical_condition,
)


def test_memory_api_order_primary_secondary_and_no_baseline_subtraction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: calls.append("allocated") or 3 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: calls.append("reserved") or 4 * 1024**2)

    def sample() -> torch.Tensor:
        calls.append("sample")
        return torch.zeros(1, 3, 32, 32)

    measured = measure_sampling_memory(sample, device=torch.device("cuda:0"))
    assert calls == ["sync", "reset", "sample", "sync", "allocated", "reserved"]
    assert measured["peak_allocated_bytes"] == 3 * 1024**2
    assert measured["peak_reserved_bytes"] == 4 * 1024**2
    assert measured["primary_metric"] == "peak_cuda_allocated_memory"
    assert measured["secondary_metric"] == "peak_cuda_reserved_memory"
    assert "pre_sampling_allocated_bytes" not in measured
    assert "incremental_peak_allocated_bytes" not in measured


def test_mib_conversion_and_result_schema() -> None:
    condition = CANONICAL_CONDITIONS[0]
    record = result_record(
        condition, device=torch.device("cuda:0"), gpu_name="fake GPU", checkpoint_path="checkpoint.pt",
        checkpoint_sha256="hash", provenance={"python_version": "test"},
        measurement={"primary_metric": "peak_cuda_allocated_memory", "secondary_metric": "peak_cuda_reserved_memory", "peak_allocated_bytes": 2 * 1024**2, "peak_allocated_mib": bytes_to_mib(2 * 1024**2), "peak_reserved_bytes": 3 * 1024**2, "peak_reserved_mib": bytes_to_mib(3 * 1024**2)},
    )
    assert bytes_to_mib(1536 * 1024) == 1.5
    assert record["peak_allocated_mib"] == 2.0
    assert record["primary_metric"] == "peak_cuda_allocated_memory"
    assert record["secondary_metric"] == "peak_cuda_reserved_memory"
    for field in ("model_type", "conditional", "batch_size", "precision", "resolution", "diffusion_steps", "sampler", "device", "gpu_name", "checkpoint_path", "checkpoint_sha256", "provenance"):
        assert field in record


@pytest.mark.parametrize("condition", CANONICAL_CONDITIONS)
def test_only_frozen_canonical_conditions_are_accepted(condition: MemoryCondition) -> None:
    validate_canonical_condition(condition)


@pytest.mark.parametrize(
    "condition",
    [
        MemoryCondition("conditional", True, 1, precision="AMP"),
        MemoryCondition("conditional", True, 1, resolution=(64, 64)),
        MemoryCondition("conditional", True, 1, diffusion_steps=4),
        MemoryCondition("conditional", True, 4),
        MemoryCondition("unconditional", True, 1),
    ],
)
def test_canonical_validation_rejects_unfair_configuration(condition: MemoryCondition) -> None:
    with pytest.raises(ValueError):
        validate_canonical_condition(condition)


def test_dispatch_uses_one_subprocess_per_canonical_condition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, check: calls.append(command))
    cli.dispatch_conditions(tmp_path / "isolated")
    assert len(calls) == 4
    assert [command[3] for command in calls] == [item.key for item in CANONICAL_CONDITIONS]
    assert all(command[2] == "--run-condition" and command[-2] == "--output" for command in calls)


def test_canonical_benchmark_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        cli.run_condition(CANONICAL_CONDITIONS[0], Path("unused"))


def test_synthetic_conditional_path_encodes_once_and_runs_each_reverse_step() -> None:
    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.weight * torch.ones(ids.shape[0], 1, 1)

    class Denoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, image: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return image * self.weight * 0

    text, denoiser, schedule = Text(), Denoiser(), LinearDDPMSchedule(num_timesteps=3)
    result = conditional_sample(schedule, text, denoiser, torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long))
    assert result.shape == (2, 3, 32, 32)
    assert text.calls == 1
    assert denoiser.calls == 3
