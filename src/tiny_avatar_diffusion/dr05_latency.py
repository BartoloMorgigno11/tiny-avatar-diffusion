"""Frozen DR05-D12 latency protocol helpers.

This module deliberately contains no model loading, image serialization, or
execution-binding updates.  The executable runner supplies the sampler.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

MIB = 1024 ** 2
WARMUPS = 3
MEASURED_RUNS = 10
SAMPLER = "ancestral_ddpm_fixed_small"
LATENCY_FILENAME = "latency_summary.json"


@dataclass(frozen=True)
class LatencyCondition:
    model_type: str
    conditional: bool
    batch_size: int
    namespace: str
    seed_base: int

    @property
    def key(self) -> str:
        return f"{self.model_type}_b{self.batch_size}"


CANONICAL_CONDITIONS = (
    LatencyCondition("conditional", True, 1, "latency_primary_b1", 602026),
    LatencyCondition("unconditional", False, 1, "latency_primary_b1", 602026),
    LatencyCondition("conditional", True, 16, "latency_secondary_b16", 612026),
    LatencyCondition("unconditional", False, 16, "latency_secondary_b16", 612026),
)


def validate_condition(condition: LatencyCondition, *, precision: str = "FP32", resolution: tuple[int, int] = (32, 32), diffusion_steps: int = 1000, sampler: str = SAMPLER, warmups: int = WARMUPS, measured_runs: int = MEASURED_RUNS) -> None:
    if condition not in CANONICAL_CONDITIONS or precision != "FP32" or resolution != (32, 32) or diffusion_steps != 1000 or sampler != SAMPLER or warmups != 3 or measured_runs != 10:
        raise ValueError("DR05-D12 requires exactly the frozen FP32, 32x32, T1000 canonical condition")


def benchmark_seeds(condition: LatencyCondition) -> tuple[list[int], list[int]]:
    return (list(range(condition.seed_base, condition.seed_base + WARMUPS)), list(range(condition.seed_base + WARMUPS, condition.seed_base + WARMUPS + MEASURED_RUNS)))


def measure_once(sample: Callable[[], torch.Tensor], *, device: torch.device, clock: Callable[[], float] = time.perf_counter) -> float:
    torch.cuda.synchronize(device)
    started = clock()
    output = sample()
    torch.cuda.synchronize(device)
    elapsed = clock() - started
    if output.shape[1:] != (3, 32, 32) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("invalid canonical sampling output")
    del output
    return elapsed


def summarize(seconds_per_batch: list[float], batch_size: int) -> dict[str, float]:
    if len(seconds_per_batch) != MEASURED_RUNS or batch_size not in (1, 16):
        raise ValueError("canonical summaries require ten B1 or B16 observations")
    median_batch = statistics.median(seconds_per_batch)
    per_image = [value / batch_size for value in seconds_per_batch]
    return {
        "median_seconds_per_batch": median_batch,
        "median_seconds_per_image": median_batch / batch_size,
        "mean_seconds_per_image": statistics.mean(per_image),
        "std_seconds_per_image": statistics.pstdev(per_image),
        "min_seconds_per_image": min(per_image),
        "max_seconds_per_image": max(per_image),
        "images_per_second": batch_size / median_batch,
    }


def benchmark_condition(sample_factory: Callable[[int], torch.Tensor], *, condition: LatencyCondition, device: torch.device, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    validate_condition(condition)
    warmup_seeds, measured_seeds = benchmark_seeds(condition)
    for seed in warmup_seeds:
        output = sample_factory(seed)
        del output
    torch.cuda.synchronize(device)
    seconds = [measure_once(lambda seed=seed: sample_factory(seed), device=device, clock=clock) for seed in measured_seeds]
    return {"warmup_seeds": warmup_seeds, "measured_seeds": measured_seeds, "raw_measured_seconds_per_batch": seconds, "statistics": summarize(seconds, condition.batch_size)}


def condition_record(condition: LatencyCondition, measured: dict[str, Any], *, checkpoint_path: str, checkpoint_sha256: str) -> dict[str, Any]:
    return {"model_type": condition.model_type, "conditional": condition.conditional, "batch_size": condition.batch_size, "precision": "FP32", "resolution": [32, 32], "diffusion_steps": 1000, "sampler": SAMPLER, "clip_denoised": True, "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha256, "role": "PRIMARY_HEADLINE" if condition.batch_size == 1 else "SECONDARY_ENGINEERING_THROUGHPUT", **measured}


def write_latency_summary(root: Path, artifact: dict[str, Any]) -> Path:
    output = root / LATENCY_FILENAME
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canonical latency artifact: {output}")
    root.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return output
