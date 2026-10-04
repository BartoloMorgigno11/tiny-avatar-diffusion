"""Frozen DR05-D13 sampling-memory benchmark primitives.

This module deliberately measures only a single sampling invocation.  Model
construction, checkpoint I/O, and process isolation are responsibilities of
the CLI, so they cannot accidentally enter the measured window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


MIB = 1024**2
SAMPLER = "ancestral_ddpm_fixed_small"


@dataclass(frozen=True)
class MemoryCondition:
    model_type: str
    conditional: bool
    batch_size: int
    precision: str = "FP32"
    resolution: tuple[int, int] = (32, 32)
    diffusion_steps: int = 1000
    sampler: str = SAMPLER
    clip_denoised: bool = True

    @property
    def key(self) -> str:
        return f"{self.model_type}_b{self.batch_size}"


CANONICAL_CONDITIONS = (
    MemoryCondition("conditional", True, 1),
    MemoryCondition("unconditional", False, 1),
    MemoryCondition("conditional", True, 16),
    MemoryCondition("unconditional", False, 16),
)


def canonical_condition(key: str) -> MemoryCondition:
    for condition in CANONICAL_CONDITIONS:
        if condition.key == key:
            return condition
    raise ValueError(f"unknown canonical memory condition: {key}")


def validate_canonical_condition(condition: MemoryCondition) -> None:
    expected = canonical_condition(condition.key)
    if condition != expected:
        raise ValueError(
            "canonical D13 conditions require the frozen model type, FP32, "
            "32x32 resolution, T=1000, and ancestral fixed-small DDPM sampler"
        )


def bytes_to_mib(value: int) -> float:
    return value / MIB


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("DR05-D13 canonical sampling-memory benchmark requires CUDA; CPU is not supported")
    return torch.device("cuda:0")


def measure_sampling_memory(
    sample: Callable[[], torch.Tensor], *, device: torch.device
) -> dict[str, Any]:
    """Measure the frozen D13 window without baseline subtraction.

    The caller must already have loaded/moved the model and prepared its stable
    state.  The output remains live through peak reads, because final GPU image
    memory belongs to the sampling footprint.
    """
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = sample()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "primary_metric": "peak_cuda_allocated_memory",
        "secondary_metric": "peak_cuda_reserved_memory",
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": bytes_to_mib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": bytes_to_mib(peak_reserved),
        "_output": output,
    }


def result_record(
    condition: MemoryCondition,
    *,
    device: torch.device,
    gpu_name: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    provenance: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Build the D13 result schema; only the CLI serializes it after sampling."""
    validate_canonical_condition(condition)
    return {
        "schema": "dr05_final_sampling_memory_v1",
        **asdict(condition),
        "device": str(device),
        "gpu_name": gpu_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "primary_metric": measurement["primary_metric"],
        "secondary_metric": measurement["secondary_metric"],
        "peak_allocated_bytes": measurement["peak_allocated_bytes"],
        "peak_allocated_mib": measurement["peak_allocated_mib"],
        "peak_reserved_bytes": measurement["peak_reserved_bytes"],
        "peak_reserved_mib": measurement["peak_reserved_mib"],
        "provenance": provenance,
    }
