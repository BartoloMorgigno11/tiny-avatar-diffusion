"""MODEL-COMPUTE-ONLY FP32 training-step profile for Tiny U-Net V1."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.unet import TinyUNet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "dr02_gpu_profile" / "fp32_training"
BASE_BATCH_SIZES = (8, 16, 32, 64)
OPTIONAL_BATCH_SIZE = 128
OPTIONAL_BATCH_MEMORY_FRACTION = 0.60
WARMUP_STEPS = 5
MEASURED_STEPS = 20
RESOLUTION = 32
CHANNELS = 3


@dataclass(frozen=True)
class ProfileConfig:
    """Frozen engineering parameters for the FP32 compute-only profile."""

    warmup_steps: int = WARMUP_STEPS
    measured_steps: int = MEASURED_STEPS


def validate_config(config: ProfileConfig) -> None:
    """Reject nonsensical benchmark step counts before any GPU allocation."""
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.measured_steps <= 0:
        raise ValueError("measured_steps must be positive")


def images_per_second(batch_size: int, median_step_ms: float) -> float:
    """Return throughput based on the median synchronized full-step latency."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if median_step_ms <= 0.0:
        raise ValueError("median_step_ms must be positive")
    return batch_size / (median_step_ms / 1_000.0)


def should_try_optional_batch_128(
    batch_64_result: dict[str, Any], *, total_vram_mb: float
) -> bool:
    """Apply the frozen optional-128 policy after a successful batch-64 run."""
    return (
        batch_64_result["status"] == "success"
        and batch_64_result["peak_allocated_mb"] < OPTIONAL_BATCH_MEMORY_FRACTION
        * total_vram_mb
    )


def _full_training_step(
    model: nn.Module,
    schedule: LinearDDPMSchedule,
    optimizer: torch.optim.Optimizer,
    x_start: torch.Tensor,
) -> None:
    """Run the validated epsilon-prediction path for one FP32 optimizer step."""
    timesteps = torch.randint(
        0,
        schedule.betas.shape[0],
        (x_start.shape[0],),
        device=x_start.device,
    )
    noise = torch.randn_like(x_start)
    x_t = schedule.q_sample(x_start, timesteps, noise=noise)
    optimizer.zero_grad(set_to_none=True)
    predicted_noise = model(x_t, timesteps)
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()
    optimizer.step()


def _profile_batch(
    batch_size: int,
    *,
    device: torch.device,
    config: ProfileConfig,
    total_vram_mb: float,
) -> dict[str, Any]:
    """Profile one isolated batch size, returning an OOM row instead of raising."""
    model: TinyUNet | None = None
    schedule: LinearDDPMSchedule | None = None
    optimizer: torch.optim.Optimizer | None = None
    x_start: torch.Tensor | None = None
    try:
        model = TinyUNet(dropout=0.0).to(device)
        schedule = LinearDDPMSchedule(num_timesteps=1_000).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.0
        )
        x_start = torch.empty(
            batch_size, CHANNELS, RESOLUTION, RESOLUTION, device=device
        ).uniform_(-1.0, 1.0)

        for _ in range(config.warmup_steps):
            _full_training_step(model, schedule, optimizer, x_start)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_times_ms: list[float] = []
        for _ in range(config.measured_steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _full_training_step(model, schedule, optimizer, x_start)
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1_000.0)

        median_step_ms = statistics.median(step_times_ms)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        return {
            "batch_size": batch_size,
            "status": "success",
            "warmup_steps": config.warmup_steps,
            "measured_steps": config.measured_steps,
            "mean_step_ms": statistics.mean(step_times_ms),
            "median_step_ms": median_step_ms,
            "std_step_ms": statistics.pstdev(step_times_ms),
            "images_per_second": images_per_second(batch_size, median_step_ms),
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "total_vram_mb": total_vram_mb,
            "allocated_fraction": peak_allocated_mb / total_vram_mb,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "batch_size": batch_size,
            "status": "OOM",
            "warmup_steps": config.warmup_steps,
            "measured_steps": config.measured_steps,
            "mean_step_ms": None,
            "median_step_ms": None,
            "std_step_ms": None,
            "images_per_second": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "total_vram_mb": total_vram_mb,
            "allocated_fraction": None,
        }
    finally:
        del model, schedule, optimizer, x_start
        gc.collect()
        torch.cuda.empty_cache()


def write_profile_artifacts(
    output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Write only the summary and per-batch CSV required by Phase 8A."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "batch_size",
        "status",
        "warmup_steps",
        "measured_steps",
        "mean_step_ms",
        "median_step_ms",
        "std_step_ms",
        "images_per_second",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "total_vram_mb",
        "allocated_fraction",
    ]
    with (output_dir / "batch_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_fp32_training_profile(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_index: int = 0,
    config: ProfileConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen FP32 Tiny U-Net compute/memory profile on one CUDA GPU."""
    if config is None:
        config = ProfileConfig()
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 8A requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("device_index is not an available CUDA device")

    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    total_vram_mb = properties.total_memory / (1024**2)
    rows: list[dict[str, Any]] = []
    for batch_size in BASE_BATCH_SIZES:
        row = _profile_batch(
            batch_size,
            device=device,
            config=config,
            total_vram_mb=total_vram_mb,
        )
        rows.append(row)
        if row["status"] == "OOM":
            break

    batch_64_result = next(
        (row for row in rows if row["batch_size"] == 64), None
    )
    optional_batch_128_attempted = False
    if batch_64_result is not None and should_try_optional_batch_128(
        batch_64_result, total_vram_mb=total_vram_mb
    ):
        optional_batch_128_attempted = True
        rows.append(
            _profile_batch(
                OPTIONAL_BATCH_SIZE,
                device=device,
                config=config,
                total_vram_mb=total_vram_mb,
            )
        )

    summary = {
        "profile_type": "MODEL-COMPUTE-ONLY PROFILE",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_mb": total_vram_mb,
        "device_index": device_index,
        "precision": "FP32",
        "resolution": RESOLUTION,
        "model_parameter_count": sum(
            parameter.numel() for parameter in TinyUNet(dropout=0.0).parameters()
        ),
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "diffusion_timesteps": 1_000,
        "schedule": "linear",
        "target": "epsilon",
        "base_batch_sizes": list(BASE_BATCH_SIZES),
        "optional_batch_size": OPTIONAL_BATCH_SIZE,
        "optional_batch_memory_fraction": OPTIONAL_BATCH_MEMORY_FRACTION,
        "optional_batch_128_attempted": optional_batch_128_attempted,
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.measured_steps,
        "throughput_definition": "batch_size / median_step_seconds",
        "batch_results": rows,
    }
    write_profile_artifacts(output_dir, summary, rows)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the Phase 8A profile from the command line."""
    args = _parse_args()
    run_fp32_training_profile(
        output_dir=args.output_dir,
        device_index=args.device_index,
    )


if __name__ == "__main__":
    main()
