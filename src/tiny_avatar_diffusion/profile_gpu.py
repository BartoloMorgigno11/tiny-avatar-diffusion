"""MODEL-COMPUTE-ONLY GPU training-step profiles for Tiny U-Net V1."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_debug_tiny_overfit import (
    create_conditional_debug_optimizer,
    select_text_conditioning,
)
from tiny_avatar_diffusion.conditional_training import conditional_diffusion_loss
from tiny_avatar_diffusion.debug_tiny_overfit import (
    load_train_records,
    preload_images,
    select_tiny_subset,
)
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, TinyUNet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "dr02_gpu_profile" / "fp32_training"
AMP_OUTPUT_DIR = ROOT / "reports" / "dr02_gpu_profile" / "amp_parity"
SAMPLING_OUTPUT_DIR = ROOT / "reports" / "dr02_gpu_profile" / "ancestral_sampling"
BASE_BATCH_SIZES = (8, 16, 32, 64)
OPTIONAL_BATCH_SIZE = 128
OPTIONAL_BATCH_MEMORY_FRACTION = 0.60
AMP_PILOT_BATCH_SIZE = 32
PILOT_SEED = 2026
WARMUP_STEPS = 5
MEASURED_STEPS = 20
RESOLUTION = 32
CHANNELS = 3
PRECISIONS = ("FP32", "AMP_FP16")
SAMPLING_BATCH_SIZES = (1, 32)
SAMPLING_WARMUP_FORWARDS = 5
SAMPLING_REPETITIONS = 2
SAMPLING_SEED = 2026
CONDITIONAL_BATCH_SIZES = (4, 8, 16, 32)
CONDITIONAL_MEASURED_STEPS = 10
CONDITIONAL_SAFE_RESERVED_FRACTION = 0.85
PARITY_PARAMETER_NAMES = (
    "time_embedding.mlp.0.weight",
    "input_projection.weight",
    "encoder16_block1.conv1.weight",
    "bottleneck_block1.conv1.weight",
    "decoder16_block1.conv1.weight",
    "output_projection.weight",
)
MEASUREMENT_FIELDS = (
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
)
AMP_MEASUREMENT_FIELDS = (
    "precision",
    *MEASUREMENT_FIELDS,
    "grad_scaler_initial_scale",
    "grad_scaler_final_scale",
    "warmup_skipped_steps",
    "measured_skipped_steps",
)
SAMPLING_MEASUREMENT_FIELDS = (
    "batch_size",
    "precision",
    "repetition",
    "status",
    "total_seconds",
    "seconds_per_image",
    "images_per_second",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "total_vram_mb",
    "allocated_fraction",
    "final_shape",
    "final_finite",
    "final_min",
    "final_max",
    "final_mean",
    "final_std",
)


@dataclass(frozen=True)
class ProfileConfig:
    """Frozen engineering parameters for the FP32 compute-only profile."""

    warmup_steps: int = WARMUP_STEPS
    measured_steps: int = MEASURED_STEPS


@dataclass(frozen=True)
class SamplingProfileConfig:
    """Frozen engineering parameters for the ancestral-sampling profile."""

    warmup_model_forwards: int = SAMPLING_WARMUP_FORWARDS
    repetitions: int = SAMPLING_REPETITIONS


@dataclass(frozen=True)
class ConditionalProfileConfig:
    """DR03 engineering-only full conditional-training profile settings."""

    batch_sizes: tuple[int, ...] = CONDITIONAL_BATCH_SIZES
    warmup_steps: int = WARMUP_STEPS
    measured_steps: int = CONDITIONAL_MEASURED_STEPS
    seed: int = PILOT_SEED


def validate_config(config: ProfileConfig) -> None:
    """Reject nonsensical benchmark step counts before any GPU allocation."""
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.measured_steps <= 0:
        raise ValueError("measured_steps must be positive")


def validate_sampling_config(config: SamplingProfileConfig) -> None:
    """Reject invalid fixed-profile counts before any GPU allocation."""
    if config.warmup_model_forwards < 0:
        raise ValueError("warmup_model_forwards must be non-negative")
    if config.repetitions <= 0:
        raise ValueError("repetitions must be positive")


def validate_conditional_profile_config(config: ConditionalProfileConfig) -> None:
    """Validate only the scoped DR03 engineering-pilot controls."""
    if not config.batch_sizes or any(batch_size <= 0 for batch_size in config.batch_sizes):
        raise ValueError("batch_sizes must contain positive values")
    if tuple(sorted(set(config.batch_sizes))) != config.batch_sizes:
        raise ValueError("batch_sizes must be unique and increasing")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.measured_steps <= 0:
        raise ValueError("measured_steps must be positive")


def validate_sampling_batch_size(batch_size: int) -> None:
    """Restrict Phase 8C to its approved B=1 and B=32 measurements."""
    if batch_size not in SAMPLING_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {SAMPLING_BATCH_SIZES}")


def images_per_second(batch_size: int, median_step_ms: float) -> float:
    """Return throughput based on the median synchronized full-step latency."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if median_step_ms <= 0.0:
        raise ValueError("median_step_ms must be positive")
    return batch_size / (median_step_ms / 1_000.0)


def seconds_per_image(batch_size: int, total_seconds: float) -> float:
    """Return full-loop latency per generated image."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total_seconds <= 0.0:
        raise ValueError("total_seconds must be positive")
    return total_seconds / batch_size


def sampling_images_per_second(batch_size: int, total_seconds: float) -> float:
    """Return full-loop sampling throughput for one batch."""
    return 1.0 / seconds_per_image(batch_size, total_seconds)


def validate_precision(precision: str) -> None:
    """Restrict the pilot to its frozen FP32 and AMP-FP16 policies."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")


def backend_precision_metadata() -> dict[str, bool]:
    """Record, without changing, the process-wide TF32 backend flags."""
    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def compare_precision_results(
    fp32: dict[str, Any], amp: dict[str, Any]
) -> dict[str, float]:
    """Derive the frozen latency, throughput, and memory comparisons."""
    return {
        "speedup": fp32["median_step_ms"] / amp["median_step_ms"],
        "throughput_gain_percent": (
            amp["images_per_second"] / fp32["images_per_second"] - 1.0
        )
        * 100.0,
        "allocated_memory_reduction_percent": (
            1.0 - amp["peak_allocated_mb"] / fp32["peak_allocated_mb"]
        )
        * 100.0,
        "reserved_memory_reduction_percent": (
            1.0 - amp["peak_reserved_mb"] / fp32["peak_reserved_mb"]
        )
        * 100.0,
    }


def compare_sampling_precision_results(
    fp32: dict[str, Any], amp: dict[str, Any]
) -> dict[str, float]:
    """Compare paired full-loop FP32 and AMP ancestral-sampling results."""
    return {
        "speedup": fp32["median_total_seconds"] / amp["median_total_seconds"],
        "throughput_gain_percent": (
            amp["images_per_second"] / fp32["images_per_second"] - 1.0
        )
        * 100.0,
        "allocated_memory_reduction_percent": (
            1.0 - amp["peak_allocated_mb"] / fp32["peak_allocated_mb"]
        )
        * 100.0,
        "reserved_memory_reduction_percent": (
            1.0 - amp["peak_reserved_mb"] / fp32["peak_reserved_mb"]
        )
        * 100.0,
    }


def should_try_optional_batch_128(
    batch_64_result: dict[str, Any], *, total_vram_mb: float
) -> bool:
    """Apply the frozen optional-128 policy after a successful batch-64 run."""
    return (
        batch_64_result["status"] == "success"
        and batch_64_result["peak_allocated_mb"] < OPTIONAL_BATCH_MEMORY_FRACTION
        * total_vram_mb
    )


def _training_step(
    model: nn.Module,
    schedule: LinearDDPMSchedule,
    optimizer: torch.optim.Optimizer,
    x_start: torch.Tensor,
    *,
    precision: str,
    scaler: torch.amp.GradScaler | None,
) -> bool:
    """Run one shared FP32 or AMP-FP16 optimizer step."""
    validate_precision(precision)
    optimizer.zero_grad(set_to_none=True)
    timesteps = torch.randint(
        0,
        schedule.betas.shape[0],
        (x_start.shape[0],),
        device=x_start.device,
    )
    noise = torch.randn_like(x_start)
    x_t = schedule.q_sample(x_start, timesteps, noise=noise)
    if precision == "FP32":
        predicted_noise = model(x_t, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        return False

    if scaler is None:
        raise ValueError("AMP_FP16 requires a GradScaler")
    scale_before = float(scaler.get_scale())
    with torch.amp.autocast("cuda", dtype=torch.float16):
        predicted_noise = model(x_t, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) < scale_before


def _oom_result(
    batch_size: int,
    *,
    precision: str,
    config: ProfileConfig,
    total_vram_mb: float,
) -> dict[str, Any]:
    """Return the explicit OOM row used by the preserved Phase 8A policy."""
    return {
        "precision": precision,
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
        "grad_scaler_initial_scale": None,
        "grad_scaler_final_scale": None,
        "warmup_skipped_steps": 0,
        "measured_skipped_steps": 0,
    }


def _profile_training_run(
    batch_size: int,
    *,
    precision: str,
    device: torch.device,
    config: ProfileConfig,
    total_vram_mb: float,
    seed: int | None = None,
    record_oom: bool = False,
) -> dict[str, Any]:
    """Profile one precision policy with the shared measurement semantics."""
    validate_precision(precision)
    model: TinyUNet | None = None
    schedule: LinearDDPMSchedule | None = None
    optimizer: torch.optim.Optimizer | None = None
    scaler: torch.amp.GradScaler | None = None
    x_start: torch.Tensor | None = None
    try:
        if seed is not None:
            torch.manual_seed(seed)
        model = TinyUNet(dropout=0.0).to(device=device, dtype=torch.float32)
        schedule = LinearDDPMSchedule(num_timesteps=1_000).to(
            device=device, dtype=torch.float32
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.0
        )
        if precision == "AMP_FP16":
            scaler = torch.amp.GradScaler("cuda")
        x_start = torch.empty(
            batch_size,
            CHANNELS,
            RESOLUTION,
            RESOLUTION,
            device=device,
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)
        initial_scale = float(scaler.get_scale()) if scaler is not None else None

        warmup_skipped_steps = 0
        for _ in range(config.warmup_steps):
            warmup_skipped_steps += int(
                _training_step(
                    model,
                    schedule,
                    optimizer,
                    x_start,
                    precision=precision,
                    scaler=scaler,
                )
            )
        if not optimizer.state:
            raise RuntimeError("Warm-up did not materialize AdamW optimizer state")

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured_skipped_steps = 0
        step_times_ms: list[float] = []
        for _ in range(config.measured_steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            measured_skipped_steps += int(
                _training_step(
                    model,
                    schedule,
                    optimizer,
                    x_start,
                    precision=precision,
                    scaler=scaler,
                )
            )
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1_000.0)

        median_step_ms = statistics.median(step_times_ms)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        return {
            "precision": precision,
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
            "grad_scaler_initial_scale": initial_scale,
            "grad_scaler_final_scale": (
                float(scaler.get_scale()) if scaler is not None else None
            ),
            "warmup_skipped_steps": warmup_skipped_steps,
            "measured_skipped_steps": measured_skipped_steps,
        }
    except torch.cuda.OutOfMemoryError:
        if not record_oom:
            raise
        return _oom_result(
            batch_size,
            precision=precision,
            config=config,
            total_vram_mb=total_vram_mb,
        )
    finally:
        del model, schedule, optimizer, scaler, x_start
        gc.collect()
        torch.cuda.empty_cache()


def _run_parity_probe(device: torch.device) -> dict[str, Any]:
    """Compare FP32 and AMP on identical weights and corruption tensors."""
    fp32_model: TinyUNet | None = None
    amp_model: TinyUNet | None = None
    schedule: LinearDDPMSchedule | None = None
    fp32_optimizer: torch.optim.Optimizer | None = None
    amp_optimizer: torch.optim.Optimizer | None = None
    try:
        torch.manual_seed(PILOT_SEED)
        fp32_model = TinyUNet(dropout=0.0).to(device=device, dtype=torch.float32)
        amp_model = TinyUNet(dropout=0.0).to(device=device, dtype=torch.float32)
        amp_model.load_state_dict(fp32_model.state_dict())
        schedule = LinearDDPMSchedule(num_timesteps=1_000).to(
            device=device, dtype=torch.float32
        )
        fp32_optimizer = torch.optim.AdamW(
            fp32_model.parameters(), lr=1e-4, weight_decay=0.0
        )
        amp_optimizer = torch.optim.AdamW(
            amp_model.parameters(), lr=1e-4, weight_decay=0.0
        )
        generator = torch.Generator(device=device).manual_seed(PILOT_SEED)
        x_start = torch.empty(
            AMP_PILOT_BATCH_SIZE,
            CHANNELS,
            RESOLUTION,
            RESOLUTION,
            device=device,
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0, generator=generator)
        timesteps = torch.randint(
            0,
            schedule.betas.shape[0],
            (AMP_PILOT_BATCH_SIZE,),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            x_start.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        x_t = schedule.q_sample(x_start, timesteps, noise=noise)

        fp32_optimizer.zero_grad(set_to_none=True)
        fp32_prediction = fp32_model(x_t, timesteps)
        fp32_loss_tensor = F.mse_loss(fp32_prediction, noise)
        fp32_loss_tensor.backward()

        amp_optimizer.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            amp_prediction = amp_model(x_t, timesteps)
            amp_loss_tensor = F.mse_loss(amp_prediction, noise)
        scaler.scale(amp_loss_tensor).backward()
        scaler.unscale_(amp_optimizer)

        fp32_parameters = dict(fp32_model.named_parameters())
        amp_parameters = dict(amp_model.named_parameters())
        gradients: dict[str, dict[str, float | bool | None]] = {}
        for name in PARITY_PARAMETER_NAMES:
            fp32_gradient = fp32_parameters[name].grad
            amp_gradient = amp_parameters[name].grad
            if fp32_gradient is None or amp_gradient is None:
                gradients[name] = {
                    "fp32_finite": False,
                    "amp_finite": False,
                    "fp32_nonzero": False,
                    "amp_nonzero": False,
                    "fp32_grad_norm": None,
                    "amp_unscaled_grad_norm": None,
                    "cosine_similarity": None,
                }
                continue
            fp32_gradient_float = fp32_gradient.float().flatten()
            amp_gradient_float = amp_gradient.float().flatten()
            fp32_norm = torch.linalg.vector_norm(fp32_gradient_float)
            amp_norm = torch.linalg.vector_norm(amp_gradient_float)
            gradients[name] = {
                "fp32_finite": bool(torch.isfinite(fp32_gradient).all()),
                "amp_finite": bool(torch.isfinite(amp_gradient).all()),
                "fp32_nonzero": bool(torch.count_nonzero(fp32_gradient)),
                "amp_nonzero": bool(torch.count_nonzero(amp_gradient)),
                "fp32_grad_norm": float(fp32_norm),
                "amp_unscaled_grad_norm": float(amp_norm),
                "cosine_similarity": float(
                    F.cosine_similarity(
                        fp32_gradient_float, amp_gradient_float, dim=0
                    )
                ),
            }

        fp32_loss = float(fp32_loss_tensor)
        amp_loss = float(amp_loss_tensor)
        absolute_difference = abs(fp32_loss - amp_loss)
        return {
            "batch_size": AMP_PILOT_BATCH_SIZE,
            "same_initial_state": all(
                torch.equal(fp32_parameters[name], amp_parameters[name])
                for name in fp32_parameters
            ),
            "shared_input_shape": list(x_start.shape),
            "fp32_output_finite": bool(torch.isfinite(fp32_prediction).all()),
            "amp_output_finite": bool(torch.isfinite(amp_prediction).all()),
            "fp32_loss_finite": bool(torch.isfinite(fp32_loss_tensor)),
            "amp_loss_finite": bool(torch.isfinite(amp_loss_tensor)),
            "fp32_loss": fp32_loss,
            "amp_loss": amp_loss,
            "absolute_loss_difference": absolute_difference,
            "relative_loss_difference": absolute_difference / abs(fp32_loss),
            "representative_gradients": gradients,
            "all_representative_gradients_finite": all(
                metric["fp32_finite"] and metric["amp_finite"]
                for metric in gradients.values()
            ),
            "all_representative_gradients_nonzero": all(
                metric["fp32_nonzero"] and metric["amp_nonzero"]
                for metric in gradients.values()
            ),
        }
    finally:
        del fp32_model, amp_model, schedule, fp32_optimizer, amp_optimizer
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
    with (output_dir / "batch_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_amp_pilot_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    parity: dict[str, Any],
) -> None:
    """Write the three frozen Phase 8B artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "precision_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=AMP_MEASUREMENT_FIELDS)
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
        measurement = _profile_training_run(
            batch_size,
            precision="FP32",
            device=device,
            config=config,
            total_vram_mb=total_vram_mb,
            record_oom=True,
        )
        row = {field: measurement[field] for field in MEASUREMENT_FIELDS}
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
        measurement = _profile_training_run(
            OPTIONAL_BATCH_SIZE,
            precision="FP32",
            device=device,
            config=config,
            total_vram_mb=total_vram_mb,
            record_oom=True,
        )
        rows.append({field: measurement[field] for field in MEASUREMENT_FIELDS})

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


def run_amp_parity_profile(
    *,
    output_dir: Path = AMP_OUTPUT_DIR,
    device_index: int = 0,
    config: ProfileConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen B=32 FP32 versus AMP-FP16 parity and benefit pilot."""
    if config is None:
        config = ProfileConfig()
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 8B requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("device_index is not an available CUDA device")

    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    total_vram_mb = properties.total_memory / (1024**2)
    parity = _run_parity_probe(device)
    gc.collect()
    torch.cuda.empty_cache()
    fp32_result = _profile_training_run(
        AMP_PILOT_BATCH_SIZE,
        precision="FP32",
        device=device,
        config=config,
        total_vram_mb=total_vram_mb,
        seed=PILOT_SEED,
    )
    amp_result = _profile_training_run(
        AMP_PILOT_BATCH_SIZE,
        precision="AMP_FP16",
        device=device,
        config=config,
        total_vram_mb=total_vram_mb,
        seed=PILOT_SEED,
    )
    rows = [fp32_result, amp_result]
    comparison = compare_precision_results(fp32_result, amp_result)
    summary = {
        "profile_type": "MODEL-COMPUTE-ONLY AMP FP16 PARITY + BENEFIT PILOT",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_mb": total_vram_mb,
        "device_index": device_index,
        **backend_precision_metadata(),
        "precisions": list(PRECISIONS),
        "batch_size": AMP_PILOT_BATCH_SIZE,
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
        "pilot_seed": PILOT_SEED,
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.measured_steps,
        "throughput_definition": "batch_size / median_step_seconds",
        "precision_results": rows,
        "comparison": comparison,
    }
    write_amp_pilot_artifacts(output_dir, summary, rows, parity)
    return summary


def _sampling_oom_result(
    batch_size: int,
    *,
    precision: str,
    total_vram_mb: float,
) -> dict[str, Any]:
    """Record an OOM without attempting a larger Phase 8C batch."""
    return {
        "batch_size": batch_size,
        "precision": precision,
        "status": "OOM",
        "run_results": [],
        "mean_total_seconds": None,
        "median_total_seconds": None,
        "std_total_seconds": None,
        "seconds_per_image": None,
        "images_per_second": None,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
        "total_vram_mb": total_vram_mb,
        "allocated_fraction": None,
    }


def _sample_with_precision(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    precision: str,
) -> torch.Tensor:
    """Predict noise while keeping FP32 weights in both precision policies."""
    if precision == "FP32":
        return model(x_t, timesteps)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        return model(x_t, timesteps)


def _profile_ancestral_sampling_configuration(
    batch_size: int,
    *,
    precision: str,
    device: torch.device,
    config: SamplingProfileConfig,
    total_vram_mb: float,
) -> dict[str, Any]:
    """Measure two complete validated ancestral DDPM sampling loops."""
    validate_sampling_batch_size(batch_size)
    validate_precision(precision)
    model: TinyUNet | None = None
    schedule: LinearDDPMSchedule | None = None
    x_t: torch.Tensor | None = None
    try:
        torch.manual_seed(SAMPLING_SEED)
        model = TinyUNet(dropout=0.0).to(device=device, dtype=torch.float32)
        model.eval()
        schedule = LinearDDPMSchedule(num_timesteps=1_000).to(
            device=device, dtype=torch.float32
        )
        generator = torch.Generator(device=device).manual_seed(SAMPLING_SEED)
        x_t = torch.randn(
            batch_size,
            CHANNELS,
            RESOLUTION,
            RESOLUTION,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        warmup_timesteps = torch.full(
            (batch_size,),
            schedule.betas.shape[0] - 1,
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            for _ in range(config.warmup_model_forwards):
                _sample_with_precision(
                    model, x_t, warmup_timesteps, precision=precision
                )
        torch.cuda.synchronize(device)

        noise_predictor = partial(_sample_with_precision, model, precision=precision)
        expected_shape = (batch_size, CHANNELS, RESOLUTION, RESOLUTION)

        run_results: list[dict[str, Any]] = []
        for repetition in range(1, config.repetitions + 1):
            torch.manual_seed(SAMPLING_SEED + repetition)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            final_sample = schedule.sample_loop_from_noise_predictor(
                x_t, noise_predictor
            )
            torch.cuda.synchronize(device)
            total_seconds = time.perf_counter() - start
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
            if tuple(final_sample.shape) != expected_shape:
                raise RuntimeError(
                    "ancestral sampling returned shape "
                    f"{tuple(final_sample.shape)}, expected {expected_shape}"
                )
            finite = bool(torch.isfinite(final_sample).all())
            if not finite:
                raise RuntimeError("ancestral sampling produced NaN or Inf")
            run_results.append(
                {
                    "batch_size": batch_size,
                    "precision": precision,
                    "repetition": repetition,
                    "status": "success",
                    "total_seconds": total_seconds,
                    "seconds_per_image": seconds_per_image(batch_size, total_seconds),
                    "images_per_second": sampling_images_per_second(
                        batch_size, total_seconds
                    ),
                    "peak_allocated_mb": peak_allocated_mb,
                    "peak_reserved_mb": peak_reserved_mb,
                    "total_vram_mb": total_vram_mb,
                    "allocated_fraction": peak_allocated_mb / total_vram_mb,
                    "final_shape": list(final_sample.shape),
                    "final_finite": finite,
                    "final_min": float(final_sample.min()),
                    "final_max": float(final_sample.max()),
                    "final_mean": float(final_sample.mean()),
                    "final_std": float(final_sample.std()),
                }
            )
            del final_sample

        total_seconds = [item["total_seconds"] for item in run_results]
        median_total_seconds = statistics.median(total_seconds)
        peak_allocated_mb = max(item["peak_allocated_mb"] for item in run_results)
        peak_reserved_mb = max(item["peak_reserved_mb"] for item in run_results)
        return {
            "batch_size": batch_size,
            "precision": precision,
            "status": "success",
            "run_results": run_results,
            "mean_total_seconds": statistics.mean(total_seconds),
            "median_total_seconds": median_total_seconds,
            "std_total_seconds": statistics.pstdev(total_seconds),
            "seconds_per_image": seconds_per_image(batch_size, median_total_seconds),
            "images_per_second": sampling_images_per_second(
                batch_size, median_total_seconds
            ),
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "total_vram_mb": total_vram_mb,
            "allocated_fraction": peak_allocated_mb / total_vram_mb,
        }
    except torch.cuda.OutOfMemoryError:
        return _sampling_oom_result(
            batch_size, precision=precision, total_vram_mb=total_vram_mb
        )
    finally:
        del model, schedule, x_t
        gc.collect()
        torch.cuda.empty_cache()


def write_sampling_profile_artifacts(
    output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Write only the Phase 8C summary and per-run sampling CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "sampling_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLING_MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_ancestral_sampling_profile(
    *,
    output_dir: Path = SAMPLING_OUTPUT_DIR,
    device_index: int = 0,
    config: SamplingProfileConfig | None = None,
) -> dict[str, Any]:
    """Profile full T=1000 ancestral DDPM sampling; not image-quality evaluation."""
    if config is None:
        config = SamplingProfileConfig()
    validate_sampling_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 8C requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("device_index is not an available CUDA device")

    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    total_vram_mb = properties.total_memory / (1024**2)
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for batch_size in SAMPLING_BATCH_SIZES:
        batch_results = []
        for precision in PRECISIONS:
            result = _profile_ancestral_sampling_configuration(
                batch_size,
                precision=precision,
                device=device,
                config=config,
                total_vram_mb=total_vram_mb,
            )
            batch_results.append(result)
            results.append(result)
            rows.extend(result["run_results"])
        if any(result["status"] == "OOM" for result in batch_results):
            break

    comparisons: dict[str, dict[str, float]] = {}
    for batch_size in SAMPLING_BATCH_SIZES:
        paired = [result for result in results if result["batch_size"] == batch_size]
        if len(paired) == 2 and all(result["status"] == "success" for result in paired):
            fp32, amp = paired
            comparisons[str(batch_size)] = compare_sampling_precision_results(fp32, amp)

    batching_efficiency: dict[str, float] = {}
    for precision in PRECISIONS:
        by_batch = {
            result["batch_size"]: result
            for result in results
            if result["precision"] == precision and result["status"] == "success"
        }
        if all(batch_size in by_batch for batch_size in SAMPLING_BATCH_SIZES):
            batching_efficiency[precision] = (
                by_batch[32]["images_per_second"] / by_batch[1]["images_per_second"]
            )

    summary = {
        "profile_type": (
            "COMPUTE/MEMORY PROFILE ONLY — NOT A GENERATIVE-QUALITY EXPERIMENT"
        ),
        "not_a_generative_quality_experiment": True,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_mb": total_vram_mb,
        "device_index": device_index,
        **backend_precision_metadata(),
        "model_parameter_count": sum(
            parameter.numel() for parameter in TinyUNet(dropout=0.0).parameters()
        ),
        "model_weights_dtype": "torch.float32",
        "resolution": RESOLUTION,
        "channels": CHANNELS,
        "diffusion_timesteps": 1_000,
        "schedule": "linear",
        "sampler": "ancestral DDPM",
        "posterior_variance": "fixed-small",
        "final_step_noise": False,
        "model_mode": "eval",
        "gradient_mode": "torch.no_grad",
        "sampling_batch_sizes": list(SAMPLING_BATCH_SIZES),
        "precision_policies": list(PRECISIONS),
        "warmup_model_forwards": config.warmup_model_forwards,
        "repetitions": config.repetitions,
        "starting_noise_seed": SAMPLING_SEED,
        "reverse_noise_seed_base": SAMPLING_SEED + 1,
        "throughput_definition": "batch_size / median_total_seconds",
        "configuration_results": results,
        "fp32_to_amp_comparisons": comparisons,
        "batching_efficiency": batching_efficiency,
    }
    write_sampling_profile_artifacts(output_dir, summary, rows)
    return summary


def _conditional_profile_batch(
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cycle the real deterministic TRAIN-only subset to a physical batch size."""
    indices = torch.arange(batch_size) % images.shape[0]
    return (
        images[indices].to(device),
        input_ids[indices].to(device),
        attention_mask[indices].to(device),
    )


def _conditional_profile_training_step(
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    schedule: LinearDDPMSchedule,
    optimizer: torch.optim.Optimizer,
    x_start: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    precision: str,
    scaler: torch.amp.GradScaler | None,
) -> tuple[bool, bool, bool]:
    """Execute the full joint conditional step under one engineering precision mode."""
    validate_precision(precision)
    optimizer.zero_grad(set_to_none=True)
    if precision == "FP32":
        result = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, input_ids, attention_mask
        )
        result.loss.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in [*text_encoder.parameters(), *denoiser.parameters()]
        )
        optimizer.step()
        return bool(torch.isfinite(result.loss)), finite_gradients, False

    if scaler is None:
        raise ValueError("AMP_FP16 requires a GradScaler")
    scale_before = float(scaler.get_scale())
    with torch.amp.autocast("cuda", dtype=torch.float16):
        result = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, input_ids, attention_mask
        )
    scaler.scale(result.loss).backward()
    scaler.unscale_(optimizer)
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in [*text_encoder.parameters(), *denoiser.parameters()]
    )
    scaler.step(optimizer)
    scaler.update()
    return (
        bool(torch.isfinite(result.loss)),
        finite_gradients,
        float(scaler.get_scale()) < scale_before,
    )


def _conditional_oom_result(
    batch_size: int,
    *,
    precision: str,
    config: ConditionalProfileConfig,
    total_vram_mb: float,
) -> dict[str, Any]:
    """Serialize a clean OOM result for one conditional batch candidate."""
    return {
        "precision": precision,
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
        "reserved_fraction": None,
        "finite_loss": False,
        "finite_gradients": False,
        "grad_scaler_initial_scale": None,
        "grad_scaler_final_scale": None,
        "warmup_skipped_steps": 0,
        "measured_skipped_steps": 0,
    }


def _profile_conditional_training_run(
    batch_size: int,
    *,
    precision: str,
    device: torch.device,
    config: ConditionalProfileConfig,
    total_vram_mb: float,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    record_oom: bool = True,
) -> dict[str, Any]:
    """Profile full joint DR03 training steps on real 10k TRAIN records."""
    validate_precision(precision)
    text_encoder: TextConditioningEncoder | None = None
    denoiser: ConditionalTinyUNet | None = None
    schedule: LinearDDPMSchedule | None = None
    optimizer: torch.optim.Optimizer | None = None
    scaler: torch.amp.GradScaler | None = None
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    try:
        torch.manual_seed(config.seed)
        text_encoder = TextConditioningEncoder().to(device=device, dtype=torch.float32)
        denoiser = ConditionalTinyUNet(dropout=0.0).to(device=device, dtype=torch.float32)
        schedule = LinearDDPMSchedule(num_timesteps=1_000).to(device=device, dtype=torch.float32)
        optimizer = create_conditional_debug_optimizer(text_encoder, denoiser)
        if precision == "AMP_FP16":
            scaler = torch.amp.GradScaler("cuda")
        batch = _conditional_profile_batch(
            images, input_ids, attention_mask, batch_size, device=device
        )
        initial_scale = float(scaler.get_scale()) if scaler is not None else None
        warmup_skipped_steps = 0
        for _ in range(config.warmup_steps):
            _, _, skipped = _conditional_profile_training_step(
                text_encoder,
                denoiser,
                schedule,
                optimizer,
                *batch,
                precision=precision,
                scaler=scaler,
            )
            warmup_skipped_steps += int(skipped)
        if not optimizer.state:
            raise RuntimeError("Warm-up did not materialize AdamW optimizer state")

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_times_ms: list[float] = []
        finite_losses = True
        finite_gradients = True
        measured_skipped_steps = 0
        for _ in range(config.measured_steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            finite_loss, finite_gradients_step, skipped = _conditional_profile_training_step(
                text_encoder,
                denoiser,
                schedule,
                optimizer,
                *batch,
                precision=precision,
                scaler=scaler,
            )
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1_000.0)
            finite_losses = finite_losses and finite_loss
            finite_gradients = finite_gradients and finite_gradients_step
            measured_skipped_steps += int(skipped)

        median_step_ms = statistics.median(step_times_ms)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        return {
            "precision": precision,
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
            "reserved_fraction": peak_reserved_mb / total_vram_mb,
            "finite_loss": finite_losses,
            "finite_gradients": finite_gradients,
            "grad_scaler_initial_scale": initial_scale,
            "grad_scaler_final_scale": float(scaler.get_scale()) if scaler else None,
            "warmup_skipped_steps": warmup_skipped_steps,
            "measured_skipped_steps": measured_skipped_steps,
        }
    except torch.cuda.OutOfMemoryError:
        if not record_oom:
            raise
        return _conditional_oom_result(
            batch_size,
            precision=precision,
            config=config,
            total_vram_mb=total_vram_mb,
        )
    finally:
        del text_encoder, denoiser, schedule, optimizer, scaler, batch
        gc.collect()
        torch.cuda.empty_cache()


def _conditional_amp_parity_probe(
    *,
    batch_size: int,
    device: torch.device,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    """Check conditional FP32/AMP health from identical weights and corruption."""
    fp32_text: TextConditioningEncoder | None = None
    fp32_denoiser: ConditionalTinyUNet | None = None
    amp_text: TextConditioningEncoder | None = None
    amp_denoiser: ConditionalTinyUNet | None = None
    try:
        torch.manual_seed(seed)
        fp32_text = TextConditioningEncoder().to(device)
        fp32_denoiser = ConditionalTinyUNet(dropout=0.0).to(device)
        amp_text = TextConditioningEncoder().to(device)
        amp_denoiser = ConditionalTinyUNet(dropout=0.0).to(device)
        amp_text.load_state_dict(fp32_text.state_dict())
        amp_denoiser.load_state_dict(fp32_denoiser.state_dict())
        schedule = LinearDDPMSchedule().to(device)
        x_start, ids, mask = _conditional_profile_batch(
            images, input_ids, attention_mask, batch_size, device=device
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        timesteps = torch.randint(0, 1_000, (batch_size,), device=device, generator=generator)
        noise = torch.randn(x_start.shape, device=device, generator=generator)
        fp32_result = conditional_diffusion_loss(
            fp32_text,
            fp32_denoiser,
            schedule,
            x_start,
            ids,
            mask,
            timesteps=timesteps,
            noise=noise,
        )
        fp32_result.loss.backward()
        amp_optimizer = create_conditional_debug_optimizer(amp_text, amp_denoiser)
        scaler = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            amp_result = conditional_diffusion_loss(
                amp_text,
                amp_denoiser,
                schedule,
                x_start,
                ids,
                mask,
                timesteps=timesteps,
                noise=noise,
            )
        scaler.scale(amp_result.loss).backward()
        scaler.unscale_(amp_optimizer)
        amp_parameters = [*amp_text.parameters(), *amp_denoiser.parameters()]
        return {
            "batch_size": batch_size,
            "same_initial_state": all(
                torch.equal(left, right)
                for left, right in zip(fp32_text.parameters(), amp_text.parameters(), strict=True)
            )
            and all(
                torch.equal(left, right)
                for left, right in zip(fp32_denoiser.parameters(), amp_denoiser.parameters(), strict=True)
            ),
            "fp32_loss": float(fp32_result.loss),
            "amp_loss": float(amp_result.loss),
            "fp32_loss_finite": bool(torch.isfinite(fp32_result.loss)),
            "amp_loss_finite": bool(torch.isfinite(amp_result.loss)),
            "fp32_output_finite": bool(torch.isfinite(fp32_result.predicted_noise).all()),
            "amp_output_finite": bool(torch.isfinite(amp_result.predicted_noise).all()),
            "amp_gradients_finite": all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in amp_parameters
            ),
            "amp_gradients_nonzero": any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in amp_parameters
            ),
            "grad_scaler_scale": float(scaler.get_scale()),
        }
    finally:
        del fp32_text, fp32_denoiser, amp_text, amp_denoiser
        gc.collect()
        torch.cuda.empty_cache()


def _write_conditional_profile_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    fp32_rows: list[dict[str, Any]],
    amp_rows: list[dict[str, Any]],
    parity: dict[str, Any],
) -> None:
    """Write the concise machine-readable DR03 engineering-pilot artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "amp_parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )
    fields = tuple(fp32_rows[0]) if fp32_rows else tuple(amp_rows[0])
    for filename, rows in (("fp32_results.csv", fp32_rows), ("amp_results.csv", amp_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run_conditional_gpu_pilot(
    *,
    output_dir: Path = ROOT / "reports" / "dr03_conditional_gpu_pilot",
    device_index: int = 0,
    config: ConditionalProfileConfig | None = None,
) -> dict[str, Any]:
    """Measure full joint DR03 training steps on real TRAIN data in FP32 and AMP."""
    if config is None:
        config = ConditionalProfileConfig()
    validate_conditional_profile_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("DR03 conditional GPU pilot requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("device_index is not an available CUDA device")
    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    total_vram_mb = properties.total_memory / (1024**2)
    selected = select_tiny_subset(load_train_records())
    images = preload_images(selected)
    input_ids, attention_mask = select_text_conditioning(selected)

    fp32_rows: list[dict[str, Any]] = []
    for batch_size in config.batch_sizes:
        row = _profile_conditional_training_run(
            batch_size,
            precision="FP32",
            device=device,
            config=config,
            total_vram_mb=total_vram_mb,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        fp32_rows.append(row)
        if row["status"] == "OOM" or row["reserved_fraction"] >= CONDITIONAL_SAFE_RESERVED_FRACTION:
            break

    safe_rows = [
        row
        for row in fp32_rows
        if row["status"] == "success"
        and row["finite_loss"]
        and row["finite_gradients"]
        and row["reserved_fraction"] <= CONDITIONAL_SAFE_RESERVED_FRACTION
    ]
    if not safe_rows:
        raise RuntimeError("No FP32 conditional batch met the engineering safety criterion")
    same_batch_size = safe_rows[-1]["batch_size"]
    parity = _conditional_amp_parity_probe(
        batch_size=same_batch_size,
        device=device,
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        seed=config.seed,
    )
    amp_rows: list[dict[str, Any]] = []
    amp_health = all(
        parity[field]
        for field in (
            "fp32_loss_finite",
            "amp_loss_finite",
            "fp32_output_finite",
            "amp_output_finite",
            "amp_gradients_finite",
            "amp_gradients_nonzero",
        )
    )
    if amp_health:
        for batch_size in config.batch_sizes:
            row = _profile_conditional_training_run(
                batch_size,
                precision="AMP_FP16",
                device=device,
                config=config,
                total_vram_mb=total_vram_mb,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            amp_rows.append(row)
            if row["status"] == "OOM" or row["reserved_fraction"] >= CONDITIONAL_SAFE_RESERVED_FRACTION:
                break

    summary = {
        "profile_type": "DR03 CONDITIONAL FULL-TRAINING ENGINEERING PILOT ONLY",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_mb": total_vram_mb,
        "device": str(device),
        "dataset": "10k",
        "split": "TRAIN",
        "subset_size": len(selected),
        "seed": config.seed,
        "resolution": RESOLUTION,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "optimizer_policy": "ENGINEERING PILOT ONLY",
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.measured_steps,
        "text_encoder_parameter_count": sum(parameter.numel() for parameter in TextConditioningEncoder().parameters()),
        "conditional_denoiser_parameter_count": sum(parameter.numel() for parameter in ConditionalTinyUNet().parameters()),
        "complete_system_parameter_count": sum(parameter.numel() for parameter in TextConditioningEncoder().parameters()) + sum(parameter.numel() for parameter in ConditionalTinyUNet().parameters()),
        "fp32_results": fp32_rows,
        "amp_same_batch_size": same_batch_size,
        "amp_health_passed": amp_health,
        "amp_results": amp_rows,
        "safe_reserved_memory_fraction": CONDITIONAL_SAFE_RESERVED_FRACTION,
    }
    _write_conditional_profile_artifacts(output_dir, summary, fp32_rows, amp_rows, parity)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fp32", "amp-parity", "ancestral-sampling", "conditional-pilot"),
        default="fp32",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run one explicitly selected GPU profiling phase."""
    args = _parse_args()
    if args.mode == "conditional-pilot":
        run_conditional_gpu_pilot(
            output_dir=args.output_dir or ROOT / "reports" / "dr03_conditional_gpu_pilot",
            device_index=args.device_index,
        )
    elif args.mode == "amp-parity":
        run_amp_parity_profile(
            output_dir=args.output_dir or AMP_OUTPUT_DIR,
            device_index=args.device_index,
        )
    elif args.mode == "ancestral-sampling":
        run_ancestral_sampling_profile(
            output_dir=args.output_dir or SAMPLING_OUTPUT_DIR,
            device_index=args.device_index,
        )
    else:
        run_fp32_training_profile(
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            device_index=args.device_index,
        )


if __name__ == "__main__":
    main()
