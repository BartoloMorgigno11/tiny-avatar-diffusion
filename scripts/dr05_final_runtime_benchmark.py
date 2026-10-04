"""D24-only canonical FP32 DDPM runtime and memory benchmark."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

try:  # Supports both ``python scripts/...`` and package imports in tests.
    from scripts.dr05_materialize_canonical_checkpoints import (
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
except ModuleNotFoundError:
    from dr05_materialize_canonical_checkpoints import (  # type: ignore[no-redef]
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_runtime_benchmark.json"
BINDINGS = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json"
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
VOCAB = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "vocab.json"
CONDITIONAL = ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt"
UNCONDITIONAL = ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt"
EVALUATOR = ROOT / "artifacts" / "checkpoints" / "dr04_evaluator_epoch41.pt"
UNCONDITIONAL_SOURCE = ROOT / "results" / "runs" / "dr05_unconditional_main_10k_seed2026_r1" / "milestone_20000.pt"

PROTOCOL_SHA256 = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
PRE_BINDINGS_SHA256 = "e7215461790cb3bbcea89ccef579b9c91b02c8de7734c2c086654653c1d23c7e"
CONDITIONAL_SHA256 = "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"
UNCONDITIONAL_SHA256 = "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e"
EVALUATOR_SHA256 = "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"
UNCONDITIONAL_SOURCE_SHA256 = "a23436b8f3713d87374ef3ad847aadeffd38469fff89bcb169424539c9842129"
PROMPT_ID = "cs10002930376453395152"
PROMPT = "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"
PRIMARY = {"batch_size": 1, "namespace": "latency_primary_b1", "seed_base": 602026}
SECONDARY = {"batch_size": 16, "namespace": "latency_secondary_b16", "seed_base": 612026}
WARMUPS = 3
MEASURED = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_seeds(seed_base: int) -> tuple[list[int], list[int]]:
    return list(range(seed_base, seed_base + WARMUPS)), list(range(seed_base + WARMUPS, seed_base + WARMUPS + MEASURED))


def latency_summary(observations: list[dict[str, Any]], batch_size: int) -> dict[str, float]:
    seconds = [item["seconds"] for item in observations]
    per_image = [value / batch_size for value in seconds]
    return {
        "median_seconds_per_image": statistics.median(per_image),
        "mean_seconds_per_image": statistics.mean(per_image),
        "std_seconds_per_image": statistics.pstdev(per_image),
        "min_seconds_per_image": min(per_image),
        "max_seconds_per_image": max(per_image),
        "median_images_per_second": statistics.median([batch_size / value for value in seconds]),
        "mean_images_per_second": batch_size / statistics.mean(seconds),
        "std_images_per_second": statistics.pstdev([batch_size / value for value in seconds]),
        "min_images_per_second": min(batch_size / value for value in seconds),
        "max_images_per_second": max(batch_size / value for value in seconds),
    }


def memory_summary(observations: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "maximum_peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in observations),
        "median_peak_allocated_bytes": statistics.median(item["peak_allocated_bytes"] for item in observations),
        "maximum_peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in observations),
        "median_pre_sampling_allocated_bytes": statistics.median(item["pre_sampling_allocated_bytes"] for item in observations),
        "median_incremental_peak_allocated_bytes": statistics.median(item["incremental_peak_allocated_bytes"] for item in observations),
    }


def conditional_overhead(conditional: dict[str, float], unconditional: dict[str, float]) -> dict[str, float]:
    absolute = conditional["median_seconds_per_image"] - unconditional["median_seconds_per_image"]
    return {"absolute_median_seconds_per_image": absolute, "percent_median_overhead": 100 * absolute / unconditional["median_seconds_per_image"]}


def environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("D24 final benchmark requires CUDA")
    device = torch.device("cuda:0")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    smi_output = subprocess.check_output(["nvidia-smi"], text=True)
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", smi_output)
    return {
        "benchmark_device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "driver_version": driver.strip(), "driver_reported_cuda_compatibility": cuda_match.group(1) if cuda_match else None,
        "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__, "python_version": platform.python_version(),
    }


def _tokenizer() -> WhitespaceTokenizer:
    vocabulary = json.loads(VOCAB.read_text(encoding="utf-8"))["token_to_id"]
    return WhitespaceTokenizer(vocabulary)


def _sample_unconditional(schedule: LinearDDPMSchedule, model: torch.nn.Module, *, batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    x_t = torch.randn((batch_size, 3, 32, 32), device="cuda:0", dtype=torch.float32, generator=generator)
    with torch.inference_mode():
        for value in range(schedule.betas.shape[0] - 1, -1, -1):
            timesteps = torch.full((batch_size,), value, dtype=torch.long, device="cuda:0")
            noise = torch.randn(x_t.shape, device="cuda:0", dtype=torch.float32, generator=generator)
            x_t = schedule.p_sample_from_noise(x_t, timesteps, model(x_t, timesteps), noise=noise, clip_denoised=True)["sample"]
    return x_t


def _sample_conditional(schedule: LinearDDPMSchedule, text: torch.nn.Module, model: torch.nn.Module, tokenizer: WhitespaceTokenizer, *, batch_size: int, seed: int) -> torch.Tensor:
    encoded = tokenizer.encode(PROMPT)
    input_ids = torch.tensor([encoded.token_ids] * batch_size, dtype=torch.long, device="cuda:0")
    mask = torch.tensor([encoded.attention_mask] * batch_size, dtype=torch.long, device="cuda:0")
    return conditional_sample(schedule, text, model, input_ids, mask, generator=torch.Generator(device="cuda:0").manual_seed(seed), clip_denoised=True)


def _measure(sample: Callable[[], torch.Tensor], device: torch.device) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    pre = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = sample()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if output.shape[1:] != (3, 32, 32) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("invalid benchmark sampler output")
    del output
    return {"seconds": elapsed, "pre_sampling_allocated_bytes": pre, "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved, "incremental_peak_allocated_bytes": peak_allocated - pre}


def _benchmark_model(sample_factory: Callable[[int], torch.Tensor], *, device: torch.device, seed_base: int, batch_size: int) -> dict[str, Any]:
    warmup_seeds, measured_seeds = benchmark_seeds(seed_base)
    for seed in warmup_seeds:
        output = sample_factory(seed)
        del output
    torch.cuda.synchronize(device)
    rows = [{"seed": seed, **_measure(lambda seed=seed: sample_factory(seed), device)} for seed in measured_seeds]
    return {"warmup_seeds": warmup_seeds, "measured_seeds": measured_seeds, "observations": rows, "latency_summary": latency_summary(rows, batch_size), "memory_summary": memory_summary(rows)}


def update_hardware_bindings(environment_values: dict[str, Any]) -> str:
    before = json.loads(BINDINGS.read_text(encoding="utf-8"))
    after = json.loads(json.dumps(before))
    fields = ("benchmark_device", "gpu_name", "gpu_vram", "driver_version", "cuda_version", "pytorch_version", "python_version")
    values = {**environment_values, "gpu_vram": environment_values["gpu_vram_bytes"]}
    for field in fields:
        after["bindings"][field] = {"status": "resolved", "value": values[field]}
    changed = {key for key in before["bindings"] if before["bindings"][key] != after["bindings"][key]}
    if changed != set(fields):
        raise ValueError(f"unexpected execution-binding changes: {changed}")
    BINDINGS.write_bytes((json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return sha256_file(BINDINGS)


def run() -> dict[str, Any]:
    required = {PROTOCOL: PROTOCOL_SHA256, BINDINGS: PRE_BINDINGS_SHA256, CONDITIONAL: CONDITIONAL_SHA256, UNCONDITIONAL: UNCONDITIONAL_SHA256, EVALUATOR: EVALUATOR_SHA256, UNCONDITIONAL_SOURCE: UNCONDITIONAL_SOURCE_SHA256}
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {EVIDENCE}")
    for path, expected in required.items():
        if sha256_file(path) != expected:
            raise ValueError(f"unexpected frozen artifact SHA256: {path}")
    device = torch.device("cuda:0")
    before_environment = environment()
    schedule = LinearDDPMSchedule().to(device).eval()
    tokenizer = _tokenizer()
    order = ["conditional_b1", "unconditional_b1", "conditional_b16", "unconditional_b16"]
    conditional_artifact = torch.load(CONDITIONAL, map_location="cpu", weights_only=False)
    text, conditional = _conditional_models_from_artifact(conditional_artifact)
    text, conditional = text.to(device).eval(), conditional.to(device).eval()
    conditional_b1 = _benchmark_model(lambda seed: _sample_conditional(schedule, text, conditional, tokenizer, batch_size=1, seed=seed), device=device, seed_base=PRIMARY["seed_base"], batch_size=1)
    del text, conditional
    torch.cuda.empty_cache()
    unconditional_artifact = torch.load(UNCONDITIONAL, map_location="cpu", weights_only=False)
    unconditional = _unconditional_model_from_artifact(unconditional_artifact).to(device).eval()
    unconditional_b1 = _benchmark_model(lambda seed: _sample_unconditional(schedule, unconditional, batch_size=1, seed=seed), device=device, seed_base=PRIMARY["seed_base"], batch_size=1)
    del unconditional
    torch.cuda.empty_cache()
    conditional_artifact = torch.load(CONDITIONAL, map_location="cpu", weights_only=False)
    text, conditional = _conditional_models_from_artifact(conditional_artifact)
    text, conditional = text.to(device).eval(), conditional.to(device).eval()
    conditional_b16 = _benchmark_model(lambda seed: _sample_conditional(schedule, text, conditional, tokenizer, batch_size=16, seed=seed), device=device, seed_base=SECONDARY["seed_base"], batch_size=16)
    del text, conditional
    torch.cuda.empty_cache()
    unconditional_artifact = torch.load(UNCONDITIONAL, map_location="cpu", weights_only=False)
    unconditional = _unconditional_model_from_artifact(unconditional_artifact).to(device).eval()
    unconditional_b16 = _benchmark_model(lambda seed: _sample_unconditional(schedule, unconditional, batch_size=16, seed=seed), device=device, seed_base=SECONDARY["seed_base"], batch_size=16)
    after_environment = environment()
    if before_environment != after_environment:
        raise RuntimeError("benchmark environment changed during D24 execution")
    primary_conditional, primary_unconditional = conditional_b1["latency_summary"], unconditional_b1["latency_summary"]
    evidence = {
        "schema": "dr05_runtime_benchmark", "version": 1, "protocol_sha256": PROTOCOL_SHA256,
        "bindings_sha256_before": PRE_BINDINGS_SHA256,
        "checkpoints": {"conditional": {"path": CONDITIONAL.relative_to(ROOT).as_posix(), "sha256": CONDITIONAL_SHA256}, "unconditional": {"path": UNCONDITIONAL.relative_to(ROOT).as_posix(), "sha256": UNCONDITIONAL_SHA256}},
        "environment": before_environment,
        "primary": {"configuration": {"batch_size": 1, "precision": "FP32", "resolution": [32, 32], "timesteps": 1000, "sampler": "ancestral_ddpm_fixed_small", "clip_denoised": True, "warmups": WARMUPS, "measured_runs": MEASURED, "rng": {**PRIMARY, "warmup_seeds": benchmark_seeds(PRIMARY["seed_base"])[0], "measured_seeds": benchmark_seeds(PRIMARY["seed_base"])[1]}, "prompt": {"id": PROMPT_ID, "caption": PROMPT}, "execution_order": order[:2]}, "conditional": conditional_b1, "unconditional": unconditional_b1, "comparison": conditional_overhead(primary_conditional, primary_unconditional)},
        "secondary_batch16": {"configuration": {"batch_size": 16, "precision": "FP32", "resolution": [32, 32], "timesteps": 1000, "sampler": "ancestral_ddpm_fixed_small", "clip_denoised": True, "warmups": WARMUPS, "measured_runs": MEASURED, "rng": {**SECONDARY, "warmup_seeds": benchmark_seeds(SECONDARY["seed_base"])[0], "measured_seeds": benchmark_seeds(SECONDARY["seed_base"])[1]}, "prompt": {"id": PROMPT_ID, "caption": PROMPT}, "execution_order": order[2:]}, "conditional": conditional_b16, "unconditional": unconditional_b16},
        "compliance": {"id_val_images_opened": 0, "comp_val_images_opened": 0, "id_test_images_opened": 0, "final_ood_images_opened": 0, "saved_generated_images": 0, "evaluator_runs": 0, "kid_runs": 0, "fid_runs": 0, "lpips_runs": 0, "disk_serialization_inside_timed_regions": False},
    }
    EVIDENCE.write_bytes((json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    evidence["bindings_sha256_after"] = update_hardware_bindings(before_environment)
    EVIDENCE.write_bytes((json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return evidence


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
