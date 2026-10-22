"""Authorised-only scratch U64 training runner; CPU preflight is side-effect free."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.v2_matched_budget_training import AUTHENTIC_CACHE_SHA256, AUTHENTIC_MANIFEST_SHA256, TRAIN_COUNT, sha256_file
from tiny_avatar_diffusion.v2a_unconditional_native64 import EXPECTED_PARAMETERS, V2AUnconditionalNative64UNet, parameter_count

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "v2a_unconditional_native64_60k_seed20261021"
OUT = ROOT / "results/runs" / RUN_ID
# This is the authenticated cache location recorded by the historical V2-A run.
# It is read-only input, not a cache created or copied by this experiment.
CACHE = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_V2_NATIVE64\data\v2_authentic_native64_train_20261017")
MANIFEST = ROOT / "data/processed/dr01_v1/text/10k/train.jsonl"
TARGET = 60_000


@dataclass(frozen=True)
class Config:
    run_id: str = RUN_ID
    successful_updates: int = TARGET
    microbatch_size: int = 8
    gradient_accumulation: int = 4
    seed: int = 20_261_021
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    resolution: int = 64
    checkpoint_interval_updates: int = 5_000
    max_grad_norm: None = None

    def __post_init__(self) -> None:
        if self.successful_updates != 60_000 or self.resolution != 64:
            raise ValueError("U64 protocol is exactly 60k, RGB64")
        if (self.microbatch_size, self.gradient_accumulation) not in {(8, 4), (16, 2)}:
            raise ValueError("U64 batching must preserve effective batch 32 (8x4 or 16x2)")
        if (self.lr, self.weight_decay, self.ema_decay, self.max_grad_norm) != (2e-4, 0.0, 0.9999, None):
            raise ValueError("U64 optimizer/EMA contract mismatch")

    @property
    def effective_batch(self) -> int: return self.microbatch_size * self.gradient_accumulation


@dataclass
class Progress:
    successful_updates: int = 0
    attempted_optimizer_updates: int = 0
    amp_skipped_updates: int = 0
    microbatch_attempts: int = 0
    images_presented: int = 0
    offset: int = TRAIN_COUNT
    permutation: torch.Tensor | None = None
    losses: list[float] | None = None
    def __post_init__(self) -> None:
        if self.losses is None: self.losses = []


def cache_identity() -> dict[str, Any]:
    cache, integrity = CACHE / "train64_rgb_uint8.npy", CACHE / "integrity_report.json"
    if not cache.is_file() or not integrity.is_file() or sha256_file(cache) != AUTHENTIC_CACHE_SHA256 or sha256_file(MANIFEST) != AUTHENTIC_MANIFEST_SHA256:
        raise ValueError("authenticated V2-A cache or TRAIN manifest mismatch")
    report = json.loads(integrity.read_text())
    if report.get("shape") != [8388, 3, 64, 64] or report.get("cache_sha256") != AUTHENTIC_CACHE_SHA256 or not report.get("pixel_exact_independent_verification", {}).get("passed"):
        raise ValueError("native64 cache integrity contract failed")
    if len(_load_text_records(MANIFEST, "TRAIN")) != TRAIN_COUNT: raise ValueError("TRAIN identity count mismatch")
    return {"train_manifest": str(MANIFEST.relative_to(ROOT)), "train_manifest_sha256": AUTHENTIC_MANIFEST_SHA256, "cache": str(cache), "cache_sha256": AUTHENTIC_CACHE_SHA256, "cache_integrity_sha256": sha256_file(integrity), "cache_format": "npy uint8 [8388,3,64,64], mmap read-only; normalize per microbatch to [-1,1]"}


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]: return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_ema(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    for key, value in model.state_dict().items():
        if value.is_floating_point(): ema[key].mul_(decay).add_(value.detach().cpu(), alpha=1.0-decay)
        else: ema[key].copy_(value.detach().cpu())


def payload(config: Config, progress: Progress, model: torch.nn.Module, ema: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, sampler: torch.Generator, noise: torch.Generator, provenance: dict[str, Any]) -> dict[str, Any]:
    if progress.microbatch_attempts % config.gradient_accumulation: raise ValueError("cannot checkpoint inside accumulation window")
    return {"schema": "v2a_unconditional_native64_training.v1", "config": asdict(config), "provenance": provenance, "progress": asdict(progress), "model": clone_state(model), "ema": ema, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "sampler_state": sampler.get_state(), "noise_generator_state": noise.get_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as stream: torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_preflight() -> dict[str, Any]:
    config, provenance = Config(), cache_identity(); model = V2AUnconditionalNative64UNet()
    if parameter_count(model) != EXPECTED_PARAMETERS: raise ValueError("U64 parameter-count identity mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ema, sampler, noise = clone_state(model), torch.Generator().manual_seed(config.seed), torch.Generator().manual_seed(config.seed)
    state = payload(config, Progress(), model, ema, optimizer, torch.amp.GradScaler("cuda", enabled=False), sampler, noise, provenance)
    encoded = io.BytesIO(); torch.save(state, encoded); restored = torch.load(io.BytesIO(encoded.getvalue()), map_location="cpu", weights_only=False)
    clone = V2AUnconditionalNative64UNet(); clone.load_state_dict(restored["model"], strict=True)
    if not all(torch.equal(v, restored["ema"][k]) for k, v in clone_state(model).items()): raise RuntimeError("EMA serialization mismatch")
    return {"status": "CPU_PREFLIGHT_PASSED", "parameter_count": parameter_count(model), "config": asdict(config), "cache": provenance, "checkpoint_schema": state["schema"], "restore": "CPU disposable serialize/deserialize passed; no CUDA allocation or cache rebuild."}


def restore(path: Path, config: Config, provenance: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, sampler: torch.Generator, noise: torch.Generator) -> tuple[Progress, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "v2a_unconditional_native64_training.v1" or checkpoint.get("config") != asdict(config) or checkpoint.get("provenance") != provenance:
        raise ValueError("strict incompatible U64 resume rejection")
    model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
    sampler.set_state(checkpoint["sampler_state"]); noise.set_state(checkpoint["noise_generator_state"])
    random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"]); torch.set_rng_state(checkpoint["torch_rng"]); torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    progress, ema = Progress(**checkpoint["progress"]), checkpoint["ema"]
    if progress.successful_updates >= TARGET or progress.microbatch_attempts % config.gradient_accumulation:
        raise ValueError("invalid U64 resume counters")
    return progress, ema


def optimizer_window(losses: list[torch.Tensor], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, model: torch.nn.Module, ema: dict[str, torch.Tensor], progress: Progress, config: Config) -> tuple[bool, float]:
    if len(losses) != config.gradient_accumulation: raise ValueError("incomplete accumulation window")
    optimizer.zero_grad(set_to_none=True)
    for loss in losses: scaler.scale(loss / config.gradient_accumulation).backward()
    scaler.unscale_(optimizer); before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); progress.attempted_optimizer_updates += 1
    mean = float(torch.stack([loss.detach() for loss in losses]).mean().cpu())
    if float(scaler.get_scale()) < before:
        progress.amp_skipped_updates += 1; optimizer.zero_grad(set_to_none=True); return False, mean
    progress.successful_updates += 1; update_ema(ema, model, config.ema_decay); optimizer.zero_grad(set_to_none=True); return True, mean


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp"); temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8"); temporary.replace(path)


def smoke(microbatch_size: int) -> dict[str, Any]:
    """Short disposable fixed-effective-batch benchmark; never writes a file."""
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for U64 smoke")
    accumulation = 32 // microbatch_size
    config, provenance, device = Config(microbatch_size=microbatch_size, gradient_accumulation=accumulation), cache_identity(), torch.device("cuda")
    images = np.load(CACHE / "train64_rgb_uint8.npy", mmap_mode="r")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    model = V2AUnconditionalNative64UNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay); scaler = torch.amp.GradScaler("cuda")
    ema, sampler, noise, progress = clone_state(model), torch.Generator().manual_seed(config.seed), torch.Generator(device=device).manual_seed(config.seed), Progress()
    progress.permutation = torch.randperm(TRAIN_COUNT, generator=sampler); losses = []
    schedule = LinearDDPMSchedule().to(device)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); last_loss = float("nan")
    for _ in range(3):
        losses = []
        for _ in range(config.gradient_accumulation):
            if progress.offset + config.microbatch_size > TRAIN_COUNT:
                progress.permutation = torch.randperm(TRAIN_COUNT, generator=sampler); progress.offset = 0
            indices = progress.permutation[progress.offset:progress.offset + config.microbatch_size]
            progress.offset += config.microbatch_size; progress.microbatch_attempts += 1; progress.images_presented += config.microbatch_size
            batch = torch.from_numpy(images[indices.numpy()].copy()).to(torch.float32).div_(255).mul_(2).sub_(1).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                steps = torch.randint(0, 1000, (config.microbatch_size,), device=device, generator=noise); epsilon = torch.randn(batch.shape, device=device, generator=noise)
                losses.append(F.mse_loss(model(schedule.q_sample(batch, steps, epsilon), steps), epsilon))
        ok, last_loss = optimizer_window(losses, optimizer, scaler, model, ema, progress, config)
        if not ok or not np.isfinite(last_loss): raise RuntimeError("U64 disposable smoke failed")
    elapsed = time.perf_counter() - started
    return {"status": "GPU_SMOKE_PASSED", "loss": last_loss, "successful_updates": progress.successful_updates, "seconds_per_successful_update": elapsed / progress.successful_updates, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "amp_skipped_updates": progress.amp_skipped_updates, "microbatch_size": config.microbatch_size, "gradient_accumulation": config.gradient_accumulation, "effective_batch": config.effective_batch, "cache_sha256": provenance["cache_sha256"]}


def train(resume: Path | None, config: Config) -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if resume is None and OUT.exists(): raise FileExistsError("new U64 output directory already exists")
    provenance, device = cache_identity(), torch.device("cuda")
    records = _load_text_records(MANIFEST, "TRAIN"); images = np.load(CACHE / "train64_rgb_uint8.npy", mmap_mode="r")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    model = V2AUnconditionalNative64UNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay); scaler = torch.amp.GradScaler("cuda")
    ema, sampler, noise, progress = clone_state(model), torch.Generator().manual_seed(config.seed), torch.Generator(device=device).manual_seed(config.seed), Progress()
    if resume is not None: progress, ema = restore(resume, config, provenance, model, optimizer, scaler, sampler, noise)
    OUT.mkdir(parents=True, exist_ok=True); write_json(OUT / "training_config.json", {"config": asdict(config), "provenance": provenance, "initialization": "scratch; no conditional checkpoint loaded", "parameter_count": parameter_count(model)})
    schedule = LinearDDPMSchedule().to(device); initial = progress.successful_updates; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); metrics = OUT / "metrics.jsonl"
    while progress.successful_updates < TARGET:
        losses = []
        for _ in range(config.gradient_accumulation):
            if progress.permutation is None or progress.offset + config.microbatch_size > len(records): progress.permutation = torch.randperm(len(records), generator=sampler); progress.offset = 0
            indices = progress.permutation[progress.offset:progress.offset + config.microbatch_size]; progress.offset += config.microbatch_size; progress.microbatch_attempts += 1; progress.images_presented += config.microbatch_size
            batch = torch.from_numpy(images[indices.numpy()].copy()).to(torch.float32).div_(255).mul_(2).sub_(1).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                steps = torch.randint(0, 1000, (config.microbatch_size,), device=device, generator=noise); epsilon = torch.randn(batch.shape, device=device, generator=noise)
                losses.append(F.mse_loss(model(schedule.q_sample(batch, steps, epsilon), steps), epsilon))
        success, loss = optimizer_window(losses, optimizer, scaler, model, ema, progress, config); progress.losses = (progress.losses + [loss])[-100:]
        if success and (progress.successful_updates % 50 == 0 or progress.successful_updates % config.checkpoint_interval_updates == 0):
            elapsed = time.perf_counter()-started; rate = (progress.successful_updates-initial)/elapsed
            row = {"successful_updates": progress.successful_updates, "remaining_updates": TARGET-progress.successful_updates, "amp_skipped_updates": progress.amp_skipped_updates, "loss": loss, "rolling_loss_100": sum(progress.losses)/len(progress.losses), "learning_rate": config.lr, "elapsed_seconds": elapsed, "seconds_per_successful_update": elapsed/(progress.successful_updates-initial), "eta_seconds": (TARGET-progress.successful_updates)/rate}
            with metrics.open("a", encoding="utf8") as stream: stream.write(json.dumps(row)+"\n")
            write_json(OUT / "progress.json", row)
        if success and progress.successful_updates % config.checkpoint_interval_updates == 0:
            state = payload(config, progress, model, ema, optimizer, scaler, sampler, noise, provenance); atomic_save(OUT / f"checkpoint_success_{progress.successful_updates:06d}.pt", state); atomic_save(OUT / "latest.pt", state)
    state = payload(config, progress, model, ema, optimizer, scaler, sampler, noise, provenance); atomic_save(OUT / "latest.pt", state)
    ema_artifact = {"artifact_type": "v2a_unconditional_native64_ema", "selected_step": TARGET, "parameter_count": EXPECTED_PARAMETERS, "architecture": "V2-A image backbone without text encoder/CA8/CA16", "source_checkpoint_sha256": sha256_file(OUT / "latest.pt"), "model_state_dict": ema}; torch.save(ema_artifact, OUT / "ema_step60000.pt")
    result = {"status": "TARGET_REACHED", "successful_updates": progress.successful_updates, "amp_skipped_updates": progress.amp_skipped_updates, "images_presented": progress.images_presented, "wall_seconds": time.perf_counter()-started, "terminal_sha256": sha256_file(OUT / "latest.pt"), "ema_sha256": sha256_file(OUT / "ema_step60000.pt"), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}; write_json(OUT / "training_telemetry.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--resume", type=Path); parser.add_argument("--microbatch", type=int, choices=(8, 16), default=8); args = parser.parse_args()
    if sum((args.preflight, args.smoke, args.execute)) != 1: parser.error("select exactly one mode")
    if args.resume and not args.execute: parser.error("--resume requires --execute")
    selected = Config(microbatch_size=args.microbatch, gradient_accumulation=32 // args.microbatch)
    result = cpu_preflight() if args.preflight else smoke(args.microbatch) if args.smoke else train(args.resume, selected)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
