"""Resumable D17 unconditional MAIN trainer; importing never starts training."""
from __future__ import annotations

import copy
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_lr_pilot import PHYSICAL_BATCH_SIZE, SEED, _load_text_records
from tiny_avatar_diffusion.conditional_main_training import (
    TrainingProgress,
    _nested_exact,
    _restore_rng_state,
    _rng_state,
    atomic_save_checkpoint,
)
from tiny_avatar_diffusion.debug_tiny_overfit import preload_images
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.unet import TinyUNet


ROOT = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "dr05_unconditional_main_training"
CHECKPOINT_SCHEMA = "dr05_unconditional_main_training_v1"
CANONICAL_RUN_ID = "dr05_unconditional_main_10k_seed2026"
MILESTONES = (5_000, 10_000, 15_000, 20_000)
EXPECTED_PARAMETERS = 4_595_779


@dataclass(frozen=True)
class UnconditionalMainConfig:
    """Frozen D17 controls; test limits are execution limits, never config changes."""

    run_id: str = CANONICAL_RUN_ID
    dataset_variant: str = "10k"
    train_count: int = 8_388
    resolution: tuple[int, int] = (32, 32)
    channels: int = 3
    physical_batch_size: int = PHYSICAL_BATCH_SIZE
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    seed: int = SEED
    diffusion_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    precision: str = "AMP_FP16"
    successful_steps: int = 20_000
    ema_decay: float = 0.9999
    milestones: tuple[int, ...] = MILESTONES

    def __post_init__(self) -> None:
        frozen = (
            self.dataset_variant == "10k" and self.train_count == 8_388
            and self.resolution == (32, 32) and self.channels == 3
            and self.physical_batch_size == 32 and self.learning_rate == 2e-4
            and self.weight_decay == 0.0 and self.seed == 2026
            and self.diffusion_timesteps == 1_000 and self.beta_start == 1e-4
            and self.beta_end == 0.02 and self.precision == "AMP_FP16"
            and self.successful_steps == 20_000 and self.ema_decay == 0.9999
            and self.milestones == MILESTONES
        )
        if not self.run_id or not frozen:
            raise ValueError("D17 unconditional MAIN controls are frozen")


class UnconditionalEMA:
    """Single-module counterpart to the validated conditional EMA semantics."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.shadow = self._clone(model.state_dict())

    @staticmethod
    def _clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in state.items()}

    def update(self, model: torch.nn.Module, *, decay: float) -> None:
        raw = model.state_dict()
        if self.shadow.keys() != raw.keys():
            raise ValueError("EMA and model state dictionaries differ")
        with torch.no_grad():
            for name, value in raw.items():
                if value.is_floating_point():
                    self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                else:
                    self.shadow[name].copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._clone(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = self._clone(state)


def unconditional_diffusion_loss(
    model: TinyUNet, schedule: LinearDDPMSchedule, images: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """The D17 unweighted pixel-space DDPM epsilon objective."""
    timesteps = torch.randint(0, schedule.betas.shape[0], (images.shape[0],), device=images.device, generator=generator)
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    return F.mse_loss(model(schedule.q_sample(images, timesteps, noise=noise), timesteps), noise)


def finalize_unconditional_amp_attempt(
    *, scaler: Any, optimizer: torch.optim.Optimizer, gradients_finite: bool,
    progress: TrainingProgress, ema: UnconditionalEMA, model: TinyUNet, ema_decay: float,
) -> bool:
    """Finalize one AMP attempt; only an actual optimizer update is successful."""
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    skipped = float(scaler.get_scale()) < before
    if not gradients_finite and not skipped:
        raise FloatingPointError("non-finite gradients were not skipped by GradScaler")
    if skipped:
        progress.skipped_step_count += 1
        return True
    ema.update(model, decay=ema_decay)
    progress.step += 1
    return False


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def build_unconditional_checkpoint(
    *, config: UnconditionalMainConfig, progress: TrainingProgress, model: TinyUNet,
    ema: UnconditionalEMA, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    sampler_generator: torch.Generator, corruption_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": config.run_id,
        "config": asdict(config),
        "progress": {
            "step": progress.step, "images_seen": progress.images_seen,
            "batch_offset": progress.batch_offset, "permutation": progress.permutation,
            "train_losses": progress.train_losses, "skipped_step_count": progress.skipped_step_count,
            "sampler_generator_state": sampler_generator.get_state(),
            "corruption_generator_state": corruption_generator.get_state(),
        },
        "tiny_unet_state_dict": copy.deepcopy(model.state_dict()),
        "ema_tiny_unet_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "rng_state": _rng_state(),
        "provenance": {
            "git_commit": _git_head(),
            "train_manifest": TRAIN_MANIFEST.relative_to(ROOT).as_posix(),
            "split_role": "TRAIN",
            "canonical_checkpoint_policy": "EMA_at_exactly_20000_successful_updates",
        },
    }


def load_unconditional_checkpoint(
    checkpoint: dict[str, Any], *, config: UnconditionalMainConfig, model: TinyUNet,
    ema: UnconditionalEMA, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    sampler_generator: torch.Generator, corruption_generator: torch.Generator,
) -> TrainingProgress:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("run_id") != config.run_id:
        raise ValueError("checkpoint is not compatible with the D17 run")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint frozen configuration does not match requested run")
    model.load_state_dict(checkpoint["tiny_unet_state_dict"])
    ema.load_state_dict(checkpoint["ema_tiny_unet_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    stored = checkpoint["progress"]
    sampler_generator.set_state(stored["sampler_generator_state"].cpu())
    corruption_generator.set_state(stored["corruption_generator_state"].cpu())
    _restore_rng_state(checkpoint["rng_state"])
    return TrainingProgress(
        stored["step"], stored["images_seen"], stored["batch_offset"],
        stored["permutation"].cpu() if stored["permutation"] is not None else None,
        stored["train_losses"], stored.get("skipped_step_count", 0),
    )


def canonical_checkpoint_ready(progress: TrainingProgress) -> bool:
    """D18 admits only EMA from the exact successful-update endpoint."""
    return progress.step == 20_000


def load_canonical_train_images() -> torch.Tensor:
    records = _load_text_records(TRAIN_MANIFEST, "TRAIN")
    if len(records) != 8_388:
        raise ValueError("D17 requires exactly 8388 TRAIN examples")
    return preload_images(records)


def _resume_continuity_audit(
    *, checkpoint: dict[str, Any], config: UnconditionalMainConfig, progress: TrainingProgress,
    model: TinyUNet, ema: UnconditionalEMA, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, sampler_generator: torch.Generator,
    corruption_generator: torch.Generator,
) -> dict[str, bool]:
    """Confirm every serialized continuation input immediately after restore.

    This is observational preflight instrumentation: it neither mutates the
    restored state nor changes the training path.
    """
    stored = checkpoint["progress"]

    def cpu_nested(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_nested(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cpu_nested(item) for item in value)
        if isinstance(value, list):
            return [cpu_nested(item) for item in value]
        return value

    expected_provenance = {
        "git_commit": _git_head(),
        "train_manifest": TRAIN_MANIFEST.relative_to(ROOT).as_posix(),
        "split_role": "TRAIN",
        "canonical_checkpoint_policy": "EMA_at_exactly_20000_successful_updates",
    }
    return {
        "raw_model": _nested_exact(model.state_dict(), checkpoint["tiny_unet_state_dict"]),
        "ema": _nested_exact(ema.state_dict(), checkpoint["ema_tiny_unet_state_dict"]),
        "optimizer": _nested_exact(optimizer.state_dict(), checkpoint["optimizer_state_dict"]),
        "grad_scaler": _nested_exact(scaler.state_dict(), checkpoint["grad_scaler_state_dict"]),
        "successful_step_counter": progress.step == stored["step"],
        "image_presentation_counter": progress.images_seen == stored["images_seen"],
        "skip_accounting": progress.skipped_step_count == stored.get("skipped_step_count", 0),
        "sampler_rng": torch.equal(sampler_generator.get_state(), stored["sampler_generator_state"].cpu()),
        "corruption_rng": torch.equal(corruption_generator.get_state(), stored["corruption_generator_state"].cpu()),
        "process_rng": _nested_exact(_rng_state(), cpu_nested(checkpoint["rng_state"])),
        "data_permutation": _nested_exact(progress.permutation, stored["permutation"].cpu() if stored["permutation"] is not None else None),
        "data_offset": progress.batch_offset == stored["batch_offset"],
        "frozen_config": checkpoint["config"] == asdict(config),
        "provenance": checkpoint.get("provenance") == expected_provenance,
    }


def run_unconditional_main_training(
    *, config: UnconditionalMainConfig, output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_successful_steps: int, resume_from: Path | None = None, audit_resume: bool = False,
) -> dict[str, Any]:
    """Run a caller-bounded prefix; the canonical result exists only at 20k successes."""
    if not 1 <= max_successful_steps <= config.successful_steps:
        raise ValueError("max_successful_steps must be within the frozen 20k budget")
    if not torch.cuda.is_available():
        raise RuntimeError("D17 MAIN training requires CUDA AMP FP16")
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    images = load_canonical_train_images()
    model = TinyUNet(dropout=0.0).to(device)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != EXPECTED_PARAMETERS:
        raise ValueError("TinyUNet parameter count differs from frozen D17 architecture")
    schedule = LinearDDPMSchedule(config.diffusion_timesteps, config.beta_start, config.beta_end).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = UnconditionalEMA(model)
    sampler_generator = torch.Generator().manual_seed(config.seed)
    corruption_generator = torch.Generator(device=device).manual_seed(config.seed)
    progress = TrainingProgress()
    initial_step = 0
    initial_skips = 0
    resume_continuity_audit: dict[str, bool] | None = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        progress = load_unconditional_checkpoint(checkpoint, config=config, model=model, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator, corruption_generator=corruption_generator)
        if audit_resume:
            resume_continuity_audit = _resume_continuity_audit(
                checkpoint=checkpoint, config=config, progress=progress, model=model, ema=ema,
                optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator,
                corruption_generator=corruption_generator,
            )
            if not all(resume_continuity_audit.values()):
                raise RuntimeError(f"resume continuity audit failed: {resume_continuity_audit}")
    initial_step, initial_skips = progress.step, progress.skipped_step_count
    run_dir = output_root / config.run_id
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    successful_step_seconds: list[float] = []
    losses: list[float] = []
    scaler_start = float(scaler.get_scale())
    wall_started = time.perf_counter()
    while progress.step < max_successful_steps:
        if progress.permutation is None or progress.batch_offset + config.physical_batch_size > images.shape[0]:
            progress.permutation = torch.randperm(images.shape[0], generator=sampler_generator)
            progress.batch_offset = 0
        indices = progress.permutation[progress.batch_offset:progress.batch_offset + config.physical_batch_size]
        progress.batch_offset += config.physical_batch_size
        batch = images[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = unconditional_diffusion_loss(model, schedule, batch, generator=corruption_generator)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite unconditional DDPM loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        if finalize_unconditional_amp_attempt(scaler=scaler, optimizer=optimizer, gradients_finite=finite, progress=progress, ema=ema, model=model, ema_decay=config.ema_decay):
            continue
        torch.cuda.synchronize(device)
        successful_step_seconds.append(time.perf_counter() - step_started)
        progress.images_seen += config.physical_batch_size
        progress.train_losses.append(float(loss.detach()))
        losses.append(float(loss.detach()))
        checkpoint = build_unconditional_checkpoint(config=config, progress=progress, model=model, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator, corruption_generator=corruption_generator)
        atomic_save_checkpoint(run_dir / "latest.pt", checkpoint)
        if progress.step in config.milestones:
            atomic_save_checkpoint(run_dir / f"milestone_{progress.step:05d}.pt", checkpoint)
    return {
        "run_id": config.run_id, "successful_steps": progress.step,
        "canonical_checkpoint_ready": canonical_checkpoint_ready(progress),
        "latest_checkpoint": run_dir / "latest.pt", "segment_successful_steps": progress.step - initial_step,
        "segment_attempted_updates": (progress.step + progress.skipped_step_count) - (initial_step + initial_skips),
        "segment_skipped_updates": progress.skipped_step_count - initial_skips,
        "images_seen": progress.images_seen, "scaler_start": scaler_start,
        "scaler_end": float(scaler.get_scale()), "losses": losses,
        "successful_step_seconds": successful_step_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "resume_continuity_audit": resume_continuity_audit,
    }
