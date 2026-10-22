"""Fresh V2-A matched-budget (8x4) training primitives.

This deliberately does not modify the authenticated V2 trainer.  It is a new,
strictly identified experiment whose optimizer-update semantics are explicit.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.v2_native64 import make_v2_model, parameter_count

SCHEMA = "tiny_avatar_diffusion.v2a_matched_budget_training_v1"
V2_B_SCHEMA = "tiny_avatar_diffusion.v2b_matched_budget_training_v1"
RUN_ID = "v2_a_native64_matched_batch32_seed20261017"
V2_B_RUN_ID = "v2_b_native64_matched_batch32_seed20261017"
TARGET_SUCCESSFUL_UPDATES = 40_000
MICROBATCH_SIZE, GRADIENT_ACCUMULATION = 8, 4
EFFECTIVE_BATCH = MICROBATCH_SIZE * GRADIENT_ACCUMULATION
TRAIN_COUNT = 8_388
AUTHENTIC_CACHE_SHA256 = "7776ca0bc4c14b3789f9f0a6491fefae1a30c8801637b65eaafeebdd9e4a25fd"
AUTHENTIC_MANIFEST_SHA256 = "295f7c93766294243871ebb2bf742a9fdae2f5a716ac773b0ead16dc8e6eb014"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class MatchedBudgetConfig:
    variant: str = "v2_a"
    run_id: str = RUN_ID
    successful_updates: int = TARGET_SUCCESSFUL_UPDATES
    microbatch_size: int = MICROBATCH_SIZE
    gradient_accumulation: int = GRADIENT_ACCUMULATION
    seed: int = 20261017
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    resolution: int = 64
    checkpoint_interval_updates: int = 5_000
    max_grad_norm: float | None = None  # Authentic V2 uses no clipping.

    def __post_init__(self) -> None:
        expected_run_id = {"v2_a": RUN_ID, "v2_b": V2_B_RUN_ID}.get(self.variant)
        if expected_run_id is None or self.run_id != expected_run_id:
            raise ValueError("matched-budget contract requires the canonical V2-A or V2-B run identity")
        if (self.successful_updates, self.microbatch_size, self.gradient_accumulation, self.resolution) != (40_000, 8, 4, 64):
            raise ValueError("matched-budget contract is fixed to 40k successful updates, 8x4, RGB64")
        if self.lr != 2e-4 or self.weight_decay != 0.0 or self.ema_decay != 0.9999 or self.max_grad_norm is not None:
            raise ValueError("matched-budget contract preserves authentic V2 optimizer/EMA and no clipping")

    @property
    def effective_batch_size(self) -> int: return self.microbatch_size * self.gradient_accumulation


def config_for_variant(variant: str) -> MatchedBudgetConfig:
    """Return the immutable matched-budget contract for one approved V2 variant."""
    if variant == "v2_a":
        return MatchedBudgetConfig()
    if variant == "v2_b":
        return MatchedBudgetConfig(variant="v2_b", run_id=V2_B_RUN_ID)
    raise ValueError("variant must be v2_a or v2_b")


def schema_for_config(config: MatchedBudgetConfig) -> str:
    return SCHEMA if config.variant == "v2_a" else V2_B_SCHEMA


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


class Scaler(Protocol):
    def scale(self, loss: torch.Tensor): ...
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...
    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...
    def update(self) -> None: ...
    def get_scale(self) -> float: ...
    def state_dict(self) -> dict[str, Any]: ...


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def update_ema(ema: dict[str, dict[str, torch.Tensor]], model: torch.nn.Module, text: torch.nn.Module, decay: float) -> None:
    for group, current in (("model", model.state_dict()), ("text", text.state_dict())):
        for key, value in current.items(): ema[group][key].mul_(decay).add_(value.detach().cpu(), alpha=1.0 - decay)


def optimizer_window(losses: list[torch.Tensor], *, optimizer: torch.optim.Optimizer, scaler: Scaler, model: torch.nn.Module, text: torch.nn.Module, ema: dict[str, dict[str, torch.Tensor]], progress: Progress, config: MatchedBudgetConfig) -> tuple[bool, float]:
    """Backpropagate one complete window and update EMA only if AMP stepped."""
    if len(losses) != config.gradient_accumulation:
        raise ValueError("checkpointing/stepping is permitted only at complete accumulation windows")
    optimizer.zero_grad(set_to_none=True)
    for loss in losses:
        scaler.scale(loss / config.gradient_accumulation).backward()
    scaler.unscale_(optimizer)
    # Explicitly preserved authentic behavior: no gradient clipping.
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    skipped = float(scaler.get_scale()) < before
    progress.attempted_optimizer_updates += 1
    mean_loss = float(torch.stack([loss.detach() for loss in losses]).mean().cpu())
    if skipped:
        progress.amp_skipped_updates += 1
        optimizer.zero_grad(set_to_none=True)
        return False, mean_loss
    progress.successful_updates += 1
    update_ema(ema, model, text, config.ema_decay)
    optimizer.zero_grad(set_to_none=True)
    return True, mean_loss


def preflight(data_root: Path, cache_dir: Path, config: MatchedBudgetConfig | None = None) -> dict[str, Any]:
    config = config or MatchedBudgetConfig()
    manifest = data_root / "data/processed/dr01_v1/text/10k/train.jsonl"
    cache, integrity = cache_dir / "train64_rgb_uint8.npy", cache_dir / "integrity_report.json"
    if not manifest.is_file() or not cache.is_file() or not integrity.is_file(): raise FileNotFoundError("matched-budget preflight requires authentic V2 TRAIN manifest/cache/integrity report")
    report = json.loads(integrity.read_text(encoding="utf8"))
    if sha256_file(manifest) != AUTHENTIC_MANIFEST_SHA256 or sha256_file(cache) != AUTHENTIC_CACHE_SHA256: raise ValueError("authentic V2 data provenance hash mismatch")
    if report.get("shape") != [8388,3,64,64] or report.get("cache_sha256") != AUTHENTIC_CACHE_SHA256 or not report.get("pixel_exact_independent_verification",{}).get("passed"): raise ValueError("native64 cache integrity contract failed")
    records = _load_text_records(manifest, "TRAIN")
    if len(records) != TRAIN_COUNT: raise ValueError("TRAIN count mismatch")
    return {"schema": schema_for_config(config), "config": asdict(config), "effective_batch": EFFECTIVE_BATCH, "nominal_images": EFFECTIVE_BATCH * TARGET_SUCCESSFUL_UPDATES, "manifest_sha256": sha256_file(manifest), "cache_sha256": sha256_file(cache), "cache_integrity_sha256": sha256_file(integrity), "parameter_count": parameter_count(make_v2_model(config.variant)[1], TextConditioningEncoder())}


def checkpoint_payload(config: MatchedBudgetConfig, progress: Progress, model: torch.nn.Module, text: torch.nn.Module, ema: dict[str, dict[str, torch.Tensor]], optimizer: torch.optim.Optimizer, scaler: Scaler, sampler: torch.Generator, noise: torch.Generator, provenance: dict[str, Any]) -> dict[str, Any]:
    if progress.microbatch_attempts % config.gradient_accumulation: raise ValueError("cannot checkpoint inside accumulation window")
    return {"schema":schema_for_config(config),"config":asdict(config),"provenance":provenance,"progress":asdict(progress),"model":clone_state(model.state_dict()),"text":clone_state(text.state_dict()),"ema":ema,"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"sampler_state":sampler.get_state(),"noise_generator_state":noise.get_state(),"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("wb") as f: torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def checkpoint_integrity(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


TELEMETRY_COUNTERS = (
    "successful_updates",
    "attempted_optimizer_updates",
    "amp_skipped_updates",
    "microbatch_attempts",
    "images_presented",
)


def telemetry_counters(progress: Progress) -> dict[str, int]:
    """Return only cumulative counters used for segment-accounting telemetry."""
    return {name: int(getattr(progress, name)) for name in TELEMETRY_COUNTERS}


def telemetry_payload(progress: Progress, wall_seconds: float, cuda: Any = torch.cuda, *, baseline: dict[str, int] | None = None) -> dict[str, Any]:
    """Serialize terminal telemetry with cumulative and current-process counters.

    ``baseline`` is captured after a checkpoint restore and before the measured
    loop.  It makes resumed-session rates describe the current process segment,
    not the entire accumulated run.
    """
    cumulative = telemetry_counters(progress)
    initial = {name: 0 for name in TELEMETRY_COUNTERS} if baseline is None else {
        name: int(baseline[name]) for name in TELEMETRY_COUNTERS
    }
    segment = {name: cumulative[name] - initial[name] for name in TELEMETRY_COUNTERS}
    if any(value < 0 for value in segment.values()):
        raise ValueError("telemetry baseline cannot exceed cumulative counters")
    rate = lambda value: None if wall_seconds <= 0 else value / wall_seconds
    result = {
        "wall_seconds": wall_seconds,
        # Retain these established top-level fields as cumulative counters.
        **cumulative,
        "cumulative": cumulative,
        "segment": segment,
        "successful_updates_per_second": rate(segment["successful_updates"]),
        "microbatches_per_second": rate(segment["microbatch_attempts"]),
        "images_per_second": rate(segment["images_presented"]),
        "cuda": {"available": bool(cuda.is_available())},
    }
    if not cuda.is_available(): return result
    device = cuda.current_device(); props = cuda.get_device_properties(device); allocated = int(cuda.max_memory_allocated(device)); reserved = int(cuda.max_memory_reserved(device))
    result["cuda"].update({"device_index": device, "device_name": props.name, "total_vram_bytes": int(props.total_memory), "peak_allocated_bytes": allocated, "peak_allocated_mib": allocated / 2**20, "peak_reserved_bytes": reserved, "peak_reserved_mib": reserved / 2**20})
    return result
