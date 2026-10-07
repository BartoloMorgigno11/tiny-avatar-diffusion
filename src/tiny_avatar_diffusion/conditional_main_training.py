"""Resumable DR03 main conditional-training runner (the 20k budget is never auto-started)."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.conditional_lr_pilot import (
    COMP_VAL_MANIFEST,
    ID_VAL_MANIFEST,
    PHYSICAL_BATCH_SIZE,
    SEED,
    FixedValidationSplit,
    TextImageSplit,
    _batch_from_indices,
    _finite_gradients,
    _load_text_records,
    build_fixed_validation_split,
    evaluate_fixed_validation,
    load_text_image_split,
)
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.conditional_training import conditional_diffusion_loss
from tiny_avatar_diffusion.debug_tiny_overfit import preload_images
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "dr03_main_training"
TRAIN_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl"
CHECKPOINT_SCHEMA = "dr03_conditional_main_training_v1"
CA16_CHECKPOINT_SCHEMA = "ca16_ablation_main_training_v1"
VALIDATION_INTERVAL = 1_000
MILESTONES = (5_000, 10_000, 15_000, 20_000)
ATOMIC_REPLACE_ATTEMPTS = 6
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05


@dataclass(frozen=True)
class MainTrainingConfig:
    """Frozen scientific controls recorded in every resumable checkpoint."""

    run_id: str
    dataset_variant: str = "10k"
    resolution: tuple[int, int] = (32, 32)
    channels: int = 3
    physical_batch_size: int = PHYSICAL_BATCH_SIZE
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    seed: int = SEED
    diffusion_timesteps: int = 1_000
    schedule: str = "linear"
    precision: str = "AMP_FP16"
    validation_interval: int = VALIDATION_INTERVAL
    milestones: tuple[int, ...] = MILESTONES
    budget_steps: int = 20_000
    ema_decay: float = 0.9999

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.dataset_variant != "10k" or self.resolution != (32, 32) or self.channels != 3:
            raise ValueError("DR03 main runner is frozen to 10k RGB 32x32 artifacts")
        if self.physical_batch_size != 32 or self.gradient_accumulation != 1:
            raise ValueError("batch size 32 and accumulation 1 are frozen")
        if self.learning_rate != 2e-4 or self.weight_decay != 0.0:
            raise ValueError("AdamW LR=2e-4 and weight_decay=0 are frozen")
        if self.diffusion_timesteps != 1_000 or self.schedule != "linear":
            raise ValueError("the DR02 linear T=1000 schedule is frozen")
        if self.precision != "AMP_FP16" or self.validation_interval != 1_000:
            raise ValueError("native AMP FP16 and validation every 1000 steps are frozen")
        if self.milestones != MILESTONES or self.budget_steps != 20_000 or self.ema_decay != 0.9999:
            raise ValueError("main budget, milestones, and EMA decay are frozen")


class ConditionalEMA:
    """EMA shadows for both conditional components, including state-dict buffers."""

    def __init__(self, text_encoder: torch.nn.Module, denoiser: torch.nn.Module) -> None:
        self.text_encoder = self._clone(text_encoder.state_dict())
        self.denoiser = self._clone(denoiser.state_dict())

    @staticmethod
    def _clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in state.items()}

    @staticmethod
    def _update(shadow: dict[str, torch.Tensor], raw: dict[str, torch.Tensor], decay: float) -> None:
        if shadow.keys() != raw.keys():
            raise ValueError("EMA and raw state dictionaries differ")
        with torch.no_grad():
            for name, value in raw.items():
                if value.is_floating_point():
                    shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                else:
                    shadow[name].copy_(value.detach())

    def update(self, text_encoder: torch.nn.Module, denoiser: torch.nn.Module, *, decay: float) -> None:
        self._update(self.text_encoder, text_encoder.state_dict(), decay)
        self._update(self.denoiser, denoiser.state_dict(), decay)

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"text_encoder": self._clone(self.text_encoder), "conditional_unet": self._clone(self.denoiser)}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.text_encoder = self._clone(state["text_encoder"])
        self.denoiser = self._clone(state["conditional_unet"])

    @contextmanager
    def applied_to(self, text_encoder: torch.nn.Module, denoiser: torch.nn.Module) -> Iterator[None]:
        """Temporarily apply EMA weights for a preview and restore raw weights exactly."""
        raw_text, raw_denoiser = self._clone(text_encoder.state_dict()), self._clone(denoiser.state_dict())
        text_encoder.load_state_dict(self.text_encoder)
        denoiser.load_state_dict(self.denoiser)
        try:
            yield
        finally:
            text_encoder.load_state_dict(raw_text)
            denoiser.load_state_dict(raw_denoiser)


@dataclass
class TrainingProgress:
    step: int = 0
    images_seen: int = 0
    batch_offset: int = 0
    permutation: torch.Tensor | None = None
    train_losses: list[float] | None = None
    skipped_step_count: int = 0
    attempted_batch_count: int = 0
    skipped_attempt_positions: list[int] | None = None

    def __post_init__(self) -> None:
        if self.train_losses is None:
            self.train_losses = []
        if self.skipped_attempt_positions is None:
            self.skipped_attempt_positions = []


def _is_transient_windows_replace_error(error: OSError) -> bool:
    """Return whether ``error`` is a retryable Windows sharing/access denial."""
    return os.name == "nt" and (
        isinstance(error, PermissionError)
        or getattr(error, "winerror", None) in {5, 32, 33}
        or error.errno in {errno.EACCES, errno.EPERM, errno.EBUSY}
    )


def atomic_save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Durably serialize then atomically replace ``path`` with bounded Windows retries.

    Each invocation writes to a unique sibling temporary path, so stale files and
    concurrent writers cannot collide on a shared ``.tmp`` name.  Only transient
    Windows sharing/access denials around ``os.replace`` are retried; serialization
    failures and non-Windows errors propagate immediately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                if not _is_transient_windows_replace_error(error):
                    raise
                if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                    raise PermissionError(
                        f"atomic checkpoint replacement failed after {attempt + 1} attempts: {path}"
                    ) from error
                time.sleep(ATOMIC_REPLACE_INITIAL_DELAY_SECONDS * (2 ** attempt))
    finally:
        # Never remove another writer's uniquely named temporary file.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda_all"]])


def build_checkpoint(
    *, config: MainTrainingConfig, progress: TrainingProgress, text_encoder: torch.nn.Module,
    denoiser: torch.nn.Module, ema: ConditionalEMA, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, sampler_generator: torch.Generator,
    corruption_generator: torch.Generator, validation_history: list[dict[str, Any]],
    experiment_spec: Any | None = None,
    execution_git_sha: str | None = None,
    execution_branch: str | None = None,
) -> dict[str, Any]:
    """Return every state required to continue exactly from an optimizer update."""
    if experiment_spec is not None:
        experiment_spec.validate_config(config)
        experiment_spec.verify_manifests()
        schema = CA16_CHECKPOINT_SCHEMA
        if execution_git_sha is None:
            execution_git_sha, execution_branch = _git_provenance()
        experiment_identity = experiment_spec.provenance(
            execution_git_sha=execution_git_sha,
            branch=execution_branch,
        )
    else:
        schema = CHECKPOINT_SCHEMA
        experiment_identity = None
    checkpoint = {
        "schema": schema, "run_id": config.run_id, "config": asdict(config),
        "progress": {"step": progress.step, "images_seen": progress.images_seen,
                     "batch_offset": progress.batch_offset, "permutation": progress.permutation,
                     "train_losses": progress.train_losses,
                     "skipped_step_count": progress.skipped_step_count,
                     "attempted_batch_count": progress.attempted_batch_count,
                     "skipped_attempt_positions": progress.skipped_attempt_positions,
                     "sampler_generator_state": sampler_generator.get_state(),
                     "corruption_generator_state": corruption_generator.get_state()},
        "text_encoder_state_dict": text_encoder.state_dict(),
        "conditional_unet_state_dict": denoiser.state_dict(), "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "grad_scaler_state_dict": scaler.state_dict(),
        "rng_state": _rng_state(), "validation_history": validation_history,
        "provenance": {"train_manifest": str(TRAIN_MANIFEST), "id_val_manifest": str(ID_VAL_MANIFEST),
                       "comp_val_manifest": str(COMP_VAL_MANIFEST), "validation_weight_set": "raw"},
    }
    if experiment_identity is not None:
        checkpoint["experiment"] = experiment_identity
    return checkpoint


def load_checkpoint(
    path: Path, *, config: MainTrainingConfig, text_encoder: torch.nn.Module, denoiser: torch.nn.Module,
    ema: ConditionalEMA, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    sampler_generator: torch.Generator, corruption_generator: torch.Generator,
    map_location: torch.device, experiment_spec: Any | None = None,
) -> tuple[TrainingProgress, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    expected_schema = CA16_CHECKPOINT_SCHEMA if experiment_spec is not None else CHECKPOINT_SCHEMA
    if checkpoint.get("schema") != expected_schema or checkpoint.get("run_id") != config.run_id:
        raise ValueError("checkpoint is not compatible with this DR03 main run ID")
    if experiment_spec is not None:
        experiment_spec.validate_config(config)
        experiment_spec.verify_manifests()
        identity = checkpoint.get("experiment")
        expected = experiment_spec.provenance(
            execution_git_sha=identity.get("git_commit", "") if isinstance(identity, dict) else "",
            branch=identity.get("branch") if isinstance(identity, dict) else None,
        )
        if not isinstance(identity, dict) or identity != expected:
            raise ValueError("checkpoint CA16 experiment identity does not match requested run")
    elif checkpoint.get("schema") == CA16_CHECKPOINT_SCHEMA:
        raise ValueError("CA16 checkpoint requires an explicit experiment spec")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint frozen configuration does not match requested run")
    required = {"progress", "ema_state_dict", "optimizer_state_dict", "grad_scaler_state_dict", "rng_state"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint lacks resumable DR03 main-training state")
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    denoiser.load_state_dict(checkpoint["conditional_unet_state_dict"])
    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    stored = checkpoint["progress"]
    # Generator states are serialized CPU byte tensors even when model tensors are
    # restored directly to CUDA.
    sampler_generator.set_state(stored["sampler_generator_state"].cpu())
    corruption_generator.set_state(stored["corruption_generator_state"].cpu())
    _restore_rng_state(checkpoint["rng_state"])
    permutation = stored["permutation"]
    if permutation is not None:
        permutation = permutation.cpu()
    skipped_step_count = stored.get("skipped_step_count", 0)
    # Oldest checkpoints cannot recover attempts without skip history; step is
    # only a safe compatibility fallback, not reconstructed historical truth.
    attempted_batch_count = stored.get("attempted_batch_count", stored["step"] + skipped_step_count)
    skipped_attempt_positions = stored.get("skipped_attempt_positions", [])
    return TrainingProgress(
        stored["step"],
        stored["images_seen"],
        stored["batch_offset"],
        permutation,
        stored["train_losses"],
        skipped_step_count,
        attempted_batch_count,
        skipped_attempt_positions,
    ), checkpoint.get("validation_history", [])


def _git_provenance() -> tuple[str, str | None]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip() or None
    if not sha:
        raise RuntimeError("unable to resolve execution Git SHA")
    return sha, branch


def _reset_runtime_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_models(
    device: torch.device, seed: int, *, use_decoder16_cross_attention: bool = False,
) -> tuple[TextConditioningEncoder, ConditionalTinyUNet]:
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    models = (
        TextConditioningEncoder().to(device),
        ConditionalTinyUNet(
            dropout=0.0,
            use_decoder16_cross_attention=use_decoder16_cross_attention,
        ).to(device),
    )
    return models


def _new_models_for_spec(
    device: torch.device, spec: Any
) -> tuple[TextConditioningEncoder, ConditionalTinyUNet]:
    config = spec.main_training_config()
    random.seed(config.seed)
    models = _new_models(
        device,
        config.seed,
        use_decoder16_cross_attention=spec.use_decoder16_cross_attention,
    )
    _reset_runtime_rng(config.seed)
    return models


def audit_paired_initialization(seed: int, *, device: torch.device | None = None) -> dict[str, Any]:
    """Audit paired CA16 construction and initial runtime streams without training."""
    from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec

    caller_python = random.getstate()
    caller_cpu = torch.get_rng_state().clone()
    caller_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        target = torch.device("cpu") if device is None else device
        control_spec = CA16RunSpec(ArchitectureArm.CA8, seed)
        treatment_spec = CA16RunSpec(ArchitectureArm.CA8_CA16, seed)
        control_text, control_denoiser = _new_models_for_spec(target, control_spec)
        control_python = random.getstate()
        control_cpu = torch.get_rng_state()
        control_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        control_sampler = torch.Generator().manual_seed(seed)
        control_corruption = torch.Generator(device=target).manual_seed(seed)
        treatment_text, treatment_denoiser = _new_models_for_spec(target, treatment_spec)
        treatment_python = random.getstate()
        treatment_cpu = torch.get_rng_state()
        treatment_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        treatment_sampler = torch.Generator().manual_seed(seed)
        treatment_corruption = torch.Generator(device=target).manual_seed(seed)
        shared_equal = all(
            torch.equal(control_text.state_dict()[name], treatment_text.state_dict()[name])
            for name in control_text.state_dict()
        ) and all(
            torch.equal(control_denoiser.state_dict()[name], treatment_denoiser.state_dict()[name])
            for name in control_denoiser.state_dict()
        )
        control_keys = set(control_denoiser.state_dict())
        treatment_keys = set(treatment_denoiser.state_dict())
        treatment_only = treatment_keys - control_keys
        expected_only = {
            f"decoder16_image_text_cross_attention.{name}"
            for name in treatment_denoiser.decoder16_image_text_cross_attention.state_dict()
        }
        protocol_equal = control_spec.frozen_protocol == treatment_spec.frozen_protocol
        manifests_equal = control_spec.manifests == treatment_spec.manifests
        return {
            "shared_parameters_equal": shared_equal,
            "control_has_no_ca16_namespace": not any(
                key.startswith("decoder16_image_text_cross_attention.") for key in control_keys
            ),
            "treatment_only_namespace_exact": treatment_only == expected_only,
            "python_rng_equal": control_python == treatment_python,
            "torch_cpu_rng_equal": torch.equal(control_cpu, treatment_cpu),
            "torch_cuda_rng_equal": _nested_exact(control_cuda, treatment_cuda),
            "sampler_generator_equal": torch.equal(control_sampler.get_state(), treatment_sampler.get_state()),
            "corruption_generator_equal": torch.equal(
                control_corruption.get_state(), treatment_corruption.get_state()
            ),
            "frozen_protocol_equal": protocol_equal,
            "manifest_references_equal": manifests_equal,
        }
    finally:
        random.setstate(caller_python)
        torch.set_rng_state(caller_cpu)
        if caller_cuda is not None:
            torch.cuda.set_rng_state_all(caller_cuda)


def _real_subset(path: Path, split: str, count: int) -> TextImageSplit:
    """Deterministic artifact-order subset used only by the short integrity gate."""
    records = _load_text_records(path, split)[:count]
    return TextImageSplit(split, preload_images(records), torch.tensor([r["token_ids"] for r in records]), torch.tensor([r["attention_mask"] for r in records]))


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def finalize_amp_attempt(*, scaler: Any, optimizer: torch.optim.Optimizer, gradients_finite: bool, progress: TrainingProgress, ema: ConditionalEMA, text_encoder: torch.nn.Module, denoiser: torch.nn.Module, ema_decay: float) -> bool:
    """Finalize one already-unscaled AMP attempt; returns whether it was skipped."""
    before = float(scaler.get_scale())
    progress.attempted_batch_count += 1
    scaler.step(optimizer); scaler.update()
    skipped = float(scaler.get_scale()) < before
    if not gradients_finite and not skipped:
        raise FloatingPointError("non-finite gradients were not skipped by GradScaler")
    if skipped:
        progress.skipped_step_count += 1
        progress.skipped_attempt_positions.append(progress.attempted_batch_count)
        return True
    ema.update(text_encoder, denoiser, decay=ema_decay)
    progress.step += 1
    return False


def tensor_audit(tensor: torch.Tensor) -> dict[str, Any]:
    """Compact, device-independent record of an observed training tensor."""
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def compare_continuation_audits(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    fields = ("indices", "images", "input_ids", "attention_mask", "timesteps", "noise")
    return {field: left[field] == right[field] for field in fields}


def audit_next_update(checkpoint: dict[str, Any], train: TextImageSplit) -> dict[str, Any]:
    """Observe the exact next loader/corruption values without changing their generation."""
    progress = checkpoint["progress"]
    permutation, offset = progress["permutation"], progress["batch_offset"]
    if offset + 32 > train.images.shape[0]:
        sampler = torch.Generator().set_state(progress["sampler_generator_state"].cpu())
        permutation, offset = torch.randperm(train.images.shape[0], generator=sampler), 0
    indices = permutation[offset:offset + 32].cpu()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    images, input_ids, masks = _batch_from_indices(train, indices, device)
    generator = torch.Generator(device=device).set_state(progress["corruption_generator_state"].cpu())
    timesteps = torch.randint(0, 1_000, (len(indices),), device=device, generator=generator)
    noise = torch.randn(images.shape, device=device, dtype=images.dtype, generator=generator)
    return {"indices": indices.tolist(), "images": tensor_audit(images), "input_ids": input_ids.tolist(),
            "attention_mask": masks.tolist(), "timesteps": timesteps.tolist(), "noise": tensor_audit(noise)}


def fixed_preview_manifest() -> list[dict[str, Any]]:
    """Predeclared artifact-order panel: two ID_VAL and two COMP_VAL prompts."""
    entries: list[dict[str, Any]] = []
    for offset, (path, split) in enumerate(((ID_VAL_MANIFEST, "ID_VAL"), (COMP_VAL_MANIFEST, "COMP_VAL"))):
        for index, record in enumerate(_load_text_records(path, split)[:2]):
            entries.append({"split": split, "image_id": record["image_id"], "caption": record["caption"], "token_ids": record["token_ids"], "attention_mask": record["attention_mask"], "generation_seed": SEED + offset * 2 + index})
    return entries


def generate_ema_milestone_preview(*, run_dir: Path, step: int, run_id: str, text_encoder: TextConditioningEncoder, denoiser: ConditionalTinyUNet, ema: ConditionalEMA, schedule: LinearDDPMSchedule) -> Path:
    """Generate the fixed diagnostic panel using the complete EMA pair only."""
    manifest = fixed_preview_manifest()
    panel_dir = run_dir / "previews" / f"step_{step:06d}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    device = next(denoiser.parameters()).device
    with ema.applied_to(text_encoder, denoiser):
        for index, item in enumerate(manifest):
            generator = torch.Generator(device=device).manual_seed(item["generation_seed"])
            image = conditional_sample(schedule, text_encoder, denoiser, torch.tensor([item["token_ids"]], device=device), torch.tensor([item["attention_mask"]], device=device), generator=generator, clip_denoised=True)[0]
            torch.save(image.cpu(), panel_dir / f"{index:02d}_{item['split']}_{item['image_id']}.pt")
    payload = {"run_id": run_id, "milestone_step": step, "ema": True, "ema_decay": 0.9999, "clip_denoised": True, "diagnostic_only": True, "entries": manifest}
    (panel_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return panel_dir


def run_main_training(
    *, config: MainTrainingConfig, output_root: Path = DEFAULT_OUTPUT_ROOT, device_index: int = 0,
    max_steps: int | None = None, resume_from: Path | None = None,
    train: TextImageSplit | None = None, id_validation: FixedValidationSplit | None = None,
    comp_validation: FixedValidationSplit | None = None, experiment_spec: Any | None = None,
) -> dict[str, Any]:
    """Run up to ``max_steps`` successful updates; callers must explicitly request the full budget."""
    if experiment_spec is not None:
        experiment_spec.validate_config(config)
        experiment_spec.verify_manifests()
    if not torch.cuda.is_available(): raise RuntimeError("DR03 main training requires CUDA native AMP")
    if max_steps is None: max_steps = config.budget_steps
    if not 0 < max_steps <= config.budget_steps: raise ValueError("max_steps must be in 1..20000")
    device = torch.device(f"cuda:{device_index}")
    if train is None: train = load_text_image_split(TRAIN_MANIFEST, "TRAIN")
    if id_validation is None: id_validation = build_fixed_validation_split(load_text_image_split(ID_VAL_MANIFEST, "ID_VAL"), seed=config.seed + 1)
    if comp_validation is None: comp_validation = build_fixed_validation_split(load_text_image_split(COMP_VAL_MANIFEST, "COMP_VAL"), seed=config.seed + 2)
    if experiment_spec is None:
        text_encoder, denoiser = _new_models(
            device, config.seed, use_decoder16_cross_attention=False
        )
        execution_git_sha = execution_branch = None
    else:
        text_encoder, denoiser = _new_models_for_spec(device, experiment_spec)
        execution_git_sha, execution_branch = _git_provenance()
    schedule = LinearDDPMSchedule().to(device)
    optimizer = torch.optim.AdamW([*text_encoder.parameters(), *denoiser.parameters()], lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = ConditionalEMA(text_encoder, denoiser)
    sampler_generator = torch.Generator().manual_seed(config.seed)
    corruption_generator = torch.Generator(device=device).manual_seed(config.seed)
    run_dir = output_root / config.run_id
    latest = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    if resume_from is not None:
        progress, validation_history = load_checkpoint(
            resume_from,
            config=config,
            text_encoder=text_encoder,
            denoiser=denoiser,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            sampler_generator=sampler_generator,
            corruption_generator=corruption_generator,
            map_location=device,
            experiment_spec=experiment_spec,
        )
    else:
        if latest.exists(): raise FileExistsError("run directory already has a latest checkpoint; pass resume_from")
        progress, validation_history = TrainingProgress(), []
    if progress.step == 0 and not validation_history:
        for name, split in (("ID_VAL", id_validation), ("COMP_VAL", comp_validation)):
            validation_history.append({"step": 0, "split": name, "epsilon_mse": evaluate_fixed_validation(text_encoder, denoiser, schedule, split, device=device)})
        _append_metric(metrics_path, {"kind": "validation", "step": 0, "id_val_epsilon_mse": validation_history[-2]["epsilon_mse"], "comp_val_epsilon_mse": validation_history[-1]["epsilon_mse"], "weight_set": "RAW"})
    started = time.perf_counter()
    while progress.step < max_steps:
        if progress.permutation is None or progress.batch_offset + config.physical_batch_size > train.images.shape[0]:
            progress.permutation = torch.randperm(train.images.shape[0], generator=sampler_generator)
            progress.batch_offset = 0
        indices = progress.permutation[progress.batch_offset:progress.batch_offset + config.physical_batch_size]
        progress.batch_offset += config.physical_batch_size
        x_start, input_ids, attention_mask = _batch_from_indices(train, indices, device)
        text_encoder.train(); denoiser.train(); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            result = conditional_diffusion_loss(text_encoder, denoiser, schedule, x_start, input_ids, attention_mask, generator=corruption_generator)
        if not bool(torch.isfinite(result.loss)): raise FloatingPointError("non-finite training loss")
        scaler.scale(result.loss).backward(); scaler.unscale_(optimizer)
        gradients_finite = _finite_gradients(text_encoder, denoiser)
        if finalize_amp_attempt(scaler=scaler, optimizer=optimizer, gradients_finite=gradients_finite, progress=progress, ema=ema, text_encoder=text_encoder, denoiser=denoiser, ema_decay=config.ema_decay):
            # The batch was already drawn and is intentionally consumed, preserving prior sampler semantics.
            continue
        progress.images_seen += config.physical_batch_size; progress.train_losses.append(float(result.loss.detach()))
        _append_metric(metrics_path, {"kind": "train", "step": progress.step, "images_seen": progress.images_seen, "effective_epochs": progress.images_seen / train.images.shape[0], "raw_training_epsilon_mse": progress.train_losses[-1], "smoothed_training_epsilon_mse": sum(progress.train_losses[-100:]) / min(100, len(progress.train_losses)), "learning_rate": config.learning_rate, "grad_scaler_scale": float(scaler.get_scale()), "skipped_step_count": progress.skipped_step_count, "elapsed_seconds": time.perf_counter() - started})
        if progress.step % config.validation_interval == 0:
            for name, split in (("ID_VAL", id_validation), ("COMP_VAL", comp_validation)):
                validation_history.append({"step": progress.step, "split": name, "epsilon_mse": evaluate_fixed_validation(text_encoder, denoiser, schedule, split, device=device)})
            _append_metric(metrics_path, {"kind": "validation", "step": progress.step, "id_val_epsilon_mse": validation_history[-2]["epsilon_mse"], "comp_val_epsilon_mse": validation_history[-1]["epsilon_mse"], "weight_set": "RAW"})
            atomic_save_checkpoint(latest, build_checkpoint(config=config, progress=progress, text_encoder=text_encoder, denoiser=denoiser, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator, corruption_generator=corruption_generator, validation_history=validation_history, experiment_spec=experiment_spec, execution_git_sha=execution_git_sha, execution_branch=execution_branch))
            if progress.step in config.milestones:
                milestone_path = run_dir / f"milestone_{progress.step:05d}.pt"
                atomic_save_checkpoint(milestone_path, build_checkpoint(config=config, progress=progress, text_encoder=text_encoder, denoiser=denoiser, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator, corruption_generator=corruption_generator, validation_history=validation_history, experiment_spec=experiment_spec, execution_git_sha=execution_git_sha, execution_branch=execution_branch))
                preview_path = generate_ema_milestone_preview(run_dir=run_dir, step=progress.step, run_id=config.run_id, text_encoder=text_encoder, denoiser=denoiser, ema=ema, schedule=schedule)
                _append_metric(metrics_path, {"kind": "milestone", "step": progress.step, "checkpoint_path": str(milestone_path), "ema_preview_path": str(preview_path)})
    checkpoint = build_checkpoint(config=config, progress=progress, text_encoder=text_encoder, denoiser=denoiser, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler_generator, corruption_generator=corruption_generator, validation_history=validation_history, experiment_spec=experiment_spec, execution_git_sha=execution_git_sha, execution_branch=execution_branch)
    atomic_save_checkpoint(latest, checkpoint)
    summary = {"run_id": config.run_id, "step": progress.step, "images_seen": progress.images_seen, "elapsed_seconds": time.perf_counter() - started, "validation_history": validation_history, "latest_checkpoint": str(latest)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_nested_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_nested_exact(a, b) for a, b in zip(left, right))
    return left == right


def run_resume_integrity_gate(
    *, output_root: Path = ROOT / "reports" / "dr03_main_training_integrity",
    total_steps: int = 2, split_step: int = 1, device_index: int = 0, run_tag: str = "gate",
) -> dict[str, Any]:
    """Short real CUDA AMP check: continuous execution must equal split-and-resumed execution.

    This is an engineering gate, not a training result.  Its short length deliberately
    retains the production cadence (validation at step 0, then each 1000 updates).
    """
    if not 0 < split_step < total_steps < VALIDATION_INTERVAL:
        raise ValueError("integrity gate requires 0 < split_step < total_steps < 1000")
    if not torch.cuda.is_available():
        raise RuntimeError("resume integrity gate requires CUDA")
    device = torch.device(f"cuda:{device_index}")
    train = _real_subset(TRAIN_MANIFEST, "TRAIN", 32)
    id_validation = build_fixed_validation_split(_real_subset(ID_VAL_MANIFEST, "ID_VAL", 32), seed=SEED + 1)
    comp_validation = build_fixed_validation_split(_real_subset(COMP_VAL_MANIFEST, "COMP_VAL", 32), seed=SEED + 2)
    continuous = MainTrainingConfig(run_id=f"{run_tag}_continuous")
    continuous_prefix = MainTrainingConfig(run_id=f"{run_tag}_continuous_prefix")
    resumed = MainTrainingConfig(run_id=f"{run_tag}_resumed")
    run_main_training(config=continuous, output_root=output_root, max_steps=total_steps, device_index=device_index, train=train, id_validation=id_validation, comp_validation=comp_validation)
    prefix = run_main_training(config=continuous_prefix, output_root=output_root, max_steps=split_step, device_index=device_index, train=train, id_validation=id_validation, comp_validation=comp_validation)
    first = run_main_training(config=resumed, output_root=output_root, max_steps=split_step, device_index=device_index, train=train, id_validation=id_validation, comp_validation=comp_validation)
    continuous_step_two = audit_next_update(torch.load(Path(prefix["latest_checkpoint"]), map_location="cpu", weights_only=False), train)
    resumed_step_two = audit_next_update(torch.load(Path(first["latest_checkpoint"]), map_location="cpu", weights_only=False), train)
    continuation = compare_continuation_audits(continuous_step_two, resumed_step_two)
    if not all(continuation.values()): raise AssertionError(f"resume continuation audit failed: {continuation}")
    run_main_training(config=resumed, output_root=output_root, max_steps=total_steps, device_index=device_index, resume_from=Path(first["latest_checkpoint"]), train=train, id_validation=id_validation, comp_validation=comp_validation)
    continuous_state = torch.load(output_root / continuous.run_id / "latest.pt", map_location=device, weights_only=False)
    resumed_state = torch.load(output_root / resumed.run_id / "latest.pt", map_location=device, weights_only=False)
    raw_exact = _nested_exact(continuous_state["text_encoder_state_dict"], resumed_state["text_encoder_state_dict"]) and _nested_exact(continuous_state["conditional_unet_state_dict"], resumed_state["conditional_unet_state_dict"])
    ema_exact = _nested_exact(continuous_state["ema_state_dict"], resumed_state["ema_state_dict"])
    optimizer_exact = _nested_exact(continuous_state["optimizer_state_dict"], resumed_state["optimizer_state_dict"])
    progress_exact = _nested_exact(continuous_state["progress"], resumed_state["progress"])
    validation_exact = _nested_exact(continuous_state["validation_history"], resumed_state["validation_history"])
    def final_probe(state: dict[str, Any]) -> tuple[float, float]:
        text, denoiser = _new_models(device, SEED)
        text.load_state_dict(state["text_encoder_state_dict"]); denoiser.load_state_dict(state["conditional_unet_state_dict"])
        schedule = LinearDDPMSchedule().to(device)
        return (evaluate_fixed_validation(text, denoiser, schedule, id_validation, device=device), evaluate_fixed_validation(text, denoiser, schedule, comp_validation, device=device))
    continuous_probe, resumed_probe = final_probe(continuous_state), final_probe(resumed_state)
    probe_exact = continuous_probe == resumed_probe
    passed = raw_exact and ema_exact and optimizer_exact and progress_exact and validation_exact and probe_exact and all(continuation.values())
    report = {"type": "SHORT CUDA AMP RESUME INTEGRITY GATE ONLY", "total_steps": total_steps,
              "split_step": split_step, "raw_parameters_exact": raw_exact, "ema_parameters_exact": ema_exact,
              "optimizer_progress_exact": optimizer_exact and progress_exact,
              "validation_history_exact": validation_exact, "final_id_comp_probe": {"continuous": continuous_probe, "resumed": resumed_probe, "exact": probe_exact}, "passed": passed,
              "step_2_continuous_audit": continuous_step_two, "step_2_resumed_audit": resumed_step_two, "continuation_equality": continuation,
              "tolerance": "exact torch.equal for serialized state", "real_data_subset": {"TRAIN": 32, "ID_VAL": 32, "COMP_VAL": 32}}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resume_integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed: raise AssertionError("continuous and resumed CUDA AMP states diverged")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True); parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT); parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run_main_training(config=MainTrainingConfig(run_id=args.run_id), output_root=args.output_root, max_steps=args.max_steps, resume_from=args.resume_from)


if __name__ == "__main__": main()
