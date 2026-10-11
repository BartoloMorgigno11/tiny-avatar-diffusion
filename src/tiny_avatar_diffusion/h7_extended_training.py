"""Explicit, fail-closed H7 continuation of the frozen CA8+CA16 R10 runs.

This module deliberately does nothing at import time.  ``--preflight`` is
read-only; actual CUDA training additionally requires ``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.ca16_experiment import (
    FROZEN_MANIFESTS,
    ROOT,
    SUPPORTED_SEEDS,
    ArchitectureArm,
    CA16RunSpec,
)
from tiny_avatar_diffusion.conditional_lr_pilot import (
    _batch_from_indices,
    _finite_gradients,
    load_text_image_split,
)
from tiny_avatar_diffusion.conditional_main_training import (
    CA16_CHECKPOINT_SCHEMA,
    ConditionalEMA,
    TrainingProgress,
    _new_models_for_spec,
    _restore_rng_state,
    atomic_save_checkpoint,
    finalize_amp_attempt,
)
from tiny_avatar_diffusion.conditional_training import conditional_diffusion_loss
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule

EXPERIMENT_ID = "h7_ca16_40k_v1"
SCHEMA = "tiny_avatar_diffusion.h7_extended_training_v1"
OUTPUT_ROOT = ROOT / "results" / EXPERIMENT_ID
SOURCE_STEP, TARGET_STEP = 20_000, 40_000
EXPECTED_SKIPS = {2026: 4, 2027: 5, 2028: 4}
EXPECTED_SOURCE_SHA256 = {
    2026: "ad3093446be3958d322d7af9d1109636e50e0fa7decc766177f015d04b51848d",
    2027: "6749055ce11c48eaec4605ed68d77f637ee9cb0a323ca79c7273cf2715ff64d7",
    2028: "904c45d6f601e73b12bb99031806845cc4411ba89a5f20c6b962a362e691b094",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class H7ContinuationSpec:
    """The only authorized continuation identity; it is not an R10 config."""

    training_seed: int
    target_successful_steps: int = TARGET_STEP

    def __post_init__(self) -> None:
        if self.training_seed not in SUPPORTED_SEEDS:
            raise ValueError("H7 accepts only seeds 2026, 2027, and 2028")
        if self.target_successful_steps != TARGET_STEP:
            raise ValueError("H7 terminal target is frozen to 40000 successful steps")

    @property
    def source_spec(self) -> CA16RunSpec:
        return CA16RunSpec(ArchitectureArm.CA8_CA16, self.training_seed)

    def source_path(self, assets: CA16AssetSource) -> Path:
        return (
            assets.root
            / "results"
            / "runs"
            / self.source_spec.run_id
            / "milestone_20000.pt"
        )

    @property
    def run_id(self) -> str:
        return f"{EXPERIMENT_ID}_seed{self.training_seed}"

    def output_dir(self, root: Path = OUTPUT_ROOT) -> Path:
        root = root.resolve()
        if root != OUTPUT_ROOT.resolve():
            raise ValueError("H7 output root is fixed to this H7 worktree")
        result = (root / self.run_id).resolve()
        if root not in result.parents:
            raise ValueError("H7 output escapes its dedicated output root")
        return result


@dataclass(frozen=True)
class CA16AssetSource:
    """Explicit read-only location of the frozen CA16 local-only assets."""

    root: Path

    def __post_init__(self) -> None:
        root = self.root.resolve()
        object.__setattr__(self, "root", root)
        if root == ROOT.resolve() or not (root / ".git").exists():
            raise ValueError(
                "CA16 asset root must be an explicit, separate Git worktree"
            )
        if not (root / "results" / "runs").is_dir():
            raise FileNotFoundError("CA16 asset root lacks historical results/runs")


def _required_source_keys() -> set[str]:
    return {
        "config",
        "progress",
        "text_encoder_state_dict",
        "conditional_unet_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
        "experiment",
    }


def verify_source_checkpoint(
    spec: H7ContinuationSpec, assets: CA16AssetSource, path: Path | None = None
) -> dict[str, Any]:
    """Hash and structurally verify one immutable R10 source checkpoint."""
    path = spec.source_path(assets) if path is None else path.resolve()
    if assets.root not in path.parents:
        raise ValueError("H7 source checkpoint escapes the configured CA16 asset root")
    if not path.is_file():
        raise FileNotFoundError(f"H7 R10 source checkpoint is absent: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_SOURCE_SHA256[spec.training_seed]:
        raise ValueError(
            "H7 R10 source checkpoint SHA256 differs from frozen R11 inventory"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not _required_source_keys().issubset(
        checkpoint
    ):
        raise ValueError("H7 source lacks required full continuation state")
    source = spec.source_spec
    if (
        checkpoint.get("schema") != CA16_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != source.run_id
    ):
        raise ValueError("H7 source is not the required CA8+CA16 R10 run")
    if checkpoint.get("config") != asdict(source.main_training_config()):
        raise ValueError("H7 source frozen training configuration differs from R10")
    identity = checkpoint["experiment"]
    if (
        not isinstance(identity, dict)
        or identity.get("architecture_arm") != ArchitectureArm.CA8_CA16.value
        or identity.get("training_seed") != spec.training_seed
    ):
        raise ValueError("H7 source experiment identity is incompatible")
    progress = checkpoint["progress"]
    if (
        progress.get("step") != SOURCE_STEP
        or progress.get("skipped_step_count") != EXPECTED_SKIPS[spec.training_seed]
    ):
        raise ValueError(
            "H7 source successful-step or historical AMP-skip identity differs"
        )
    required_progress = {
        "permutation",
        "batch_offset",
        "sampler_generator_state",
        "corruption_generator_state",
        "attempted_batch_count",
        "skipped_attempt_positions",
    }
    if not required_progress.issubset(progress) or not {"python", "torch_cpu"}.issubset(
        checkpoint["rng_state"]
    ):
        raise ValueError("H7 source lacks sampler/RNG/attempt continuation state")
    return checkpoint


def current_data_provenance(assets: CA16AssetSource) -> dict[str, Any]:
    """Record current verifiable files without claiming absent R10 byte hashes."""
    paths = {
        "cartoonset10k_archive": assets.root
        / "data"
        / "raw"
        / "cartoonset"
        / "archives"
        / "cartoonset10k.tgz",
        "train": assets.root
        / "data"
        / "processed"
        / "dr01_v1"
        / "text"
        / "10k"
        / "train.jsonl",
        "id_val": assets.root
        / "data"
        / "processed"
        / "dr01_v1"
        / "text"
        / "10k"
        / "id_val.jsonl",
        "comp_val": assets.root
        / "data"
        / "processed"
        / "dr01_v1"
        / "text"
        / "10k"
        / "comp_val.jsonl",
        "semantic_mapping": assets.root
        / "data"
        / "processed"
        / "dr01_v1"
        / "semantic_mapping.json",
        "tokenizer": assets.root
        / "data"
        / "processed"
        / "dr01_v1"
        / "text"
        / "10k"
        / "tokenizer_config.json",
    }
    files = {
        name: {
            "path": path.relative_to(assets.root).as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in paths.items()
    }
    if not files["semantic_mapping"]["exists"] or not files["tokenizer"]["exists"]:
        raise FileNotFoundError("H7 canonical semantic mapping or tokenizer is absent")
    mapping = json.loads(paths["semantic_mapping"].read_text(encoding="utf-8"))
    attributes = mapping.get("attributes", {})
    if (
        set(attributes) != {"face_shape", "glasses", "hair_color", "face_color"}
        or sum(len(values) for values in attributes.values()) != 40
    ):
        raise ValueError(
            "H7 canonical mapping is not the frozen four-attribute/40-value contract"
        )
    for reference in FROZEN_MANIFESTS:
        reference.verify(root=assets.root)
    return {
        "current_files": files,
        "historical_byte_identity": "NOT_ESTABLISHED_BY_R10_CHECKPOINT",
        "ca16_frozen_manifest_identity": "VERIFIED",
    }


def _verify_h7_worktree() -> None:
    module_path = Path(__file__).resolve()
    if ROOT.resolve() not in module_path.parents:
        raise RuntimeError("H7 runner was not imported from this H7 worktree")
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if branch != "exp/h7-extended-training":
        raise RuntimeError("H7 preflight requires exp/h7-extended-training")


def preflight(spec: H7ContinuationSpec, assets: CA16AssetSource) -> dict[str, Any]:
    _verify_h7_worktree()
    checkpoint = verify_source_checkpoint(spec, assets)
    provenance = current_data_provenance(assets)
    missing = [
        name for name, item in provenance["current_files"].items() if not item["exists"]
    ]
    if missing:
        raise FileNotFoundError(f"H7 current data provenance unavailable: {missing}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": spec.run_id,
        "asset_root": str(assets.root),
        "source_path": str(spec.source_path(assets)),
        "source_sha256": EXPECTED_SOURCE_SHA256[spec.training_seed],
        "source_step": SOURCE_STEP,
        "target_step": TARGET_STEP,
        "source_config": checkpoint["config"],
        "data_provenance": provenance,
    }


def preflight_all(assets: CA16AssetSource) -> list[dict[str, Any]]:
    """CPU-only asset/state verification for all three authorized continuations."""
    return [preflight(H7ContinuationSpec(seed), assets) for seed in SUPPORTED_SEEDS]


def h7_command(
    spec: H7ContinuationSpec, assets: CA16AssetSource, *, execute: bool = False
) -> tuple[str, ...]:
    """Build an argument vector, never a shell string, for a future operator."""
    mode = "--execute" if execute else "--preflight"
    return (
        "-m",
        "tiny_avatar_diffusion.h7_extended_training",
        "--seed",
        str(spec.training_seed),
        "--ca16-assets-root",
        str(assets.root),
        mode,
    )


def _restore_source(
    checkpoint: Mapping[str, Any], spec: H7ContinuationSpec, device: torch.device
):
    text, denoiser = _new_models_for_spec(device, spec.source_spec)
    optimizer = torch.optim.AdamW(
        [*text.parameters(), *denoiser.parameters()], lr=2e-4, weight_decay=0.0
    )
    scaler = torch.amp.GradScaler("cuda")
    ema = ConditionalEMA(text, denoiser)
    sampler, corruption = torch.Generator(), torch.Generator(device=device)
    # Raw optimization weights and EMA shadows intentionally have independent restores.
    text.load_state_dict(checkpoint["text_encoder_state_dict"])
    denoiser.load_state_dict(checkpoint["conditional_unet_state_dict"])
    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    stored = checkpoint["progress"]
    sampler.set_state(stored["sampler_generator_state"].cpu())
    corruption.set_state(stored["corruption_generator_state"].cpu())
    _restore_rng_state(checkpoint["rng_state"])
    progress = TrainingProgress(
        stored["step"],
        stored["images_seen"],
        stored["batch_offset"],
        stored["permutation"].cpu() if stored["permutation"] is not None else None,
        stored["train_losses"],
        stored["skipped_step_count"],
        stored["attempted_batch_count"],
        stored["skipped_attempt_positions"],
    )
    return text, denoiser, optimizer, scaler, ema, sampler, corruption, progress


def _h7_checkpoint(
    source: Mapping[str, Any],
    spec: H7ContinuationSpec,
    assets: CA16AssetSource,
    progress: TrainingProgress,
    text: torch.nn.Module,
    denoiser: torch.nn.Module,
    ema: ConditionalEMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: torch.Generator,
    corruption: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_id": spec.run_id,
        "continuation": {
            "source_run_id": spec.source_spec.run_id,
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256[spec.training_seed],
            "source_successful_step": SOURCE_STEP,
            "target_successful_step": TARGET_STEP,
            "immutable_source_training_config": source["config"],
        },
        "progress": {
            "step": progress.step,
            "images_seen": progress.images_seen,
            "batch_offset": progress.batch_offset,
            "permutation": progress.permutation,
            "train_losses": progress.train_losses,
            "skipped_step_count": progress.skipped_step_count,
            "attempted_batch_count": progress.attempted_batch_count,
            "skipped_attempt_positions": progress.skipped_attempt_positions,
            "sampler_generator_state": sampler.get_state(),
            "corruption_generator_state": corruption.get_state(),
        },
        "text_encoder_state_dict": text.state_dict(),
        "conditional_unet_state_dict": denoiser.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            **(
                {"torch_cuda_all": torch.cuda.get_rng_state_all()}
                if torch.cuda.is_available()
                else {}
            ),
        },
        "data_provenance": current_data_provenance(assets),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def _restore_h7(
    checkpoint: Mapping[str, Any],
    source: Mapping[str, Any],
    spec: H7ContinuationSpec,
    device: torch.device,
):
    expected = {
        "source_run_id": spec.source_spec.run_id,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256[spec.training_seed],
        "source_successful_step": SOURCE_STEP,
        "target_successful_step": TARGET_STEP,
        "immutable_source_training_config": source["config"],
    }
    if (
        checkpoint.get("schema") != SCHEMA
        or checkpoint.get("run_id") != spec.run_id
        or checkpoint.get("continuation") != expected
    ):
        raise ValueError("H7 recovery checkpoint identity is incompatible")
    stored = checkpoint.get("progress", {})
    if not SOURCE_STEP <= stored.get("step", -1) <= TARGET_STEP:
        raise ValueError("H7 recovery checkpoint has an invalid successful-step count")
    return _restore_source({**source, **checkpoint}, spec, device)


def run(
    spec: H7ContinuationSpec,
    *,
    assets: CA16AssetSource,
    execute: bool = False,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run only when explicitly authorized by the future operator's --execute flag."""
    report = preflight(spec, assets)
    if not execute:
        return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "H7 continuation requires CUDA; preflight never executes training"
        )
    run_dir = spec.output_dir(output_root)
    latest = run_dir / "latest.pt"
    source = verify_source_checkpoint(spec, assets)
    device = torch.device("cuda")
    if latest.exists():
        text, denoiser, optimizer, scaler, ema, sampler, corruption, progress = (
            _restore_h7(
                torch.load(latest, map_location="cpu", weights_only=False),
                source,
                spec,
                device,
            )
        )
    else:
        text, denoiser, optimizer, scaler, ema, sampler, corruption, progress = (
            _restore_source(source, spec, device)
        )
    train = load_text_image_split(
        assets.root / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl",
        "TRAIN",
        image_root=assets.root,
    )
    schedule = LinearDDPMSchedule().to(device)
    started = time.perf_counter()
    while progress.step < TARGET_STEP:
        if (
            progress.permutation is None
            or progress.batch_offset + 32 > train.images.shape[0]
        ):
            progress.permutation = torch.randperm(
                train.images.shape[0], generator=sampler
            )
            progress.batch_offset = 0
        indices = progress.permutation[
            progress.batch_offset : progress.batch_offset + 32
        ]
        progress.batch_offset += 32
        x, ids, mask = _batch_from_indices(train, indices, device)
        text.train()
        denoiser.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            result = conditional_diffusion_loss(
                text, denoiser, schedule, x, ids, mask, generator=corruption
            )
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("non-finite H7 training loss")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        if finalize_amp_attempt(
            scaler=scaler,
            optimizer=optimizer,
            gradients_finite=_finite_gradients(text, denoiser),
            progress=progress,
            ema=ema,
            text_encoder=text,
            denoiser=denoiser,
            ema_decay=0.9999,
        ):
            continue
        progress.images_seen += 32
        progress.train_losses.append(float(result.loss.detach()))
        if progress.step % 1_000 == 0:
            atomic_save_checkpoint(
                latest,
                _h7_checkpoint(
                    source,
                    spec,
                    assets,
                    progress,
                    text,
                    denoiser,
                    ema,
                    optimizer,
                    scaler,
                    sampler,
                    corruption,
                ),
            )
    payload = _h7_checkpoint(
        source,
        spec,
        assets,
        progress,
        text,
        denoiser,
        ema,
        optimizer,
        scaler,
        sampler,
        corruption,
    )
    atomic_save_checkpoint(latest, payload)
    terminal = run_dir / "terminal_40000.pt"
    atomic_save_checkpoint(terminal, payload)
    return {
        **report,
        "status": "COMPLETE",
        "latest_checkpoint": str(latest),
        "terminal_checkpoint": str(terminal),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ca16-assets-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("choose exactly one of --preflight or --execute")
    assets = CA16AssetSource(args.ca16_assets_root)
    if args.seed is None:
        if not args.preflight:
            raise SystemExit("--seed is required for execution")
        print(json.dumps(preflight_all(assets), indent=2, default=str))
        return
    print(
        json.dumps(
            run(H7ContinuationSpec(args.seed), assets=assets, execute=args.execute),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
