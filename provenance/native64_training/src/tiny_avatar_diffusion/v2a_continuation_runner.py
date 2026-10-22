"""Explicit, separately identified continuation of the authenticated V2-A 40k run.

This launcher intentionally keeps the original matched-budget config immutable.  It
accepts only its authenticated 40k source checkpoint, restores every persisted
state, and writes a new experiment directory.  Importing it performs no GPU work.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.v2_matched_budget_runner import _components_for_variant
from tiny_avatar_diffusion.v2_matched_budget_training import (
    Progress,
    checkpoint_integrity,
    checkpoint_payload,
    config_for_variant,
    optimizer_window,
    preflight,
    save_atomic,
    sha256_file,
    telemetry_counters,
    telemetry_payload,
)

SOURCE_STEP = 40_000
TARGET_STEP = 60_000
CHECKPOINT_STEPS = (45_000, 50_000, 55_000, 60_000)
SOURCE_SHA256 = "15cb3cfc580e549a748aadac513ca90e2d588b281b46fc59b15e1b267ccc54f2"
CONTINUATION_SCHEMA = "tiny_avatar_diffusion.v2a_matched_budget_continuation_v1"


def continuation_spec(source_path: Path) -> dict[str, Any]:
    """Return the immutable identity and safety contract for this experiment."""
    return {
        "schema": CONTINUATION_SCHEMA,
        "source_checkpoint": {
            "path": str(source_path.resolve()),
            "sha256": SOURCE_SHA256,
        },
        "source_successful_updates": SOURCE_STEP,
        "target_successful_updates": TARGET_STEP,
        "checkpoint_successful_updates": list(CHECKPOINT_STEPS),
        "loss_safety": {
            "source_baseline": "mean of the source checkpoint trailing 100 successful losses",
            "window_successful_updates": 100,
            "stop_if_window_mean_exceeds_source_baseline_multiplier": 2.0,
            "stop_if_consecutive_amp_skips": 8,
            "stop_immediately_on_non_finite_loss": True,
        },
    }


def validate_source_checkpoint(
    checkpoint: dict[str, Any], source_path: Path, provenance: dict[str, Any]
) -> Progress:
    """Reject every source except the exact authenticated, complete 40k state."""
    config = config_for_variant("v2_a")
    required = {
        "model",
        "text",
        "ema",
        "optimizer",
        "scaler",
        "sampler_state",
        "noise_generator_state",
        "python_rng",
        "torch_rng",
        "cuda_rng",
        "progress",
        "provenance",
        "config",
        "schema",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"source checkpoint lacks exact-resume state: {missing}")
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError(
            "source checkpoint SHA-256 is not the authenticated matched-budget 40k SHA"
        )
    if checkpoint["schema"] != provenance["schema"] or checkpoint["config"] != asdict(
        config
    ):
        raise ValueError(
            "source checkpoint schema/config differs from authenticated V2-A 40k contract"
        )
    if checkpoint["provenance"] != provenance:
        raise ValueError(
            "source checkpoint data provenance differs from authenticated native64 cache/manifest"
        )
    progress = Progress(**checkpoint["progress"])
    if (
        progress.successful_updates != SOURCE_STEP
        or progress.microbatch_attempts % config.gradient_accumulation
    ):
        raise ValueError(
            "source checkpoint is not a complete authenticated 40k accumulation boundary"
        )
    if not isinstance(progress.losses, list) or len(progress.losses) < 100:
        raise ValueError(
            "source checkpoint lacks the trailing loss history required by the safety contract"
        )
    return progress


def _load_exact_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    text: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: torch.Generator,
    noise: torch.Generator,
) -> tuple[dict[str, dict[str, torch.Tensor]], Progress]:
    model.load_state_dict(checkpoint["model"], strict=True)
    text.load_state_dict(checkpoint["text"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    sampler.set_state(checkpoint["sampler_state"])
    noise.set_state(checkpoint["noise_generator_state"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint["ema"], Progress(**checkpoint["progress"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--verify-cuda-resume",
        action="store_true",
        help="Restore all state on CUDA and verify it without an optimizer update or file write.",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Required before any optimizer update."
    )
    args = parser.parse_args()

    config = config_for_variant("v2_a")
    provenance = preflight(args.data_root, args.cache_dir, config)
    source_path = args.source_checkpoint.resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_progress = validate_source_checkpoint(checkpoint, source_path, provenance)
    spec = continuation_spec(source_path)
    if args.execute and args.verify_cuda_resume:
        raise ValueError("--execute and --verify-cuda-resume are mutually exclusive")
    if not args.execute and not args.verify_cuda_resume:
        print(
            json.dumps(
                {
                    "status": "READY_NOT_EXECUTED",
                    "provenance": provenance,
                    "source_progress": telemetry_counters(source_progress),
                    "continuation": spec,
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.verify_cuda_resume:
        device = torch.device("cuda")
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        _architecture, model, text = _components_for_variant("v2_a", config.seed)
        model.to(device)
        text.to(device)
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *text.parameters()],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda")
        sampler = torch.Generator()
        noise = torch.Generator(device=device)
        ema, restored = _load_exact_state(
            checkpoint,
            model=model,
            text=text,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            noise=noise,
        )
        torch.cuda.synchronize(device)
        verified = {
            "model_tensor_count": len(model.state_dict()),
            "text_tensor_count": len(text.state_dict()),
            "ema_model_tensor_count": len(ema["model"]),
            "ema_text_tensor_count": len(ema["text"]),
            "optimizer_state_entries": len(optimizer.state),
            "scaler_state_exact": scaler.state_dict() == checkpoint["scaler"],
            "sampler_state_exact": torch.equal(
                sampler.get_state(), checkpoint["sampler_state"]
            ),
            "noise_generator_state_exact": torch.equal(
                noise.get_state(), checkpoint["noise_generator_state"]
            ),
            "torch_rng_state_exact": torch.equal(
                torch.get_rng_state(), checkpoint["torch_rng"]
            ),
            "cuda_rng_state_exact": all(
                torch.equal(actual, expected)
                for actual, expected in zip(
                    torch.cuda.get_rng_state_all(), checkpoint["cuda_rng"], strict=True
                )
            ),
            "restored_progress": telemetry_counters(restored),
        }
        if not all(
            value is True for key, value in verified.items() if key.endswith("_exact")
        ):
            raise RuntimeError("exact CUDA resume-state verification failed")
        print(
            json.dumps(
                {
                    "status": "CUDA_RESUME_VERIFIED_NOT_EXECUTED",
                    "verification": verified,
                },
                indent=2,
            )
        )
        return
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(
            "continuation run directory must be new; source and official 40k run are never overwritten"
        )

    records = _load_text_records(
        args.data_root / "data/processed/dr01_v1/text/10k/train.jsonl", "TRAIN"
    )
    images = np.load(args.cache_dir / "train64_rgb_uint8.npy", mmap_mode="r")
    ids = torch.tensor([record["token_ids"] for record in records])
    masks = torch.tensor([record["attention_mask"] for record in records])
    device = torch.device("cuda")
    # Initial seeds are immaterial after the exact state restoration, but make
    # construction itself reproducible before checkpoint tensors replace it.
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    _architecture, model, text = _components_for_variant("v2_a", config.seed)
    model.to(device)
    text.to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *text.parameters()],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    sampler = torch.Generator()
    noise = torch.Generator(device=device)
    ema, progress = _load_exact_state(
        checkpoint,
        model=model,
        text=text,
        optimizer=optimizer,
        scaler=scaler,
        sampler=sampler,
        noise=noise,
    )
    baseline_loss = sum(progress.losses[-100:]) / 100
    run_dir.mkdir(parents=True)
    (run_dir / "continuation_config.json").write_text(
        json.dumps(
            {
                "source_checkpoint_integrity": checkpoint_integrity(source_path),
                "source_provenance": provenance,
                "original_training_config": asdict(config),
                "continuation": spec,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    metrics = run_dir / "metrics.jsonl"
    schedule = LinearDDPMSchedule().to(device)
    initial = telemetry_counters(progress)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    consecutive_skips = 0

    def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            progress.permutation is None
            or progress.offset + config.microbatch_size > len(records)
        ):
            progress.permutation = torch.randperm(len(records), generator=sampler)
            progress.offset = 0
        indices = progress.permutation[
            progress.offset : progress.offset + config.microbatch_size
        ]
        progress.offset += config.microbatch_size
        progress.microbatch_attempts += 1
        progress.images_presented += config.microbatch_size
        x = (
            torch.from_numpy(images[indices.numpy()].copy())
            .to(torch.float32)
            .div_(255)
            .mul_(2)
            .sub_(1)
            .to(device)
        )
        return x, ids[indices].to(device), masks[indices].to(device)

    termination = "TARGET_REACHED"
    while progress.successful_updates < TARGET_STEP:
        losses = []
        for _ in range(config.gradient_accumulation):
            x, token_ids, attention_mask = batch()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                timestep = torch.randint(
                    0, 1000, (config.microbatch_size,), device=device, generator=noise
                )
                epsilon = torch.randn(x.shape, device=device, generator=noise)
                loss = F.mse_loss(
                    model(
                        schedule.q_sample(x, timestep, epsilon),
                        timestep,
                        text(token_ids, attention_mask),
                        attention_mask,
                    ),
                    epsilon,
                )
            if not bool(torch.isfinite(loss)):
                termination = "STOP_NONFINITE_LOSS"
                break
            losses.append(loss)
        if termination != "TARGET_REACHED":
            break
        success, mean = optimizer_window(
            losses,
            optimizer=optimizer,
            scaler=scaler,
            model=model,
            text=text,
            ema=ema,
            progress=progress,
            config=config,
        )
        progress.losses = (progress.losses + [mean])[-100:]
        consecutive_skips = 0 if success else consecutive_skips + 1
        rolling = sum(progress.losses[-100:]) / len(progress.losses)
        row = {
            "successful_updates": progress.successful_updates,
            "attempted_optimizer_updates": progress.attempted_optimizer_updates,
            "amp_skipped_updates": progress.amp_skipped_updates,
            "microbatch_attempts": progress.microbatch_attempts,
            "images_presented": progress.images_presented,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epsilon_mse": mean,
            "rolling_loss": rolling,
            "wall_seconds": time.perf_counter() - started,
            "successful_step": success,
        }
        with metrics.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row) + "\n")
        if success and progress.successful_updates in CHECKPOINT_STEPS:
            save_atomic(
                run_dir / f"checkpoint_success_{progress.successful_updates:06d}.pt",
                checkpoint_payload(
                    config,
                    progress,
                    model,
                    text,
                    ema,
                    optimizer,
                    scaler,
                    sampler,
                    noise,
                    provenance,
                )
                | {"continuation": spec},
            )
        if consecutive_skips >= 8:
            termination = "STOP_CONSECUTIVE_AMP_SKIPS"
            break
        if len(progress.losses) == 100 and rolling > 2.0 * baseline_loss:
            termination = "STOP_SEVERE_LOSS_REGRESSION"
            break
    payload = checkpoint_payload(
        config,
        progress,
        model,
        text,
        ema,
        optimizer,
        scaler,
        sampler,
        noise,
        provenance,
    ) | {"continuation": spec}
    save_atomic(run_dir / "latest.pt", payload)
    torch.cuda.synchronize(device)
    telemetry = telemetry_payload(
        progress, time.perf_counter() - started, baseline=initial
    )
    telemetry.update(
        {
            "termination": termination,
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_baseline_loss": baseline_loss,
            "latest": checkpoint_integrity(run_dir / "latest.pt"),
        }
    )
    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    main()
