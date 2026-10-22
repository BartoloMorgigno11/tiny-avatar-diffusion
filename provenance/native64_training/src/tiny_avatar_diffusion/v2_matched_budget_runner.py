"""Shared authorized-only launcher for fresh V2 matched-budget training."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.v2_matched_budget_training import (
    Progress,
    checkpoint_integrity,
    checkpoint_payload,
    clone_state,
    config_for_variant,
    optimizer_window,
    preflight,
    save_atomic,
    schema_for_config,
    telemetry_counters,
    telemetry_payload,
)
from tiny_avatar_diffusion.v2_native64 import make_v2_model, parameter_count


def _components_for_variant(variant: str, seed: int):
    """Build the selected model and preserve the historical shared A/B initialization.

    V2-B's CA32 is initialized after its shared V2-A-compatible layers.  Its
    text encoder is initialized in a forked stream matching fresh V2-A, so the
    shared model and text tensors match the historic paired-initialization
    protocol while CA32 remains newly initialized.
    """
    torch.manual_seed(seed)
    architecture, model = make_v2_model(variant)
    if variant == "v2_a":
        text = TextConditioningEncoder()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            make_v2_model("v2_a")
            text = TextConditioningEncoder()
    expected = {"v2_a": 5_230_403, "v2_b": 5_247_171}[variant]
    if parameter_count(model, text)["total"] != expected:
        raise ValueError("V2 architecture parameter-count identity mismatch")
    if (variant == "v2_b") != hasattr(model, "decoder32_image_text_cross_attention"):
        raise ValueError("V2-B CA32 architecture identity mismatch")
    return architecture, model, text


def main_for_variant(variant: str) -> None:
    config = config_for_variant(variant)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-updates", type=int)
    mode.add_argument("--full-experiment", action="store_true")
    args = parser.parse_args()
    provenance = preflight(args.data_root, args.cache_dir, config)
    if args.smoke_updates is not None and not 1 <= args.smoke_updates <= 100:
        raise ValueError("--smoke-updates is strictly limited to 1..100")
    if not args.execute:
        print(json.dumps({"status": "READY_NOT_EXECUTED", "provenance": provenance, "command_requires": "--execute plus --smoke-updates (1..100) or explicit --full-experiment"}, indent=2))
        return
    if (args.smoke_updates is None) == (not args.full_experiment):
        raise ValueError("execution requires exactly one explicit mode: --smoke-updates or --full-experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    run_dir = args.run_dir.resolve()
    latest = run_dir / "latest.pt"
    if run_dir.exists() and not args.resume:
        raise FileExistsError("new matched-budget run directory already exists; use explicit --resume")
    if args.resume and latest.exists():
        raise FileExistsError("resume target directory must be new/empty")
    records = _load_text_records(args.data_root / "data/processed/dr01_v1/text/10k/train.jsonl", "TRAIN")
    images = np.load(args.cache_dir / "train64_rgb_uint8.npy", mmap_mode="r")
    ids = torch.tensor([record["token_ids"] for record in records])
    masks = torch.tensor([record["attention_mask"] for record in records])
    random.seed(config.seed); torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")
    _architecture, model, text = _components_for_variant(config.variant, config.seed)
    model.to(device); text.to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *text.parameters()], lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = {"model": clone_state(model.state_dict()), "text": clone_state(text.state_dict())}
    sampler = torch.Generator().manual_seed(config.seed)
    noise = torch.Generator(device=device).manual_seed(config.seed)
    progress = Progress()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != schema_for_config(config) or checkpoint.get("config") != asdict(config) or checkpoint.get("provenance") != provenance:
            raise ValueError("strict incompatible matched-budget resume rejection")
        model.load_state_dict(checkpoint["model"], strict=True); text.load_state_dict(checkpoint["text"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"]); ema = checkpoint["ema"]
        progress = Progress(**checkpoint["progress"]); sampler.set_state(checkpoint["sampler_state"]); noise.set_state(checkpoint["noise_generator_state"])
        random.setstate(checkpoint["python_rng"]); torch.set_rng_state(checkpoint["torch_rng"]); torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    segment_baseline = telemetry_counters(progress)

    def batch():
        if progress.permutation is None or progress.offset + config.microbatch_size > len(records):
            progress.permutation = torch.randperm(len(records), generator=sampler); progress.offset = 0
        indices = progress.permutation[progress.offset:progress.offset + config.microbatch_size]
        progress.offset += config.microbatch_size; progress.microbatch_attempts += 1; progress.images_presented += config.microbatch_size
        x = torch.from_numpy(images[indices.numpy()].copy()).to(torch.float32).div_(255).mul_(2).sub_(1).to(device)
        return x, ids[indices].to(device), masks[indices].to(device)

    schedule = LinearDDPMSchedule().to(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(json.dumps({"provenance": provenance, "config": asdict(config)}, indent=2) + "\n")
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); metrics = run_dir / "metrics.jsonl"
    target_updates = config.successful_updates if args.full_experiment else args.smoke_updates
    if progress.successful_updates >= target_updates:
        raise ValueError("resume checkpoint already meets/exceeds requested execution target")
    while progress.successful_updates < target_updates:
        losses = []
        for _ in range(config.gradient_accumulation):
            x, token_ids, attention_mask = batch()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                timestep = torch.randint(0, 1000, (config.microbatch_size,), device=device, generator=noise)
                epsilon = torch.randn(x.shape, device=device, generator=noise)
                loss = F.mse_loss(model(schedule.q_sample(x, timestep, epsilon), timestep, text(token_ids, attention_mask), attention_mask), epsilon)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            losses.append(loss)
        success, mean = optimizer_window(losses, optimizer=optimizer, scaler=scaler, model=model, text=text, ema=ema, progress=progress, config=config)
        progress.losses = (progress.losses + [mean])[-100:]
        row = {"successful_updates": progress.successful_updates, "attempted_optimizer_updates": progress.attempted_optimizer_updates, "amp_skipped_updates": progress.amp_skipped_updates, "microbatch_attempts": progress.microbatch_attempts, "images_presented": progress.images_presented, "learning_rate": optimizer.param_groups[0]["lr"], "epsilon_mse": mean, "wall_seconds": time.perf_counter() - started, "successful_step": success}
        with metrics.open("a", encoding="utf8") as stream: stream.write(json.dumps(row) + "\n")
        if success and progress.successful_updates % config.checkpoint_interval_updates == 0:
            payload = checkpoint_payload(config, progress, model, text, ema, optimizer, scaler, sampler, noise, provenance)
            save_atomic(run_dir / f"checkpoint_success_{progress.successful_updates:06d}.pt", payload); save_atomic(latest, payload)
    payload = checkpoint_payload(config, progress, model, text, ema, optimizer, scaler, sampler, noise, provenance); save_atomic(latest, payload)
    terminal = latest
    if args.smoke_updates is not None:
        terminal = run_dir / f"smoke_terminal_success_{progress.successful_updates:06d}.pt"; save_atomic(terminal, payload)
    torch.cuda.synchronize(device)
    telemetry = telemetry_payload(progress, time.perf_counter() - started, baseline=segment_baseline); telemetry["terminal_checkpoint"] = checkpoint_integrity(terminal)
    (run_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    (run_dir / "checkpoint_integrity.json").write_text(json.dumps({"latest": checkpoint_integrity(latest), "terminal": checkpoint_integrity(terminal)}, indent=2) + "\n")
