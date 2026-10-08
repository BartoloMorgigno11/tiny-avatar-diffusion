"""Foreground-only, non-scientific CUDA validation pilot for R11 batching."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import tiny_avatar_diffusion.ca16_r11_orchestration as r11
from tiny_avatar_diffusion.ca16_evaluation import (
    load_ema_checkpoint,
    sample_ema_record,
    sample_ema_records_batch,
)
from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _implementation_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts" / "run_ca16_r11_batched_pilot.py",
        root / "src" / "tiny_avatar_diffusion" / "conditional_sampling.py",
        root / "src" / "tiny_avatar_diffusion" / "ca16_evaluation.py",
        root / "src" / "tiny_avatar_diffusion" / "diffusion.py",
        root / "src" / "tiny_avatar_diffusion" / "unet.py",
        root / "src" / "tiny_avatar_diffusion" / "text_conditioning.py",
        root / "src" / "tiny_avatar_diffusion" / "ca16_r11_orchestration.py",
    )
    return _sha256_bytes(b"".join(path.read_bytes() for path in paths))


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype != torch.float32 or tensor.shape != (3, 32, 32):
        raise ValueError("cached reference tensor has invalid dtype or shape")
    return _sha256_bytes(tensor.numpy().tobytes())


def _identity(
    *, arm: ArchitectureArm, checkpoint_sha256: str, record: dict[str, Any]
) -> dict[str, Any]:
    token_identity = _sha256_bytes(
        json.dumps(
            {
                "token_ids": record["token_ids"],
                "attention_mask": record["attention_mask"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return {
        "schema": "tiny_avatar_diffusion.ca16_r11_serial_reference_cache_v1",
        "arm": arm.value,
        "checkpoint_sha256": checkpoint_sha256,
        "stable_record_key": record["stable_record_key"],
        "generation_seed": record["generation_seed"],
        "token_identity": token_identity,
        "token_source_sha256": r11.sha256_file(
            r11.TOKEN_SOURCE_PATHS[record["split_role"]]
        ),
        "generation": {
            "timesteps": 1000,
            "precision": "fp32",
            "sampler": "pixel_ddpm_ancestral",
            "clip_denoised": True,
            "variance": "fixed_small",
        },
        "implementation_sha256": _implementation_digest(),
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "cudnn": None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version()),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }


def _cache_path(output_dir: Path, identity: dict[str, Any]) -> Path:
    key = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return output_dir / "serial_references" / identity["arm"] / f"{key}.pt"


def _load_cached_reference(
    path: Path, identity: dict[str, Any], device: torch.device
) -> torch.Tensor | None:
    temporary = path.with_suffix(".pt.tmp")
    if temporary.exists():
        raise RuntimeError(f"incomplete serial-reference cache entry: {temporary}")
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("identity") != identity:
        raise RuntimeError("serial-reference cache identity mismatch")
    tensor = payload.get("tensor")
    if not isinstance(tensor, torch.Tensor) or payload.get(
        "tensor_sha256"
    ) != _tensor_digest(tensor):
        raise RuntimeError("serial-reference cache tensor is corrupt")
    return tensor.to(device)


def _write_cached_reference(
    path: Path, identity: dict[str, Any], tensor: torch.Tensor
) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite serial-reference cache entry")
    stored = tensor.detach().cpu().contiguous()
    payload = {
        "identity": identity,
        "tensor": stored,
        "tensor_sha256": _tensor_digest(stored),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    if temporary.exists():
        raise RuntimeError("incomplete serial-reference cache entry exists")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    error = (reference - candidate).abs()
    relative_mask = reference.abs() >= 1e-3
    relative = (
        (error[relative_mask] / reference.abs()[relative_mask]).max().item()
        if bool(relative_mask.any())
        else 0.0
    )
    ref_png = r11.tensor_to_pixels(reference)
    actual_png = r11.tensor_to_pixels(candidate)
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "max_relative_error_abs_reference_ge_1e-3": float(relative),
        "png_different_pixels": int((ref_png != actual_png).any(axis=2).sum()),
        "png_total_pixels": 32 * 32,
    }


def _progress_hook() -> tuple[callable, dict[str, int]]:
    state = {"calls": 0}

    def hook(*_args: object) -> None:
        state["calls"] += 1
        if state["calls"] % 100 == 0:
            print(f"{_utc()} heartbeat denoise_steps={state['calls']}", flush=True)

    return hook, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=[arm.value for arm in ArchitectureArm], required=True
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.batch_size not in (1, 4, 8, 16, 32)
        or args.warmups < 0
        or args.repetitions < 1
    ):
        parser.error(
            "batch size must be 1, 4, 8, 16, or 32; warmups/repetitions invalid"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("R11 batched pilot requires CUDA")
    interrupted = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(
            f"{_utc()} interruption requested; completing no further chains", flush=True
        )

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    device = torch.device("cuda")
    arm = ArchitectureArm(args.arm)
    spec = CA16RunSpec(arm, 2026)
    output = args.output_dir / f"{arm.value}_b{args.batch_size}.json"
    if output.exists():
        print(f"{_utc()} already complete: {output}", flush=True)
        return 0
    print(f"{_utc()} loading arm={arm.value} batch={args.batch_size}", flush=True)
    inventory = r11.load_checkpoint_inventory()
    entry = r11.inventory_entry(inventory, spec, 5_000)
    text, unet, _ = load_ema_checkpoint(
        r11.verify_training_checkpoint_before_load(entry), spec, 5_000, device=device
    )
    records = [
        r11.hydrate_selection_record(row, r11.load_tokenized_source_index())
        for row in r11.canonical_records()[: args.batch_size]
    ]
    schedule = LinearDDPMSchedule(num_timesteps=1_000).to(device)
    hook, counters = _progress_hook()
    handle = unet.register_forward_hook(hook)
    try:
        print(f"{_utc()} parity reference serial start", flush=True)
        serial: list[torch.Tensor] = []
        cache_events: list[str] = []
        for record in records:
            identity = _identity(
                arm=arm, checkpoint_sha256=entry["sha256"], record=record
            )
            cache_path = _cache_path(args.output_dir, identity)
            reference = _load_cached_reference(cache_path, identity, device)
            if reference is None:
                reference = sample_ema_record(
                    schedule, text, unet, record, device=device
                )
                _write_cached_reference(cache_path, identity, reference)
                cache_events.append("GENERATED")
            else:
                cache_events.append("REUSED")
            serial.append(reference)
        print(f"{_utc()} parity batched start", flush=True)
        batched = sample_ema_records_batch(schedule, text, unet, records, device=device)
        parity = [
            _metrics(reference, actual)
            for reference, actual in zip(serial, batched, strict=True)
        ]
        repeated = sample_ema_records_batch(
            schedule, text, unet, records, device=device
        )
        deterministic = bool(torch.equal(batched, repeated))
        times: list[float] = []
        for index in range(args.warmups + args.repetitions):
            if interrupted:
                raise KeyboardInterrupt("pilot interrupted")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = sample_ema_records_batch(
                schedule, text, unet, records, device=device
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError("nonfinite pilot output")
            print(
                f"{_utc()} chain={index + 1}/{args.warmups + args.repetitions} seconds={elapsed:.3f}",
                flush=True,
            )
            if index >= args.warmups:
                times.append(elapsed)
        free, total = torch.cuda.mem_get_info(device)
        payload = {
            "schema": "tiny_avatar_diffusion.ca16_r11_batched_pilot_v1",
            "completed_utc": _utc(),
            "scientific_execution": False,
            "arm": arm.value,
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "times_seconds": times,
            "median_seconds_per_batch": statistics.median(times),
            "median_seconds_per_image": statistics.median(times) / args.batch_size,
            "images_per_second": args.batch_size / statistics.median(times),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_bytes": free,
            "total_bytes": total,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "denoiser_calls": counters["calls"],
            "serial_reference_cache_events": cache_events,
            "parity": parity,
            "repeated_batched_bitwise_equal": deterministic,
        }
        _atomic_json(output, payload)
        print(f"{_utc()} complete result={output}", flush=True)
        return 0
    finally:
        handle.remove()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"{_utc()} interrupted", flush=True)
        raise SystemExit(130)
