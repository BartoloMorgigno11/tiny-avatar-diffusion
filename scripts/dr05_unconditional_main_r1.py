"""Controlled D22-authorized fresh unconditional MAIN R1 execution."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.unconditional_main_training import (
    ROOT,
    TRAIN_MANIFEST,
    UnconditionalMainConfig,
    run_unconditional_main_training,
)


RUN_ID = "dr05_unconditional_main_10k_seed2026_r1"
RUN_DIR = ROOT / "results" / "runs" / RUN_ID
FROZEN_ARTIFACTS = {
    "dr05_protocol": (ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json", "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"),
    "dr05_execution_bindings": (ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json", "e7353eb43a9c284e8880531b663fcf0a53509ca685fc5921b28b15ae2cd1ec6d"),
    "conditional_checkpoint": (ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt", "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"),
    "dr04_evaluator": (ROOT / "artifacts" / "checkpoints" / "dr04_evaluator_epoch41.pt", "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"),
}
SOURCE_FILES = (
    "scripts/dr05_unconditional_main_r1.py",
    "src/tiny_avatar_diffusion/unconditional_main_training.py",
    "src/tiny_avatar_diffusion/conditional_main_training.py",
    "src/tiny_avatar_diffusion/conditional_lr_pilot.py",
    "src/tiny_avatar_diffusion/debug_tiny_overfit.py",
    "src/tiny_avatar_diffusion/unet.py",
    "src/tiny_avatar_diffusion/diffusion.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(name: str, payload: dict[str, Any]) -> None:
    path = RUN_DIR / name
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _command_text(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def _driver() -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(config: UnconditionalMainConfig) -> dict[str, Any]:
    records = [line for line in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    return {
        "captured_before_training": True,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _command_text(["git", "rev-parse", "HEAD"]),
        "git_subject": _command_text(["git", "log", "-1", "--format=%s"]),
        "git_status_short": _command_text(["git", "status", "--short"]),
        "source_files": [{"path": item, "sha256": _sha256(ROOT / item)} for item in SOURCE_FILES],
        "train_manifest": {"path": TRAIN_MANIFEST.relative_to(ROOT).as_posix(), "sha256": _sha256(TRAIN_MANIFEST), "record_count": len(records), "split_role": "TRAIN"},
        "frozen_config": {**config.__dict__, "resolution": list(config.resolution), "milestones": list(config.milestones)},
        "environment": {"python": sys.version, "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "vram_bytes": properties.total_memory, "driver": _driver(), "precision": "CUDA AMP FP16"},
        "fresh_initialization": {"resume_from": None, "initial_successful_step": 0, "forbidden_checkpoints_loaded": False},
    }


def _frozen_integrity() -> dict[str, dict[str, Any]]:
    return {name: {"path": path.relative_to(ROOT).as_posix(), "expected_sha256": expected, "observed_sha256": _sha256(path), "pass": _sha256(path) == expected} for name, (path, expected) in FROZEN_ARTIFACTS.items()}


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to reuse authoritative R1 directory: {RUN_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("D22 R1 requires CUDA AMP FP16")
    config = UnconditionalMainConfig(run_id=RUN_ID)
    if not all(item["pass"] for item in _frozen_integrity().values()):
        raise RuntimeError("frozen artifact integrity check failed before training")
    RUN_DIR.mkdir(parents=True)
    _write_json("source_provenance.json", _provenance(config))
    _write_json("launch_status.json", {"run_id": RUN_ID, "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "resume_from": None, "initial_successful_step": 0})
    started = time.perf_counter()
    result = run_unconditional_main_training(config=config, output_root=RUN_DIR.parent, max_successful_steps=20_000, resume_from=None)
    _write_json("training_telemetry.json", {"run_id": RUN_ID, "completed_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": result["wall_seconds"], "orchestration_wall_seconds": time.perf_counter() - started, "attempted_updates": result["segment_attempted_updates"], "successful_updates": result["successful_steps"], "amp_skipped_updates": result["segment_skipped_updates"], "image_presentations": result["images_seen"], "scaler_initial": result["scaler_start"], "scaler_final": result["scaler_end"], "first_loss": result["losses"][0], "final_loss": result["losses"][-1], "seconds_per_successful_step": result["wall_seconds"] / result["successful_steps"], "images_per_second": result["images_seen"] / result["wall_seconds"], "peak_cuda_allocated_bytes": result["peak_allocated_bytes"], "peak_cuda_reserved_bytes": result["peak_reserved_bytes"], "training_telemetry_only": True, "interruption_resume_events": []})
    _write_json("launch_status.json", {"run_id": RUN_ID, "pid": os.getpid(), "completed_utc": datetime.now(timezone.utc).isoformat(), "state": "completed", "resume_from": None, "initial_successful_step": 0, "successful_steps": result["successful_steps"]})


if __name__ == "__main__":
    main()
