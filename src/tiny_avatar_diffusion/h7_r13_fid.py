"""H7 secondary FID over the verified R13 shared pool; never samples images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.ca16_evaluation import sha256_file
from tiny_avatar_diffusion.ca16_r13_fid_secondary import (
    CLEAN_FID_VERSION,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    compute_fid_from_verified_pools,
)
from tiny_avatar_diffusion.h7_r11_evaluation import _external_path
from tiny_avatar_diffusion.h7_r13_kid import (
    R13_ROOT,
    _atomic,
    _json,
    preflight,
    role_views,
)
from tiny_avatar_diffusion.h7_r13_pool import OUTPUT_ROOT, prepare_plan, validate_pool

FID_PROTOCOL = (
    "data/processed/ca16_r13_fid_secondary_v1/r13_fid_secondary_protocol.json"
)
FID_SUMMARY = "results/r13_fid_secondary_v1/r13_fid_secondary_summary.json"


def _fid_protocol(ca16_assets_root: Path) -> dict[str, Any]:
    path = _external_path(ca16_assets_root, FID_PROTOCOL)
    if sha256_file(path) != PROTOCOL_SHA256:
        raise ValueError("frozen CA16 R13-FID protocol hash mismatch")
    protocol = _json(path)
    implementation = protocol.get("fid", {}).get("implementation", {})
    if (
        protocol.get("schema")
        != "tiny_avatar_diffusion.ca16_r13_fid_secondary_protocol_v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "FROZEN"
        or implementation
        != {
            "package": "clean-fid",
            "version": CLEAN_FID_VERSION,
            "mode": "clean",
            "feature_extractor": "standard Inception-v3",
            "feature_batch_size": 8,
            "num_workers": 0,
        }
    ):
        raise ValueError("frozen CA16 R13-FID protocol contract mismatch")
    return protocol


def historical_baseline(ca16_assets_root: Path, seed: int) -> dict[str, Any]:
    if seed not in (2026, 2027, 2028):
        raise ValueError("unauthorized H7 seed")
    protocol = _fid_protocol(ca16_assets_root)
    path = _external_path(ca16_assets_root, FID_SUMMARY)
    summary = _json(path)
    if summary.get("status") != "COMPLETE":
        raise ValueError("historical R13-FID summary is incomplete")
    rows = summary.get("results")
    if not isinstance(rows, list):
        raise ValueError("historical R13-FID results are invalid")  # noqa: TRY004
    row = next(
        (
            item
            for item in rows
            if item.get("checkpoint", {}).get("architecture_arm") == "ca8_ca16"
            and item.get("checkpoint", {}).get("training_seed") == seed
        ),
        None,
    )
    metrics = row.get("fid") if isinstance(row, dict) else None
    if not isinstance(metrics, dict) or set(metrics) != {
        "FID_ID",
        "FID_COMP",
        "FID_GLOBAL",
    }:
        raise ValueError("historical R13-FID seed baseline is incomplete")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "protocol_sha256": PROTOCOL_SHA256,
        "source_r13": protocol["source_r13"],
        "metrics": metrics,
    }


def execute(
    *, ca16_assets_root: Path, terminal_checkpoint: Path, seed: int
) -> dict[str, Any]:
    """Future CUDA-only FID calculation over the existing KID-compatible views."""
    if not torch.cuda.is_available():
        raise RuntimeError("H7 FID execution requires CUDA and never generates images")
    report = preflight(
        ca16_assets_root=ca16_assets_root,
        terminal_checkpoint=terminal_checkpoint,
        seed=seed,
    )
    baseline = historical_baseline(ca16_assets_root, seed)
    output = OUTPUT_ROOT / f"r13_fid_seed{seed}"
    complete = output / "H7_FID_COMPLETE.json"
    if complete.exists():
        summary = _json(complete)
        if (
            summary.get("terminal") != report["terminal"]
            or summary.get("pool_manifest_sha256") != report["pool_manifest_sha256"]
            or summary.get("baseline_20k", {}).get("sha256") != baseline["sha256"]
        ):
            raise ValueError("completed FID identity mismatch")
        return summary
    plan, identity = prepare_plan(ca16_assets_root)
    rows = validate_pool(Path(report["pool"]), plan, identity, report["terminal"])
    generated = role_views(Path(report["pool"]), rows, output)
    real_root = _external_path(ca16_assets_root, R13_ROOT) / "metrics" / "pools"
    real = {
        "ID": real_root / "real_id_val",
        "COMP": real_root / "real_comp_val",
        "GLOBAL": real_root / "real_global",
    }
    metrics = compute_fid_from_verified_pools(
        real, generated, device=torch.device("cuda")
    )
    summary = {
        "schema": "tiny_avatar_diffusion.h7_r13_fid_v1",
        "status": "COMPLETE",
        "terminal": report["terminal"],
        "pool_manifest_sha256": report["pool_manifest_sha256"],
        "metrics": metrics,
        "baseline_20k": baseline,
        "paired_40k_minus_20k": {
            key: metrics[key]["value"] - baseline["metrics"][key]["value"]
            for key in metrics
        },
        "warning": "FID is secondary, descriptive, and small-sample sensitive; it does not select checkpoints.",
    }
    _atomic(complete, summary)
    return summary
