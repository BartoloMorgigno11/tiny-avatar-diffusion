"""H7 R13 KID adapter over a completed shared pool; it never generates images."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.ca16_evaluation import sha256_file
from tiny_avatar_diffusion.ca16_r13_quality import _metric as r13_kid_metric
from tiny_avatar_diffusion.h7_r11_evaluation import (
    _external_path,
    verify_ca16_assets,
    verify_terminal_checkpoint,
)
from tiny_avatar_diffusion.h7_r13_pool import OUTPUT_ROOT, prepare_plan, validate_pool

R13_ROOT = "results/r13_quality_guardrail_batched_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")  # noqa: TRY004
    return value


def historical_baseline(ca16_assets_root: Path, seed: int) -> dict[str, Any]:
    if seed not in (2026, 2027, 2028):
        raise ValueError("unauthorized H7 seed")
    path = _external_path(
        ca16_assets_root,
        f"{R13_ROOT}/ca16_ablation_ca8_ca16_seed{seed}/r13_summary.json",
    )
    summary = _json(path)
    if (
        summary.get("schema") != "tiny_avatar_diffusion.ca16_r13_quality_summary_v1"
        or summary.get("status") != "COMPLETE"
    ):
        raise ValueError("historical R13 summary is invalid")
    metrics = summary.get("metrics", {})
    if set(metrics) != {"KID_ID", "KID_COMP", "KID_GLOBAL"}:
        raise ValueError("historical R13 KID metrics are incomplete")
    return {"path": str(path), "sha256": sha256_file(path), "metrics": metrics}


def preflight(
    *, ca16_assets_root: Path, terminal_checkpoint: Path, seed: int
) -> dict[str, Any]:
    verify_ca16_assets(ca16_assets_root)
    terminal = verify_terminal_checkpoint(terminal_checkpoint, seed)
    plan, identity = prepare_plan(ca16_assets_root)
    pool = OUTPUT_ROOT / f"r13_shared_pool_seed{seed}"
    marker = _json(pool / "R13_POOL_COMPLETE.json")
    if marker.get("status") != "COMPLETE":
        raise ValueError("H7 shared R13 pool is not complete")
    rows = validate_pool(pool, plan, identity, terminal)
    return {
        "terminal": terminal,
        "pool": str(pool),
        "pool_manifest_sha256": sha256_file(pool / "pool_manifest.jsonl"),
        "plan_identity": identity,
        "image_count": len(rows),
        "baseline_20k": historical_baseline(ca16_assets_root, seed),
        "kid": {
            "clean_fid_version": "0.1.35",
            "num_subsets": 100,
            "subset_seed": 2026,
            "roles": {"KID_ID": 466, "KID_COMP": 338, "KID_GLOBAL": 804},
        },
    }


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def role_views(pool: Path, rows: list[dict[str, Any]], output: Path) -> dict[str, Path]:
    """Materialize exact-byte, H7-owned views required by frozen R13 _metric."""
    roles = {
        "ID": [r for r in rows if r["split_role"] == "ID_VAL"],
        "COMP": [r for r in rows if r["split_role"] == "COMP_VAL"],
    }
    roles["GLOBAL"] = roles["ID"] + roles["COMP"]
    if {key: len(value) for key, value in roles.items()} != {
        "ID": 466,
        "COMP": 338,
        "GLOBAL": 804,
    }:
        raise ValueError("KID canonical role membership mismatch")
    result: dict[str, Path] = {}
    for role, items in roles.items():
        directory = output / "views" / f"generated_{role.lower()}"
        directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(items):
            source = pool / row["png_relative_path"]
            target = directory / f"{index:04d}.png"
            if target.exists():
                if sha256_file(target) != row["png_sha256"]:
                    raise ValueError("KID role view hash mismatch")
            else:
                shutil.copyfile(source, target)
                if sha256_file(target) != row["png_sha256"]:
                    raise RuntimeError("KID role-view copy hash mismatch")
        if len(list(directory.glob("*.png"))) != len(items):
            raise ValueError("KID role view has foreign or missing images")
        result[role] = directory
    return result


def execute(
    *, ca16_assets_root: Path, terminal_checkpoint: Path, seed: int
) -> dict[str, Any]:
    """Future CUDA-only metric calculation; shared pool must already be complete."""
    if not torch.cuda.is_available():
        raise RuntimeError("H7 KID execution requires CUDA and never generates images")
    report = preflight(
        ca16_assets_root=ca16_assets_root,
        terminal_checkpoint=terminal_checkpoint,
        seed=seed,
    )
    output = OUTPUT_ROOT / f"r13_kid_seed{seed}"
    complete = output / "H7_KID_COMPLETE.json"
    if complete.exists():
        summary = _json(complete)
        if summary.get("terminal") != report["terminal"]:
            raise ValueError("completed KID identity mismatch")
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
    counts = {"ID": 466, "COMP": 338, "GLOBAL": 804}
    for role, count in counts.items():
        if len(list(real[role].glob("*.png"))) != count:
            raise ValueError("historical real-reference pool mismatch")
    metrics = {
        f"KID_{role}": r13_kid_metric(
            real[role],
            generated[role],
            device=torch.device("cuda"),
            expected_count=count,
        )
        for role, count in counts.items()
    }
    baseline = report["baseline_20k"]["metrics"]
    summary = {
        "schema": "tiny_avatar_diffusion.h7_r13_kid_v1",
        "status": "COMPLETE",
        "terminal": report["terminal"],
        "pool_manifest_sha256": report["pool_manifest_sha256"],
        "metrics": metrics,
        "baseline_20k": baseline,
        "paired_40k_minus_20k": {
            key: metrics[key]["value"] - baseline[key]["value"] for key in metrics
        },
    }
    _atomic(complete, summary)
    return summary
