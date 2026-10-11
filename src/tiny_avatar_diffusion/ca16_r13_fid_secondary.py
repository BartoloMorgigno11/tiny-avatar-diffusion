"""Read-only-on-input secondary FID evaluation for completed R13 artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_evaluation import CLEAN_FID_VERSION, sha256_file
from tiny_avatar_diffusion.ca16_experiment import ROOT
from tiny_avatar_diffusion.ca16_r12_orchestration import resolve_selected_checkpoints
from tiny_avatar_diffusion.ca16_r13_quality import (
    PROTOCOL_SHA256 as R13_PROTOCOL_SHA256,
)
from tiny_avatar_diffusion.ca16_r13_quality import (
    R13Executor,
    atomic_json,
    build_execution_plan,
    validate_completed_run,
)
from tiny_avatar_diffusion.ca16_r13_quality import (
    load_protocol as load_r13_protocol,
)
from tiny_avatar_diffusion.dr05_final_executor import pixel_sha256

PROTOCOL_ID = "ca16_r13_fid_secondary_v1"
PROTOCOL_PATH = (
    ROOT / "data/processed/ca16_r13_fid_secondary_v1/r13_fid_secondary_protocol.json"
)
PROTOCOL_SHA256 = "75c5031af46c0da174a8de997e9991eb7facf692f3996a77d63bfadccdec00d5"
OUTPUT_NAME = "r13_fid_secondary_summary.json"
EXPECTED_COUNTS = {"ID_VAL": 466, "COMP_VAL": 338, "GLOBAL": 804}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL is absent: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL object required: {path}")
    return rows


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required FID input is absent: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"FID input hash mismatch: {_relative(path)}")
    return actual


def load_protocol() -> dict[str, Any]:
    _require_hash(PROTOCOL_PATH, PROTOCOL_SHA256)
    protocol = _read_json(PROTOCOL_PATH)
    if (
        protocol.get("schema")
        != "tiny_avatar_diffusion.ca16_r13_fid_secondary_protocol_v1"
    ):
        raise ValueError("unexpected R13-FID protocol schema")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected R13-FID protocol identifier")
    if protocol.get("status") not in {"PRE_FREEZE", "FROZEN"}:
        raise ValueError("R13-FID protocol has an invalid status")
    fid = protocol.get("fid", {})
    if fid.get("metrics") != {"FID_ID": 466, "FID_COMP": 338, "FID_GLOBAL": 804}:
        raise ValueError("R13-FID population geometry changed")
    implementation = fid.get("implementation", {})
    if (
        implementation.get("package") != "clean-fid"
        or implementation.get("version") != CLEAN_FID_VERSION
        or implementation.get("mode") != "clean"
        or implementation.get("feature_extractor") != "standard Inception-v3"
        or implementation.get("feature_batch_size") != 8
        or implementation.get("num_workers") != 0
    ):
        raise ValueError("R13-FID Clean-FID contract changed")
    return protocol


def _git_state() -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return (
        run("git", "rev-parse", "HEAD"),
        run("git", "branch", "--show-current"),
        not bool(run("git", "status", "--porcelain", "--untracked-files=normal")),
    )


def _verify_pool(
    pool: Path, source_root: Path, rows: Sequence[Mapping[str, Any]], *, name: str
) -> None:
    expected: dict[str, Path] = {}
    for row in rows:
        index = int(row["planned_sample_index"])
        expected[f"{index:04d}.png"] = source_root / str(row["png_relative_path"])
    actual = {path.name: path for path in pool.glob("*.png")} if pool.is_dir() else {}
    if set(actual) != set(expected):
        raise ValueError(f"R13 FID pool membership mismatch: {name}")
    for filename, source in expected.items():
        if not source.is_file() or sha256_file(source) != sha256_file(actual[filename]):
            raise ValueError(f"R13 FID pool hash mismatch: {name}/{filename}")


def _verify_real_references(
    source_root: Path,
    plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows) or len(by_id) != len(plan):
        raise ValueError("R13 real-reference identifiers are incomplete or duplicate")
    for item in plan:
        sample_id = str(item["sample_id"])
        row = by_id.get(sample_id)
        relative = (
            Path("real_references")
            / str(item["split_role"]).lower()
            / f"{int(item['planned_sample_index']):04d}.png"
        )
        expected = {
            "schema": "tiny_avatar_diffusion.ca16_r13_real_reference_v1",
            "sample_id": sample_id,
            "planned_sample_index": item["planned_sample_index"],
            "stable_record_key": item["stable_record_key"],
            "split_role": item["split_role"],
            "png_relative_path": relative.as_posix(),
            "status": "complete",
        }
        if row is None or any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("R13 real-reference identity mismatch")
        path = source_root / str(row["png_relative_path"])
        if not path.is_file() or sha256_file(path) != row.get("png_sha256"):
            raise ValueError("R13 real-reference PNG mismatch")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row.get(
            "pixel_sha256"
        ):
            raise ValueError("R13 real-reference pixels are invalid")


def _identities(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": "ca16_r13_quality_guardrail_batched_v1",
        "protocol_sha256": R13_PROTOCOL_SHA256,
        "quality_manifest_sha256": "e34db5251d8949a5fb37a905f6e9e4791602047f3153a9a9908d14d9cd85e84f",
        **checkpoint,
        "sampling_batch_size": 8,
    }


def verify_r13_source(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all immutable R13 evidence and return only existing pool paths."""
    source = protocol["source_r13"]
    source_root = ROOT / str(source["root"])
    _require_hash(
        ROOT / str(source["r13_protocol_path"]), str(source["r13_protocol_sha256"])
    )
    _require_hash(
        ROOT / str(source["quality_manifest_path"]),
        str(source["quality_manifest_sha256"]),
    )
    real_manifest = ROOT / str(source["real_reference_manifest_path"])
    _require_hash(real_manifest, str(source["real_reference_manifest_sha256"]))
    if R13_PROTOCOL_SHA256 != source["r13_protocol_sha256"]:
        raise ValueError("loaded R13 implementation does not match FID source protocol")
    r13_protocol = load_r13_protocol()
    plan = build_execution_plan(r13_protocol)
    real_rows = _read_jsonl(real_manifest)
    if len(real_rows) != EXPECTED_COUNTS["GLOBAL"]:
        raise ValueError("R13 real-reference population is incomplete")
    _verify_real_references(source_root, plan, real_rows)
    real_by_role = {
        role: [row for row in real_rows if row.get("split_role") == role]
        for role in ("ID_VAL", "COMP_VAL")
    }
    if {role: len(rows) for role, rows in real_by_role.items()} != {
        "ID_VAL": 466,
        "COMP_VAL": 338,
    }:
        raise ValueError("R13 real-reference role membership is invalid")
    for role, rows in real_by_role.items():
        _verify_pool(
            source_root / "metrics" / "pools" / f"real_{role.lower()}",
            source_root,
            rows,
            name=f"real_{role.lower()}",
        )
    _verify_pool(
        source_root / "metrics" / "pools" / "real_global",
        source_root,
        real_rows,
        name="real_global",
    )
    selected = resolve_selected_checkpoints()
    expected_summaries = source["completed_summary_sha256"]
    if {str(row["run_id"]) for row in selected} != set(expected_summaries):
        raise ValueError(
            "R13-FID selected checkpoint membership differs from frozen source"
        )
    runs: list[dict[str, Any]] = []
    for checkpoint in selected:
        run_id = str(checkpoint["run_id"])
        run_root = source_root / run_id
        _require_hash(run_root / "r13_summary.json", str(expected_summaries[run_id]))
        executor = R13Executor(run_root, plan, _identities(checkpoint))
        summary = validate_completed_run(executor)
        rows = _read_jsonl(run_root / "generation_manifest.jsonl")
        by_role = {
            role: [row for row in rows if row.get("split_role") == role]
            for role in ("ID_VAL", "COMP_VAL")
        }
        for role, role_rows in by_role.items():
            _verify_pool(
                run_root / "metrics" / "pools" / f"generated_{role.lower()}",
                run_root,
                role_rows,
                name=f"{run_id}/generated_{role.lower()}",
            )
        _verify_pool(
            run_root / "metrics" / "pools" / "generated_global",
            run_root,
            rows,
            name=f"{run_id}/generated_global",
        )
        runs.append(
            {
                "run_id": run_id,
                "checkpoint": dict(checkpoint),
                "summary": summary,
                "root": run_root,
            }
        )
    return {
        "source_root": source_root,
        "real_pools": {
            "FID_ID": source_root / "metrics" / "pools" / "real_id_val",
            "FID_COMP": source_root / "metrics" / "pools" / "real_comp_val",
            "FID_GLOBAL": source_root / "metrics" / "pools" / "real_global",
        },
        "runs": runs,
    }


def validate_preflight(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str | None = None
) -> dict[str, Any]:
    protocol = load_protocol()
    if importlib.metadata.version("clean-fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")
    head, branch, clean = _git_state()
    if head != expected_eval_sha or branch != "exp/ca16-ablation" or not clean:
        raise ValueError(
            "R13-FID requires the clean expected exp/ca16-ablation identity"
        )
    if expected_tag is not None:
        tagged = subprocess.run(
            ("git", "rev-parse", f"{expected_tag}^{{commit}}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tagged != head:
            raise ValueError(
                "R13-FID expected tag does not resolve to execution identity"
            )
    source = protocol["source_r13"]
    source_tagged = subprocess.run(
        ("git", "rev-parse", f"{source['annotated_tag']}^{{commit}}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_tagged != source["frozen_commit"]:
        raise ValueError("R13 source tag does not resolve to its frozen commit")
    if output_root.exists():
        raise FileExistsError("R13-FID output root must be unused")
    verified = verify_r13_source(protocol)
    return {
        "status": "preflight_pass",
        "protocol_status": protocol["status"],
        "output_root": str(output_root),
        "source_runs": [run["run_id"] for run in verified["runs"]],
        "geometry": {"models": 6, "metrics_per_model": 3, "fid_values": 18},
    }


def _fid_from_features(
    real_features: np.ndarray, generated_features: np.ndarray
) -> float:
    from cleanfid import fid

    value = fid.frechet_distance(
        np.mean(real_features, axis=0),
        np.cov(real_features, rowvar=False),
        np.mean(generated_features, axis=0),
        np.cov(generated_features, rowvar=False),
    )
    if not np.isfinite(value):
        raise RuntimeError("Clean-FID returned a non-finite FID value")
    return float(value)


def compute_fid_metrics(
    verified: Mapping[str, Any], *, device: torch.device
) -> list[dict[str, Any]]:
    """Compute only FID from verified, pre-existing R13 pools."""
    from cleanfid import fid

    model = fid.build_feature_extractor(
        mode="clean", device=device, use_dataparallel=False
    )
    real_features = {
        name: fid.get_folder_features(
            str(path),
            model=model,
            num_workers=0,
            batch_size=8,
            device=device,
            mode="clean",
            verbose=False,
        )
        for name, path in verified["real_pools"].items()
    }
    results: list[dict[str, Any]] = []
    roles = {"FID_ID": "id_val", "FID_COMP": "comp_val", "FID_GLOBAL": "global"}
    counts = {"FID_ID": 466, "FID_COMP": 338, "FID_GLOBAL": 804}
    for run in verified["runs"]:
        metrics: dict[str, Any] = {}
        for metric, directory in roles.items():
            generated = run["root"] / "metrics" / "pools" / f"generated_{directory}"
            features = fid.get_folder_features(
                str(generated),
                model=model,
                num_workers=0,
                batch_size=8,
                device=device,
                mode="clean",
                verbose=False,
            )
            metrics[metric] = {
                "status": "COMPUTED",
                "value": _fid_from_features(real_features[metric], features),
                "real_count": counts[metric],
                "generated_count": counts[metric],
                "clean_fid_version": CLEAN_FID_VERSION,
                "mode": "clean",
                "feature_extractor": "standard Inception-v3",
            }
        results.append(
            {"run_id": run["run_id"], "checkpoint": run["checkpoint"], "fid": metrics}
        )
    return results


def compute_fid_from_verified_pools(
    real_pools: Mapping[str, Path],
    generated_pools: Mapping[str, Path],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Frozen FID backend for one already-verified ID/COMP/GLOBAL pool set."""
    from cleanfid import fid

    counts = {"ID": 466, "COMP": 338, "GLOBAL": 804}
    if set(real_pools) != set(counts) or set(generated_pools) != set(counts):
        raise ValueError("FID pools require exactly ID/COMP/GLOBAL")
    for role, count in counts.items():
        real, generated = real_pools[role], generated_pools[role]
        if (
            not real.is_dir()
            or not generated.is_dir()
            or len(list(real.glob("*.png"))) != count
            or len(list(generated.glob("*.png"))) != count
        ):
            raise ValueError("FID verified pool count mismatch")
    model = fid.build_feature_extractor(
        mode="clean", device=device, use_dataparallel=False
    )
    result: dict[str, Any] = {}
    for role, count in counts.items():
        real_features = fid.get_folder_features(
            str(real_pools[role]),
            model=model,
            num_workers=0,
            batch_size=8,
            device=device,
            mode="clean",
            verbose=False,
        )
        generated_features = fid.get_folder_features(
            str(generated_pools[role]),
            model=model,
            num_workers=0,
            batch_size=8,
            device=device,
            mode="clean",
            verbose=False,
        )
        result[f"FID_{role}"] = {
            "status": "COMPUTED",
            "value": _fid_from_features(real_features, generated_features),
            "real_count": count,
            "generated_count": count,
            "clean_fid_version": CLEAN_FID_VERSION,
            "mode": "clean",
            "feature_extractor": "standard Inception-v3",
        }
    return result


def summarize_fid_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired FID descriptively; lower values are better."""
    values = {
        (
            str(row["checkpoint"]["architecture_arm"]),
            int(row["checkpoint"]["training_seed"]),
        ): row["fid"]
        for row in results
    }
    expected = {
        (arm, seed) for arm in ("ca8", "ca8_ca16") for seed in (2026, 2027, 2028)
    }
    if set(values) != expected:
        raise ValueError("R13-FID result membership is not the frozen six-run matrix")
    metrics = ("FID_ID", "FID_COMP", "FID_GLOBAL")
    paired = {
        str(seed): {
            metric: float(
                values[("ca8_ca16", seed)][metric]["value"]
                - values[("ca8", seed)][metric]["value"]
            )
            for metric in metrics
        }
        for seed in (2026, 2027, 2028)
    }
    means = {
        metric: {
            "ca8": float(
                sum(
                    values[("ca8", seed)][metric]["value"]
                    for seed in (2026, 2027, 2028)
                )
                / 3
            ),
            "ca8_ca16": float(
                sum(
                    values[("ca8_ca16", seed)][metric]["value"]
                    for seed in (2026, 2027, 2028)
                )
                / 3
            ),
            "ca8_ca16_minus_ca8": float(
                sum(paired[str(seed)][metric] for seed in (2026, 2027, 2028)) / 3
            ),
        }
        for metric in metrics
    }
    return {
        "comparison": "CA8+CA16 minus CA8; lower FID is better",
        "paired_differences": paired,
        "means": means,
        "inference": "descriptive only; no confidence intervals or significance tests are computed",
    }


def execute_fid(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str
) -> dict[str, Any]:
    preflight = validate_preflight(
        output_root=output_root,
        expected_eval_sha=expected_eval_sha,
        expected_tag=expected_tag,
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "R13-FID requires CUDA for the frozen Clean-FID execution profile"
        )
    protocol = load_protocol()
    if protocol["status"] != "FROZEN":
        raise RuntimeError("R13-FID execution requires a frozen protocol")
    verified = verify_r13_source(protocol)
    results = compute_fid_metrics(verified, device=torch.device("cuda"))
    payload = {
        "schema": "tiny_avatar_diffusion.ca16_r13_fid_secondary_result_v1",
        "status": "COMPLETE",
        "secondary_post_hoc_diagnostic": True,
        "interpretation": protocol["scientific_scope"]["interpretation"],
        "protocol": {
            "path": _relative(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
            "status": protocol["status"],
        },
        "source_r13": protocol["source_r13"],
        "preflight": preflight,
        "environment": {
            "device": "cuda",
            "torch": str(torch.__version__),
            "clean_fid": importlib.metadata.version("clean-fid"),
        },
        "results": results,
        "summary": summarize_fid_results(results),
    }
    atomic_json(output_root / OUTPUT_NAME, payload)
    return payload
