"""Read-only canonical DR04 diversity evaluation utilities for DR05.

This module never generates images and only writes below a caller-selected DR05
output root.  The actual LPIPS execution is deliberately opt-in via the CLI.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image


SEEDS = (2026, 2027, 2028, 2029, 2030)
OOD_FAMILIES = ("OOD1", "OOD2", "OOD3", "OOD4")
SOURCE_ROOT = Path("results/runs/dr04_final_conditioning_v1")
DEFAULT_OUTPUT_ROOT = Path("results/runs/dr05_final_diversity_v1")
EXPECTED_SOURCE_HASHES = {
    "metrics.json": "727eece28af8575d6e4dc6062279fb97e1dc7d3c58a40424bec16b110ca66580",
    "final_conditioning_summary.json": "307138ac4aa1d26a0b478ab2a1a1703fa248406187681896f4834147902e26c6",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_artifact_hashes(source_root: Path) -> dict[str, str]:
    observed = {name: sha256_file(source_root / name) for name in EXPECTED_SOURCE_HASHES}
    if observed != EXPECTED_SOURCE_HASHES:
        raise ValueError("canonical DR04 artifact hash mismatch")
    return observed


def reject_overlapping_output_root(source_root: Path, output_root: Path) -> None:
    source, output = source_root.resolve(), output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("DR05 output root must be separate from the canonical DR04 input root")


def validate_population(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Validate frozen prompt/seed geometry without decoding images."""
    records = list(rows)
    if len(records) != 1260:
        raise ValueError("canonical diversity population must contain 1260 image rows")
    slots = [(row.get("prompt_id"), row.get("seed")) for row in records]
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate prompt-seed row")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get("prompt_id"))].append(row)
    if len(groups) != 252 or any(len(group) != 5 for group in groups.values()):
        raise ValueError("canonical population must contain 252 five-image prompts")
    if any(tuple(sorted(row.get("seed") for row in group)) != SEEDS for group in groups.values()):
        raise ValueError("every prompt must use the exact frozen seed set")
    id_groups = [group for group in groups.values() if group[0].get("panel_split") == "ID"]
    ood_groups = [group for group in groups.values() if group[0].get("panel_split") == "OOD"]
    if len(id_groups) != 152 or len(ood_groups) != 100:
        raise ValueError("canonical ID/OOD prompt geometry mismatch")
    for family in OOD_FAMILIES:
        if sum(group[0].get("ood_family") == family for group in ood_groups) != 25:
            raise ValueError("canonical OOD family prompt geometry mismatch")
    if any(group[0].get("ood_family") not in OOD_FAMILIES for group in ood_groups):
        raise ValueError("OOD row has missing or unknown family")
    return dict(groups)


def validate_images(rows: Iterable[dict[str, Any]], source_root: Path) -> None:
    for row in rows:
        relative = row.get("png_path")
        if not isinstance(relative, str):
            raise ValueError("generation row has no PNG path")
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError("canonical diversity image must be RGB")
            if image.size != (32, 32):
                raise ValueError("canonical diversity image must be 32x32")


def enumerate_pairs(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for prompt_id in sorted(groups):
        group = sorted(groups[prompt_id], key=lambda row: int(row["seed"]))
        for left, right in itertools.combinations(group, 2):
            pairs.append({
                "pair_id": f"{prompt_id}:{left['seed']}:{right['seed']}",
                "prompt_id": prompt_id,
                "panel_split": left["panel_split"],
                "ood_family": left.get("ood_family"),
                "control_role": left.get("control_role"),
                "seed_a": left["seed"], "seed_b": right["seed"],
                "png_a": left["png_path"], "png_b": right["png_path"],
            })
    if len(pairs) != 2520 or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("canonical pair enumeration mismatch")
    return pairs


def png_to_lpips_tensor(path: Path) -> Any:
    """Load one native RGB PNG as BCHW float32 in the frozen [-1, 1] range."""
    import torch

    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("LPIPS input must be an RGB 32x32 PNG")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div(127.5).sub(1.0)


def lpips_alexnet_metric(device: str | None = None) -> Callable[[Path, Path], float]:
    """Build the frozen evaluation-only LPIPS AlexNet v0.1 metric lazily."""
    import torch
    import lpips

    if version("lpips") != "0.1.4":
        raise RuntimeError("DR05 requires lpips==0.1.4")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metric = lpips.LPIPS(net="alex", version="0.1").to(resolved_device).eval()

    def measure(left: Path, right: Path) -> float:
        with torch.no_grad():
            value = metric(png_to_lpips_tensor(left).to(resolved_device), png_to_lpips_tensor(right).to(resolved_device))
        result = float(value.item())
        if not np.isfinite(result):
            raise ValueError("LPIPS distance is non-finite")
        return result
    return measure


def measure_pairs(pairs: Iterable[dict[str, Any]], source_root: Path, metric: Callable[[Path, Path], float]) -> list[dict[str, Any]]:
    """Measure supplied pairs; callers must separately authorize real LPIPS execution."""
    measured: list[dict[str, Any]] = []
    for pair in pairs:
        left, right = source_root / pair["png_a"], source_root / pair["png_b"]
        distance = float(metric(left, right))
        if not np.isfinite(distance):
            raise ValueError("non-finite LPIPS distance")
        # Reuses the audited sanity convention: mean absolute difference in [-1, 1].
        l1 = float(np.mean(np.abs(np.asarray(png_to_lpips_tensor(left)) - np.asarray(png_to_lpips_tensor(right)))))
        measured.append({**pair, "lpips": distance, "l1": l1})
    return measured


def aggregate_pairs(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not np.isfinite(float(row["lpips"])) or ("l1" in row and not np.isfinite(float(row["l1"]))):
            raise ValueError("non-finite diversity value")
        groups[str(row["prompt_id"])].append(row)
    prompt_rows = []
    for prompt_id, values in sorted(groups.items()):
        if len(values) != 10:
            raise ValueError("every prompt must have exactly ten unordered pairs")
        first = values[0]
        prompt_rows.append({"prompt_id": prompt_id, "panel_split": first["panel_split"], "ood_family": first.get("ood_family"), "control_role": first.get("control_role"), "pair_count": 10, "mean_lpips": float(np.mean([row["lpips"] for row in values])), "mean_l1": float(np.mean([row["l1"] for row in values]))})
    by_scope = lambda predicate: float(np.mean([row["mean_lpips"] for row in prompt_rows if predicate(row)]))
    aggregates = {"ID_prompt_macro_lpips": by_scope(lambda row: row["panel_split"] == "ID"), "OOD_prompt_macro_lpips": by_scope(lambda row: row["panel_split"] == "OOD"), "GLOBAL_prompt_macro_lpips": by_scope(lambda row: True)}
    family_values = []
    for family in OOD_FAMILIES:
        value = float(np.mean([row["mean_lpips"] for row in ood_family_prompt_rows(prompt_rows, family)]))
        aggregates[f"{family}_prompt_macro_lpips"] = value
        family_values.append(value)
    aggregates["OOD_equal_family_macro_lpips"] = float(np.mean(family_values))
    return prompt_rows, aggregates


def ood_family_prompt_rows(rows: Iterable[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    """Return only true OOD prompt records for one named OOD family."""
    if family not in OOD_FAMILIES:
        raise ValueError(f"unknown OOD family: {family}")
    return [
        row for row in rows
        if row.get("panel_split") == "OOD" and row.get("ood_family") == family
    ]


def bootstrap_prompt_scopes(prompt_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Frozen DR05 percentile bootstrap, resampling prompt-level values only."""
    rows = list(prompt_rows)
    scopes = {"ID": lambda row: row["panel_split"] == "ID", "OOD": lambda row: row["panel_split"] == "OOD"}
    scopes.update({family: lambda row, family=family: row.get("panel_split") == "OOD" and row.get("ood_family") == family for family in OOD_FAMILIES})
    rng = np.random.default_rng(202605)
    result: dict[str, Any] = {"seed": 202605, "resamples": 10000, "unit": "prompt", "method": "percentile", "confidence_level": 0.95, "scopes": {}}
    for name, predicate in scopes.items():
        values = np.asarray([row["mean_lpips"] for row in rows if predicate(row)], dtype=float)
        if not len(values):
            raise ValueError(f"empty bootstrap scope: {name}")
        draws = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
        result["scopes"][name] = {"estimate": float(np.mean(values)), "percentile_95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
    return result


def bootstrap_ood_family_scopes(prompt_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recompute only family CIs while preserving frozen RNG stream semantics."""
    rows = list(prompt_rows)
    rng = np.random.default_rng(202605)
    # The canonical stream first consumes the ID and OOD draws.  Their values
    # are intentionally not recomputed or returned by this repair-only path.
    for size in (sum(row["panel_split"] == "ID" for row in rows), sum(row["panel_split"] == "OOD" for row in rows)):
        rng.choice(np.zeros(size), size=(10000, size), replace=True)
    scopes: dict[str, dict[str, Any]] = {}
    for family in OOD_FAMILIES:
        values = np.asarray([row["mean_lpips"] for row in ood_family_prompt_rows(rows, family)], dtype=float)
        if not len(values):
            continue
        draws = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
        scopes[family] = {
            "estimate": float(np.mean(values)),
            "percentile_95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        }
    return scopes


def repair_diversity_summary(output_root: Path) -> dict[str, Any]:
    """Atomically repair only DR05 OOD-family fields from frozen prompt records."""
    prompt_rows = load_jsonl(output_root / "prompt_records.jsonl")
    summary_path = output_root / "diversity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(prompt_rows) != 252:
        raise ValueError("canonical prompt record count mismatch")
    if any(len(ood_family_prompt_rows(prompt_rows, family)) != 25 for family in OOD_FAMILIES):
        raise ValueError("canonical OOD family prompt geometry mismatch")

    family_values = {
        family: float(np.mean([row["mean_lpips"] for row in ood_family_prompt_rows(prompt_rows, family)]))
        for family in OOD_FAMILIES
    }
    family_bootstrap = bootstrap_ood_family_scopes(prompt_rows)
    aggregates = summary["aggregates"]
    for family, value in family_values.items():
        aggregates[f"{family}_prompt_macro_lpips"] = value
    aggregates["OOD_equal_family_macro_lpips"] = float(np.mean(list(family_values.values())))
    for family, scope in family_bootstrap.items():
        summary["bootstrap"]["scopes"][family] = scope

    temporary_path = summary_path.with_name(f".{summary_path.name}.repair.tmp")
    try:
        temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_path, summary_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return summary


def write_results(*, output_root: Path, source_root: Path, pair_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]], aggregates: dict[str, float], bootstrap: dict[str, Any]) -> None:
    reject_overlapping_output_root(source_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite diversity output root: {output_root}")
    output_root.mkdir(parents=True)
    for name, rows in (("pair_records.jsonl", pair_rows), ("prompt_records.jsonl", prompt_rows)):
        (output_root / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    payload = {"schema": "dr05_final_diversity_v1", "run_identity": {"protocol_version": "v1", "lpips_package_version": "0.1.4", "lpips_net": "alex", "lpips_version": "0.1", "native_resolution": [32, 32], "input_range": "[-1,1]", "source_dr04_root": str(source_root), "source_artifact_hashes": source_artifact_hashes(source_root)}, "population": {"prompts": 252, "images": 1260, "seeds_per_prompt": 5, "pairs_per_prompt": 10, "total_pairs": 2520, "ID_prompts": 152, "OOD_prompts": 100, "OOD_family_prompts": {family: 25 for family in OOD_FAMILIES}}, "aggregates": aggregates, "bootstrap": bootstrap, "interpretation": "Diversity values only; higher LPIPS is not automatically better."}
    (output_root / "diversity_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
