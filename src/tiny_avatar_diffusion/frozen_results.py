"""Read-only loader for the canonical DR04 and DR05 V1 evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_LATENCY_IDENTITIES = {
    ("conditional", 1),
    ("conditional", 16),
    ("unconditional", 1),
    ("unconditional", 16),
}
REQUIRED_LATENCY_STATISTICS = (
    "median_seconds_per_batch",
    "median_seconds_per_image",
    "mean_seconds_per_image",
    "std_seconds_per_image",
    "min_seconds_per_image",
    "max_seconds_per_image",
    "images_per_second",
)
REQUIRED_MEMORY_STATISTICS = (
    "peak_allocated_bytes",
    "peak_allocated_mib",
    "peak_reserved_bytes",
    "peak_reserved_mib",
)


def _read_json(root: Path, relative: str, expected_schema: str) -> dict[str, Any]:
    path = root / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Canonical evidence is missing: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Canonical evidence is malformed: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"Canonical evidence schema mismatch: {path}")
    return payload


def _finite_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Canonical evidence value must be finite: {label}")
    return value


def _path(payload: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    current: Any = payload
    try:
        for key in keys:
            current = current[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Canonical evidence field is missing: {label}") from error
    return current


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Canonical evidence field must be an object: {label}")
    return value


def _finite_fields(
    payload: dict[str, Any], fields: tuple[str, ...], label: str
) -> dict[str, int | float]:
    return {
        field: _finite_number(_path(payload, (field,), f"{label} {field}"), f"{label} {field}")
        for field in fields
    }


def _bootstrap_interval(
    dr04: dict[str, Any], scope: str, expected_estimate: float
) -> dict[str, int | float]:
    bootstrap = _mapping(_path(dr04, ("bootstrap",), "conditioning bootstrap"), "conditioning bootstrap")
    if bootstrap.get("unit") != "prompt" or bootstrap.get("all_five_seeds_clustered") is not True:
        raise ValueError("DR04 bootstrap protocol identity mismatch")
    scope_payload = _mapping(
        _path(bootstrap, ("scopes", scope), f"{scope} bootstrap"),
        f"{scope} bootstrap",
    )
    estimate = _finite_number(
        _path(scope_payload, ("estimate",), f"{scope} bootstrap estimate"),
        f"{scope} bootstrap estimate",
    )
    if not math.isclose(estimate, expected_estimate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{scope} bootstrap estimate does not match frozen MAA")
    interval = _path(scope_payload, ("percentile_95",), f"{scope} bootstrap interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{scope} bootstrap 95% interval must contain two bounds")
    lower = _finite_number(interval[0], f"{scope} bootstrap lower bound")
    upper = _finite_number(interval[1], f"{scope} bootstrap upper bound")
    if lower > estimate or estimate > upper:
        raise ValueError(f"{scope} bootstrap estimate must lie within its 95% interval")
    return {"lower": lower, "upper": upper}


def load_frozen_results(root: Path = ROOT) -> dict[str, Any]:
    """Load and normalize frozen headline evidence without mutating its sources."""
    dr04 = _read_json(
        root,
        "results/runs/dr04_final_conditioning_v1/final_conditioning_summary.json",
        "dr04_final_conditioning_summary",
    )
    quality = _read_json(
        root,
        "results/runs/dr05_final_quality_v1/final_quality_summary.json",
        "dr05_final_quality_main",
    )
    diversity = _read_json(
        root,
        "results/runs/dr05_final_diversity_v1/diversity_summary.json",
        "dr05_final_diversity_v1",
    )
    parameters = _read_json(
        root,
        "results/runs/dr05_final_efficiency_v1/parameter_count_summary.json",
        "tiny_avatar_diffusion.dr05_parameter_count_summary",
    )
    latency = _read_json(
        root,
        "results/runs/dr05_final_efficiency_v1/latency_summary.json",
        "dr05_final_latency_v1",
    )
    memory_payloads = {
        name: _read_json(
            root,
            f"results/runs/dr05_final_efficiency_v1/{name}.json",
            "dr05_final_sampling_memory_v1",
        )
        for name in ("conditional_b1", "unconditional_b1", "conditional_b16", "unconditional_b16")
    }
    if dr04.get("FINAL_SCIENTIFIC_EVALUATION") is not True or dr04.get("version") != 3:
        raise ValueError("DR04 evidence is not the frozen final version")
    if quality.get("FINAL_SCIENTIFIC_EVALUATION") is not True or quality.get("version") != 1:
        raise ValueError("DR05 quality evidence is not the frozen final version")
    if parameters.get("version") != 1:
        raise ValueError("DR05 parameter-count evidence version mismatch")

    id_maa = _finite_number(_path(dr04, ("primary_conditioning", "ID_MAA"), "ID MAA"), "ID MAA")
    ood_maa = _finite_number(_path(dr04, ("primary_conditioning", "OOD_MAA"), "OOD MAA"), "OOD MAA")
    macro_psr = _finite_number(
        _path(dr04, ("primary_compositional", "MacroPSR"), "MacroPSR"), "MacroPSR"
    )
    id_interval = _bootstrap_interval(dr04, "ID", id_maa)
    ood_interval = _bootstrap_interval(dr04, "OOD", ood_maa)
    quality_metrics = _mapping(
        _path(quality, ("state", "phase3", "metrics"), "quality metrics"),
        "quality metrics",
    )
    normalized_quality: dict[str, dict[str, int | float]] = {}
    normalized_quality_by_split: dict[
        str, dict[str, dict[str, int | float]]
    ] = {}
    for model in ("conditional", "unconditional"):
        result = _mapping(
            _path(quality_metrics, (f"{model}:GLOBAL", "result"), f"{model} quality"),
            f"{model} quality",
        )
        normalized_quality[model] = {
            "kid": _finite_number(_path(result, ("kid",), f"{model} KID"), f"{model} KID"),
            "fid": _finite_number(_path(result, ("fid",), f"{model} FID"), f"{model} FID"),
        }
        normalized_quality_by_split[model] = {}
        for split in ("ID", "OOD"):
            split_result = _mapping(
                _path(
                    quality_metrics,
                    (f"{model}:{split}", "result"),
                    f"{model} {split} quality",
                ),
                f"{model} {split} quality",
            )
            normalized_quality_by_split[model][split.lower()] = {
                "kid": _finite_number(
                    _path(split_result, ("kid",), f"{model} {split} KID"),
                    f"{model} {split} KID",
                ),
                "fid": _finite_number(
                    _path(split_result, ("fid",), f"{model} {split} FID"),
                    f"{model} {split} FID",
                ),
            }

    lpips = _finite_number(
        _path(diversity, ("aggregates", "GLOBAL_prompt_macro_lpips"), "LPIPS diversity"),
        "LPIPS diversity",
    )
    normalized_parameters = {
        "conditional": _finite_number(
            _path(parameters, ("conditional", "full_system", "total_parameters"), "conditional parameters"),
            "conditional parameters",
        ),
        "unconditional": _finite_number(
            _path(parameters, ("unconditional", "total_parameters"), "unconditional parameters"),
            "unconditional parameters",
        ),
    }

    conditions = latency.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 4:
        raise ValueError("Canonical latency evidence must contain four conditions")
    normalized_latency: dict[str, dict[str, dict[str, int | float]]] = {
        "conditional": {},
        "unconditional": {},
    }
    latency_identities: set[tuple[str, int]] = set()
    for condition in conditions:
        try:
            model = condition["model_type"]
            batch_size = condition["batch_size"]
            statistics = condition["statistics"]
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed canonical latency condition") from error
        identity = (model, batch_size)
        if identity not in EXPECTED_LATENCY_IDENTITIES:
            raise ValueError("Unexpected canonical latency condition identity")
        if identity in latency_identities:
            raise ValueError("Duplicate canonical latency condition identity")
        latency_identities.add(identity)
        statistics = _mapping(statistics, f"{model} batch {batch_size} latency statistics")
        normalized_latency[model][f"batch_{batch_size}"] = _finite_fields(
            statistics,
            REQUIRED_LATENCY_STATISTICS,
            f"{model} batch {batch_size} latency",
        )
    if latency_identities != EXPECTED_LATENCY_IDENTITIES:
        raise ValueError("Canonical latency evidence is missing required conditions")

    normalized_memory: dict[str, dict[str, dict[str, int | float]]] = {
        "conditional": {},
        "unconditional": {},
    }
    memory_identities: set[tuple[str, int]] = set()
    for name, payload in memory_payloads.items():
        model = payload.get("model_type")
        batch_size = payload.get("batch_size")
        identity = (model, batch_size)
        if identity not in EXPECTED_LATENCY_IDENTITIES:
            raise ValueError("Unexpected canonical memory condition identity")
        if identity in memory_identities:
            raise ValueError("Duplicate canonical memory condition identity")
        if name != f"{model}_b{batch_size}":
            raise ValueError("Canonical memory filename and condition identity disagree")
        memory_identities.add(identity)
        normalized_memory[model][f"batch_{batch_size}"] = _finite_fields(
            payload, REQUIRED_MEMORY_STATISTICS, name
        )
    if memory_identities != EXPECTED_LATENCY_IDENTITIES:
        raise ValueError("Canonical memory evidence is missing required conditions")

    return {
        "conditioning": {
            "id_maa": id_maa,
            "ood_maa": ood_maa,
            "absolute_maa_gap": abs(id_maa - ood_maa),
            "macro_psr": macro_psr,
            "id_maa_bootstrap_95": id_interval,
            "ood_maa_bootstrap_95": ood_interval,
        },
        "quality_global": normalized_quality,
        "quality_by_split": normalized_quality_by_split,
        "lpips_diversity_global_prompt_macro": lpips,
        "parameter_counts": normalized_parameters,
        "latency": normalized_latency,
        "memory": normalized_memory,
    }
