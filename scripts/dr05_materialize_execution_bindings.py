"""Materialize the initial partial DR05 execution-bindings artifact."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
OUTPUT = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json"
PROTOCOL_PATH = "data/processed/dr05_final_eval_v1/dr05_final_protocol.json"
PROTOCOL_SHA256 = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
ID_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_test.jsonl"
OOD_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "final_ood.jsonl"
ID_VAL_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_val.jsonl"
QUALITY_PREFIX_TARGETS = (5000, 10000, 25000, 50000)
QUALITY_MAIN_TARGET = 10000
QUALITY_EXTENDED_TARGET = 50000
LATENCY_IDENTIFIER_FIELD = "image_id"
LATENCY_CAPTION_FIELD = "caption"
TIMING_NAMESPACE = "latency_primary_b1"
TIMING_SEED_BASE = 602026
TIMING_WARMUP_RUNS = 3
TIMING_MEASURED_RUNS = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(bindings: dict[str, Any]) -> bytes:
    return (json.dumps(bindings, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {observed!r}")


def pending(reason: str) -> dict[str, str | None]:
    return {"reason": reason, "status": "pending", "value": None}


def hamilton_split_allocation(
    target_total: int, reference_id_count: int, reference_ood_count: int
) -> dict[str, int]:
    """Allocate a target exactly by Hamilton's largest-remainder method."""
    if target_total < 0:
        raise ValueError("target total must not be negative")
    if reference_id_count < 0 or reference_ood_count < 0:
        raise ValueError("reference counts must not be negative")
    reference_total = reference_id_count + reference_ood_count
    if reference_total == 0:
        raise ValueError("reference counts must not both be zero")

    counts = {"ID": reference_id_count, "OOD": reference_ood_count}
    quotas = {
        split: Fraction(target_total * count, reference_total)
        for split, count in counts.items()
    }
    allocations = {split: quota.numerator // quota.denominator for split, quota in quotas.items()}
    remaining = target_total - sum(allocations.values())
    for split in sorted(counts, key=lambda name: (- (quotas[name] - allocations[name]), ("ID", "OOD").index(name)))[:remaining]:
        allocations[split] += 1
    if sum(allocations.values()) != target_total:
        raise ValueError("Hamilton allocation does not sum to target total")
    return allocations


def quality_binding(value: int, target_total: int, reference_id_count: int, reference_ood_count: int) -> dict[str, Any]:
    return {
        "provenance": {
            "allocation_method": "hamilton_largest_remainder",
            "derived_from_bindings": ["R_ID_count", "R_OOD_count"],
            "reference_ID_count": reference_id_count,
            "reference_OOD_count": reference_ood_count,
            "target_total": target_total,
        },
        "status": "resolved",
        "value": value,
    }


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return []


def verify_protocol() -> None:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    require_equal("frozen protocol SHA256", sha256_file(PROTOCOL), PROTOCOL_SHA256)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    identity = protocol["protocol_identity"]
    require_equal("protocol name", identity["protocol_name"], "DR05-FINAL")
    require_equal("protocol version", identity["protocol_version"], "v1")
    require_equal("protocol status", identity["status"], "frozen")
    require_equal("quality generation seed base", protocol["quality"]["quality_rng"]["quality_generation_seed_base"], 502026)
    require_equal("KID subset seed", protocol["quality"]["operational_kid"]["kid_subset_seed"], 2026)
    require_equal("diversity bootstrap seed", protocol["diversity"]["uncertainty"]["seed"], 202605)


def metadata_record_count(path: Path, expected_split: str) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL record at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"metadata manifest is empty: {path}")
    if any(record.get("split_role") != expected_split for record in records):
        raise ValueError(f"split identity mismatch in {path}")
    ids = [record.get("image_id") for record in records]
    if any(not isinstance(image_id, str) or not image_id for image_id in ids):
        raise ValueError(f"missing image_id in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate image_id in {path}")
    return len(records)


def resolved_metadata_count(path: Path, expected_split: str) -> dict[str, Any]:
    return {
        "provenance": {
            "resolution_method": "metadata_record_count",
            "source_manifest_path": path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256_file(path),
        },
        "status": "resolved",
        "value": metadata_record_count(path, expected_split),
    }


def canonical_latency_prompt() -> tuple[str, str, dict[str, str]]:
    """Select the canonical latency prompt from ID_VAL metadata only."""
    if not ID_VAL_MANIFEST.is_file():
        raise FileNotFoundError(ID_VAL_MANIFEST)
    records = []
    for line_number, line in enumerate(ID_VAL_MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSONL record at {ID_VAL_MANIFEST}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at {ID_VAL_MANIFEST}:{line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError(f"metadata manifest is empty: {ID_VAL_MANIFEST}")
    if any(record.get("split_role") != "ID_VAL" for record in records):
        raise ValueError(f"split identity mismatch in {ID_VAL_MANIFEST}")
    identifiers = [record.get(LATENCY_IDENTIFIER_FIELD) for record in records]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"missing {LATENCY_IDENTIFIER_FIELD} in {ID_VAL_MANIFEST}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate {LATENCY_IDENTIFIER_FIELD} in {ID_VAL_MANIFEST}")
    if any(not isinstance(record.get(LATENCY_CAPTION_FIELD), str) or not record[LATENCY_CAPTION_FIELD] for record in records):
        raise ValueError(f"missing {LATENCY_CAPTION_FIELD} in {ID_VAL_MANIFEST}")
    selected = min(records, key=lambda record: record[LATENCY_IDENTIFIER_FIELD])
    provenance = {
        "caption_field": LATENCY_CAPTION_FIELD,
        "identifier_field": LATENCY_IDENTIFIER_FIELD,
        "selection_rule": "lexicographically_minimum_sample_id",
        "source_manifest_path": ID_VAL_MANIFEST.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(ID_VAL_MANIFEST),
    }
    return selected[LATENCY_IDENTIFIER_FIELD], selected[LATENCY_CAPTION_FIELD], provenance


def resolved_latency_binding(value: str, provenance: dict[str, str]) -> dict[str, Any]:
    return {"provenance": provenance, "status": "resolved", "value": value}


def timing_rng_binding() -> dict[str, Any]:
    """Build the pre-registered primary batch-1 latency RNG schedule."""
    total_runs = TIMING_WARMUP_RUNS + TIMING_MEASURED_RUNS
    seeds = list(range(TIMING_SEED_BASE, TIMING_SEED_BASE + total_runs))
    return {
        "provenance": {
            "decision": "D16",
            "purpose": "primary_batch1_latency",
            "selection": "preregistered_before_benchmark",
        },
        "status": "resolved",
        "value": {
            "dedicated_rng_required": True,
            "measured_runs": TIMING_MEASURED_RUNS,
            "measured_seeds": seeds[TIMING_WARMUP_RUNS:],
            "namespace": TIMING_NAMESPACE,
            "run_index_start": 0,
            "seed_base": TIMING_SEED_BASE,
            "seed_rule": "seed_base_plus_run_index",
            "shared_between_models": True,
            "total_runs": total_runs,
            "warmup_runs": TIMING_WARMUP_RUNS,
            "warmup_seeds": seeds[:TIMING_WARMUP_RUNS],
        },
    }


def build_bindings() -> dict[str, Any]:
    id_count = metadata_record_count(ID_MANIFEST, "ID_TEST")
    ood_count = metadata_record_count(OOD_MANIFEST, "FINAL_OOD")
    quality_prefix_allocations = {
        str(target): hamilton_split_allocation(target, id_count, ood_count)
        for target in QUALITY_PREFIX_TARGETS
    }
    latency_id, latency_caption, latency_provenance = canonical_latency_prompt()
    return {
        "bindings": {
            "R_ID_count": resolved_metadata_count(ID_MANIFEST, "ID_TEST"),
            "R_OOD_count": resolved_metadata_count(OOD_MANIFEST, "FINAL_OOD"),
            "benchmark_device": pending("requires_final_hardware_binding"),
            "gpu_name": pending("requires_final_hardware_binding"),
            "gpu_vram": pending("requires_final_hardware_binding"),
            "driver_version": pending("requires_final_hardware_binding"),
            "cuda_version": pending("requires_final_hardware_binding"),
            "latency_id_val_prompt_id": resolved_latency_binding(latency_id, latency_provenance),
            "latency_prompt_text": resolved_latency_binding(latency_caption, latency_provenance),
            "python_version": pending("requires_final_hardware_binding"),
            "pytorch_version": pending("requires_final_hardware_binding"),
            "quality_extended_ID_generated_count": quality_binding(
                quality_prefix_allocations[str(QUALITY_EXTENDED_TARGET)]["ID"],
                QUALITY_EXTENDED_TARGET, id_count, ood_count,
            ),
            "quality_extended_OOD_generated_count": quality_binding(
                quality_prefix_allocations[str(QUALITY_EXTENDED_TARGET)]["OOD"],
                QUALITY_EXTENDED_TARGET, id_count, ood_count,
            ),
            "quality_main_ID_generated_count": quality_binding(
                quality_prefix_allocations[str(QUALITY_MAIN_TARGET)]["ID"],
                QUALITY_MAIN_TARGET, id_count, ood_count,
            ),
            "quality_main_OOD_generated_count": quality_binding(
                quality_prefix_allocations[str(QUALITY_MAIN_TARGET)]["OOD"],
                QUALITY_MAIN_TARGET, id_count, ood_count,
            ),
            "timing_rng_seed_or_namespace": timing_rng_binding(),
            "unconditional_checkpoint_path": pending("requires_unconditional_baseline"),
            "unconditional_checkpoint_sha256": pending("requires_unconditional_baseline"),
        },
        "protocol": {
            "path": PROTOCOL_PATH,
            "protocol_name": "DR05-FINAL",
            "protocol_version": "v1",
            "sha256": PROTOCOL_SHA256,
        },
        "quality_prefix_allocations": quality_prefix_allocations,
        "status": "partial",
        "update_policy": {
            "all_mandatory_bindings_audited_before_final_frozen": True,
            "binding_resolution_requires_provenance": True,
            "pending_bindings_resolved_only_before_corresponding_experiment": True,
            "resolved_binding_must_not_silently_revert_or_change": True,
            "results_must_not_select_or_alter_bindings": True,
            "scientific_protocol_immutable": True,
        },
    }


def validate_bindings(bindings: dict[str, Any]) -> None:
    require_equal("artifact status", bindings["status"], "partial")
    require_equal("protocol path", bindings["protocol"]["path"], PROTOCOL_PATH)
    require_equal("protocol SHA256", bindings["protocol"]["sha256"], PROTOCOL_SHA256)
    require_equal("protocol name", bindings["protocol"]["protocol_name"], "DR05-FINAL")
    require_equal("protocol version", bindings["protocol"]["protocol_version"], "v1")
    expected_names = {
        "R_ID_count", "R_OOD_count", "quality_main_ID_generated_count",
        "quality_main_OOD_generated_count", "quality_extended_ID_generated_count",
        "quality_extended_OOD_generated_count", "latency_id_val_prompt_id",
        "latency_prompt_text", "timing_rng_seed_or_namespace", "benchmark_device",
        "gpu_name", "gpu_vram", "driver_version", "cuda_version", "pytorch_version",
        "python_version", "unconditional_checkpoint_path", "unconditional_checkpoint_sha256",
    }
    require_equal("required binding names", set(bindings["bindings"]), expected_names)
    for name in ("R_ID_count", "R_OOD_count"):
        binding = bindings["bindings"][name]
        require_equal(f"{name} status", binding["status"], "resolved")
        if not isinstance(binding["value"], int):
            raise ValueError(f"{name} value must be an integer")
        provenance = binding["provenance"]
        require_equal(f"{name} resolution method", provenance["resolution_method"], "metadata_record_count")
        if Path(provenance["source_manifest_path"]).is_absolute():
            raise ValueError(f"{name} provenance path is absolute")
    id_count = bindings["bindings"]["R_ID_count"]["value"]
    ood_count = bindings["bindings"]["R_OOD_count"]["value"]
    expected_allocations = {
        str(target): hamilton_split_allocation(target, id_count, ood_count)
        for target in QUALITY_PREFIX_TARGETS
    }
    require_equal("quality prefix allocations", bindings["quality_prefix_allocations"], expected_allocations)
    for target, allocation in expected_allocations.items():
        require_equal(f"quality allocation total for {target}", sum(allocation.values()), int(target))
    for split in ("ID", "OOD"):
        values = [expected_allocations[str(target)][split] for target in QUALITY_PREFIX_TARGETS]
        if values != sorted(values):
            raise ValueError(f"quality allocation for {split} is not monotonic")
    quality_bindings = {
        "quality_main_ID_generated_count": (QUALITY_MAIN_TARGET, "ID"),
        "quality_main_OOD_generated_count": (QUALITY_MAIN_TARGET, "OOD"),
        "quality_extended_ID_generated_count": (QUALITY_EXTENDED_TARGET, "ID"),
        "quality_extended_OOD_generated_count": (QUALITY_EXTENDED_TARGET, "OOD"),
    }
    for name, (target, split) in quality_bindings.items():
        binding = bindings["bindings"][name]
        require_equal(f"{name} status", binding["status"], "resolved")
        require_equal(f"{name} value", binding["value"], expected_allocations[str(target)][split])
        require_equal(
            f"{name} provenance", binding["provenance"],
            quality_binding(binding["value"], target, id_count, ood_count)["provenance"],
        )
    latency_id, latency_caption, latency_provenance = canonical_latency_prompt()
    for name, value in (
        ("latency_id_val_prompt_id", latency_id),
        ("latency_prompt_text", latency_caption),
    ):
        binding = bindings["bindings"][name]
        require_equal(f"{name} status", binding["status"], "resolved")
        require_equal(f"{name} value", binding["value"], value)
        require_equal(f"{name} provenance", binding["provenance"], latency_provenance)
    timing_binding = bindings["bindings"]["timing_rng_seed_or_namespace"]
    expected_timing_binding = timing_rng_binding()
    require_equal("timing RNG status", timing_binding["status"], "resolved")
    require_equal("timing RNG value", timing_binding["value"], expected_timing_binding["value"])
    require_equal("timing RNG provenance", timing_binding["provenance"], expected_timing_binding["provenance"])
    timing_value = timing_binding["value"]
    all_timing_seeds = timing_value["warmup_seeds"] + timing_value["measured_seeds"]
    require_equal("timing RNG total seed count", len(all_timing_seeds), timing_value["total_runs"])
    require_equal("timing RNG unique seeds", len(set(all_timing_seeds)), timing_value["total_runs"])
    require_equal("timing RNG contiguous sequence", all_timing_seeds, list(range(timing_value["seed_base"], timing_value["seed_base"] + timing_value["total_runs"])))
    for name, binding in bindings["bindings"].items():
        if name in {"R_ID_count", "R_OOD_count", *quality_bindings, "latency_id_val_prompt_id", "latency_prompt_text", "timing_rng_seed_or_namespace"}:
            continue
        require_equal(f"{name} status", binding["status"], "pending")
        require_equal(f"{name} value", binding["value"], None)
    if any(Path(value).is_absolute() for value in _strings(bindings)):
        raise ValueError("execution bindings contain an absolute path")


def materialize() -> tuple[Path, str]:
    verify_protocol()
    bindings = build_bindings()
    validate_bindings(bindings)
    content = serialize(bindings)
    OUTPUT.write_bytes(content)
    parsed = json.loads(OUTPUT.read_text(encoding="utf-8"))
    validate_bindings(parsed)
    if serialize(parsed) != content:
        raise ValueError("execution bindings serialization is not deterministic")
    return OUTPUT, sha256_file(OUTPUT)


if __name__ == "__main__":
    artifact, digest = materialize()
    print(json.dumps({"path": artifact.relative_to(ROOT).as_posix(), "sha256": digest}, sort_keys=True))
