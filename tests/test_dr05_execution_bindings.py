from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.dr05_materialize_execution_bindings import hamilton_split_allocation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dr05_materialize_execution_bindings.py"
ARTIFACT = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json"
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
PROTOCOL_SHA256 = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
ID_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_test.jsonl"
OOD_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "final_ood.jsonl"
ID_VAL_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_val.jsonl"
REQUIRED_BINDINGS = {
    "R_ID_count", "R_OOD_count", "quality_main_ID_generated_count",
    "quality_main_OOD_generated_count", "quality_extended_ID_generated_count",
    "quality_extended_OOD_generated_count", "latency_id_val_prompt_id",
    "latency_prompt_text", "timing_rng_seed_or_namespace", "benchmark_device",
    "gpu_name", "gpu_vram", "driver_version", "cuda_version", "pytorch_version",
    "python_version", "unconditional_checkpoint_path", "unconditional_checkpoint_sha256",
}
UNCONDITIONAL = ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt"
RUNTIME_BENCHMARK = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_runtime_benchmark.json"
HARDWARE_BINDINGS = ("benchmark_device", "gpu_name", "gpu_vram", "driver_version", "cuda_version", "pytorch_version", "python_version")
QUALITY_PREFIX_ALLOCATIONS = {
    "5000": {"ID": 2884, "OOD": 2116},
    "10000": {"ID": 5767, "OOD": 4233},
    "25000": {"ID": 14418, "OOD": 10582},
    "50000": {"ID": 28837, "OOD": 21163},
}
QUALITY_BINDINGS = {
    "quality_main_ID_generated_count": (10000, "ID"),
    "quality_main_OOD_generated_count": (10000, "OOD"),
    "quality_extended_ID_generated_count": (50000, "ID"),
    "quality_extended_OOD_generated_count": (50000, "OOD"),
}
LATENCY_BINDINGS = {"latency_id_val_prompt_id", "latency_prompt_text"}


def load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_hardware_bindings_match_d24(bindings: dict) -> None:
    evidence = json.loads(RUNTIME_BENCHMARK.read_text(encoding="utf-8"))
    environment = evidence["environment"]
    expected = {
        "benchmark_device": environment["benchmark_device"],
        "gpu_name": environment["gpu_name"],
        "gpu_vram": environment["gpu_vram_bytes"],
        "driver_version": environment["driver_version"],
        "cuda_version": environment["cuda_version"],
        "pytorch_version": environment["pytorch_version"],
        "python_version": environment["python_version"],
    }
    for name, value in expected.items():
        assert bindings[name] == {"status": "resolved", "value": value}


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in strings(nested)]
    return []


def metadata_record_count(path: Path, expected_split: str) -> int:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert all(record["split_role"] == expected_split for record in records)
    image_ids = [record["image_id"] for record in records]
    assert len(set(image_ids)) == len(image_ids)
    return len(records)


def test_execution_bindings_are_partial_and_link_frozen_protocol():
    assert ARTIFACT.is_file()
    bindings = load()
    assert bindings["status"] == "partial"
    assert bindings["protocol"] == {
        "path": "data/processed/dr05_final_eval_v1/dr05_final_protocol.json",
        "protocol_name": "DR05-FINAL",
        "protocol_version": "v1",
        "sha256": PROTOCOL_SHA256,
    }
    assert digest(PROTOCOL) == PROTOCOL_SHA256


def test_real_reference_counts_are_resolved_from_canonical_metadata():
    bindings = load()["bindings"]
    assert set(bindings) == REQUIRED_BINDINGS
    for name, manifest, split in (
        ("R_ID_count", ID_MANIFEST, "ID_TEST"),
        ("R_OOD_count", OOD_MANIFEST, "FINAL_OOD"),
    ):
        binding = bindings[name]
        assert binding["status"] == "resolved"
        assert binding["value"] == metadata_record_count(manifest, split)
        assert binding["provenance"] == {
            "resolution_method": "metadata_record_count",
            "source_manifest_path": manifest.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": digest(manifest),
        }
        assert not Path(binding["provenance"]["source_manifest_path"]).is_absolute()
    pending_bindings = {
        name: binding for name, binding in bindings.items()
        if name not in {"R_ID_count", "R_OOD_count", *QUALITY_BINDINGS, *LATENCY_BINDINGS, *HARDWARE_BINDINGS, "timing_rng_seed_or_namespace", "unconditional_checkpoint_path", "unconditional_checkpoint_sha256"}
    }
    assert all(binding["status"] == "pending" for binding in pending_bindings.values())
    assert all(binding["value"] is None for binding in pending_bindings.values())
    for name in ("unconditional_checkpoint_path", "unconditional_checkpoint_sha256"):
        assert bindings[name]["status"] == "resolved"
    assert bindings["unconditional_checkpoint_path"]["value"] == "artifacts/checkpoints/unconditional_main_ema_step20000.pt"
    assert bindings["unconditional_checkpoint_sha256"]["value"] == digest(UNCONDITIONAL)
    assert_hardware_bindings_match_d24(bindings)


def test_quality_prefixes_use_exact_hamilton_largest_remainder_allocation():
    bindings = load()["bindings"]
    assert bindings["R_ID_count"]["value"] == 466
    assert bindings["R_OOD_count"]["value"] == 342
    artifact_allocations = load()["quality_prefix_allocations"]
    assert artifact_allocations == QUALITY_PREFIX_ALLOCATIONS
    for target, expected in QUALITY_PREFIX_ALLOCATIONS.items():
        allocation = hamilton_split_allocation(int(target), 466, 342)
        assert allocation == expected
        assert sum(allocation.values()) == int(target)
        assert allocation["ID"] % 16 != 0 or allocation["OOD"] % 16 != 0
    assert [artifact_allocations[target]["ID"] for target in QUALITY_PREFIX_ALLOCATIONS] == [2884, 5767, 14418, 28837]
    assert [artifact_allocations[target]["OOD"] for target in QUALITY_PREFIX_ALLOCATIONS] == [2116, 4233, 10582, 21163]


def test_quality_bindings_are_resolved_with_hamilton_provenance():
    bindings = load()["bindings"]
    for name, (target, split) in QUALITY_BINDINGS.items():
        binding = bindings[name]
        assert binding["status"] == "resolved"
        assert binding["value"] == QUALITY_PREFIX_ALLOCATIONS[str(target)][split]
        assert binding["provenance"] == {
            "allocation_method": "hamilton_largest_remainder",
            "derived_from_bindings": ["R_ID_count", "R_OOD_count"],
            "reference_ID_count": 466,
            "reference_OOD_count": 342,
            "target_total": target,
        }


def test_latency_prompt_is_lexicographic_minimum_from_canonical_id_val_metadata():
    records = [json.loads(line) for line in ID_VAL_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert all(record["split_role"] == "ID_VAL" for record in records)
    identifiers = [record["image_id"] for record in records]
    assert all(isinstance(identifier, str) and identifier for identifier in identifiers)
    assert len(identifiers) == len(set(identifiers))
    assert all(isinstance(record.get("caption"), str) and record["caption"] for record in records)
    selected = min(records, key=lambda record: record["image_id"])
    provenance = {
        "caption_field": "caption",
        "identifier_field": "image_id",
        "selection_rule": "lexicographically_minimum_sample_id",
        "source_manifest_path": ID_VAL_MANIFEST.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": digest(ID_VAL_MANIFEST),
    }
    bindings = load()["bindings"]
    assert bindings["latency_id_val_prompt_id"] == {
        "provenance": provenance,
        "status": "resolved",
        "value": selected["image_id"],
    }
    assert bindings["latency_prompt_text"] == {
        "provenance": provenance,
        "status": "resolved",
        "value": selected["caption"],
    }
    assert_hardware_bindings_match_d24(bindings)


def test_primary_latency_timing_rng_is_preregistered_and_distinct_from_quality_rng():
    bindings = load()["bindings"]
    timing = bindings["timing_rng_seed_or_namespace"]
    expected_warmup_seeds = list(range(602026, 602029))
    expected_measured_seeds = list(range(602029, 602039))
    assert timing == {
        "provenance": {
            "decision": "D16",
            "purpose": "primary_batch1_latency",
            "selection": "preregistered_before_benchmark",
        },
        "status": "resolved",
        "value": {
            "dedicated_rng_required": True,
            "measured_runs": 10,
            "measured_seeds": expected_measured_seeds,
            "namespace": "latency_primary_b1",
            "run_index_start": 0,
            "seed_base": 602026,
            "seed_rule": "seed_base_plus_run_index",
            "shared_between_models": True,
            "total_runs": 13,
            "warmup_runs": 3,
            "warmup_seeds": expected_warmup_seeds,
        },
    }
    all_seeds = expected_warmup_seeds + expected_measured_seeds
    assert len(all_seeds) == len(set(all_seeds)) == 13
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["quality"]["quality_rng"]["quality_generation_seed_base"] == 502026
    assert protocol["quality"]["operational_kid"]["kid_subset_seed"] == 2026
    assert protocol["diversity"]["uncertainty"]["seed"] == 202605
    assert timing["value"]["seed_base"] != protocol["quality"]["quality_rng"]["quality_generation_seed_base"]
    assert_hardware_bindings_match_d24(bindings)


def test_policy_paths_and_serialization_are_deterministic():
    first_bytes = ARTIFACT.read_bytes()
    first_sha256 = digest(ARTIFACT)
    bindings = load()
    assert first_bytes == (json.dumps(bindings, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert first_sha256 == digest(ARTIFACT)
    assert bindings["update_policy"] == {
        "all_mandatory_bindings_audited_before_final_frozen": True,
        "binding_resolution_requires_provenance": True,
        "pending_bindings_resolved_only_before_corresponding_experiment": True,
        "resolved_binding_must_not_silently_revert_or_change": True,
        "results_must_not_select_or_alter_bindings": True,
        "scientific_protocol_immutable": True,
    }
    assert not any(Path(value).is_absolute() for value in strings(bindings))
