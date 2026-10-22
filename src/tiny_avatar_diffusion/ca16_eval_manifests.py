"""Deterministically materialize the R2C CA16 development evaluation manifests.

This module consumes only frozen 10k `ID_VAL` and `COMP_VAL` text records.
It deliberately has no model, checkpoint, prediction, or generated-image dependency.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tiny_avatar_diffusion.release_paths import release_root

ROOT = release_root()
SOURCE_ROOT = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k"
SUMMARY_PATH = ROOT / "data" / "processed" / "dr01_v1" / "split_summary.json"
BASELINE_TAG = "v1.0.0"
BASELINE_COMMIT = "76a41e5742cf53f87eded1c49e9450bdd287469b"
ALLOWED_ROLES = frozenset(("ID_VAL", "COMP_VAL"))
DENIED_ROLES = frozenset(("TRAIN", "ID_TEST", "FINAL_OOD"))
ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
FAMILIES = {
    "CV1": ("face_shape", 0, "glasses", 0),
    "CV2": ("face_shape", 2, "hair_color", 3),
    "CV3": ("glasses", 7, "face_color", 2),
    "CV4": ("hair_color", 2, "face_color", 9),
}
AMBIGUOUS_IMAGE_ID = "cs10387983567686797037"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_record_key(record: dict[str, Any]) -> str:
    return "|".join(str(record[field]) for field in ("dataset_variant", "split_role", "image_id"))


def digest(namespace: str, key: str) -> str:
    return hashlib.sha256(f"2026|{namespace}|{key}".encode("utf-8")).hexdigest()


def assert_allowed_role(role: str) -> None:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"CA16 R2 evaluation manifests allow only {sorted(ALLOWED_ROLES)}; got {role}")


def load_records(role: str, *, source_root: Path = SOURCE_ROOT) -> list[dict[str, Any]]:
    assert_allowed_role(role)
    path = source_root / f"{role.lower()}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(record["split_role"] != role for record in records):
        raise ValueError(f"source manifest role mismatch for {role}")
    if len({stable_record_key(record) for record in records}) != len(records):
        raise ValueError(f"duplicate stable key in {role}")
    return records


def families_for(record: dict[str, Any]) -> list[str]:
    return [
        family for family, (left, left_value, right, right_value) in FAMILIES.items()
        if record[left] == left_value and record[right] == right_value
    ]


def public_record(record: dict[str, Any], *, family: str | None = None) -> dict[str, Any]:
    value = {
        "stable_record_key": stable_record_key(record),
        "dataset_variant": record["dataset_variant"],
        "split_role": record["split_role"],
        "image_id": record["image_id"],
        "source_image_path": record["source_image_path"],
        "caption": record["caption"],
        "ground_truth_attributes": {attribute: record[attribute] for attribute in ATTRIBUTES},
    }
    if family is not None:
        value["comp_family"] = family
    return value


def id_loss(selected: Iterable[dict[str, Any]], classes: dict[str, list[int]]) -> float:
    selected = list(selected)
    result = 0.0
    for attribute in ATTRIBUTES:
        values = classes[attribute]
        target = 100 / len(values)
        deficits = [max(target - sum(item[attribute] == value for item in selected), 0) / target for value in values]
        result += sum(deficit * deficit for deficit in deficits) / len(values)
    return result / len(ATTRIBUTES)


def select_robust_id(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    if len(records) != 466:
        raise ValueError(f"expected 466 ID_VAL records, got {len(records)}")
    classes = {attribute: sorted({record[attribute] for record in records}) for attribute in ATTRIBUTES}
    selected: list[dict[str, Any]] = []
    remaining = list(records)
    # Exact frozen formula: score(r) = L(S) - L(S union {r}), with deterministic digest tie-break.
    for _ in range(100):
        before = id_loss(selected, classes)
        scored = [(before - id_loss([*selected, record], classes), digest("ID_ROBUST", stable_record_key(record)), record) for record in remaining]
        maximum = max(score for score, _, _ in scored)
        _, _, chosen = min((item for item in scored if item[0] == maximum), key=lambda item: item[1])
        selected.append(chosen)
        remaining.remove(chosen)
    return selected, classes


def select_robust_comp(records: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str]], dict[str, int]]:
    if any(record["image_id"] == AMBIGUOUS_IMAGE_ID and sorted(families_for(record)) != ["CV1", "CV4"] for record in records):
        raise ValueError("unexpected ambiguous COMP record membership")
    selected: list[tuple[dict[str, Any], str]] = []
    eligibility: dict[str, int] = {}
    for family in FAMILIES:
        candidates = [record for record in records if record["image_id"] != AMBIGUOUS_IMAGE_ID and family in families_for(record)]
        eligibility[family] = len(candidates)
        selected.extend((record, family) for record in sorted(candidates, key=lambda record: digest(f"COMP_ROBUST|{family}", stable_record_key(record)))[:25])
    if eligibility != {"CV1": 65, "CV2": 139, "CV3": 52, "CV4": 81}:
        raise ValueError(f"unexpected COMP eligibility: {eligibility}")
    if len(selected) != 100 or len({stable_record_key(record) for record, _ in selected}) != 100:
        raise ValueError("robust COMP geometry/uniqueness failure")
    return selected, eligibility


def provenance() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"source_baseline_tag": BASELINE_TAG, "source_baseline_commit": BASELINE_COMMIT, "builder_git_commit": commit, "builder_working_tree_dirty": dirty}


def write_canonical(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _coverage(records: list[dict[str, Any]], classes: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {attribute: {str(value): sum(record[attribute] == value for record in records) for value in classes[attribute]} for attribute in ATTRIBUTES}


def build(output_root: Path) -> dict[str, Any]:
    """Build all R2C manifests from frozen source artifacts without model-dependent inputs."""
    id_records, comp_records = load_records("ID_VAL"), load_records("COMP_VAL")
    robust_id, classes = select_robust_id(id_records)
    robust_comp_pairs, eligibility = select_robust_comp(comp_records)
    robust_comp = [record for record, _ in robust_comp_pairs]
    robust_id_keys, robust_comp_keys = {stable_record_key(record) for record in robust_id}, {stable_record_key(record) for record in robust_comp}
    selection_id = sorted((record for record in id_records if stable_record_key(record) not in robust_id_keys), key=lambda record: digest("ID_SELECTION", stable_record_key(record)))[:256]
    selection_comp = [record for record in comp_records if stable_record_key(record) not in robust_comp_keys and record["image_id"] != AMBIGUOUS_IMAGE_ID]
    selection_comp = sorted(selection_comp, key=stable_record_key)
    if len(selection_id) != 256 or len(id_records) - len(robust_id) - len(selection_id) != 110:
        raise ValueError("ID selection geometry failure")
    if len(selection_comp) != 237:
        raise ValueError("COMP selection geometry failure")
    meta = provenance()
    source_hashes = {role: sha256_file(SOURCE_ROOT / f"{role.lower()}.jsonl") for role in ALLOWED_ROLES}
    robust_entries = [
        {**public_record(record), "panel": "robust_id", "generation_seeds": [2026, 2027, 2028, 2029, 2030]}
        for record in sorted(robust_id, key=stable_record_key)
    ] + [
        {**public_record(record, family=family), "panel": "robust_comp", "generation_seeds": [2026, 2027, 2028, 2029, 2030]}
        for record, family in sorted(robust_comp_pairs, key=lambda pair: (pair[1], stable_record_key(pair[0])))
    ]
    robust = {
        "schema": "tiny_avatar_diffusion.ca16_robust_conditioning_manifest", "version": "ca16_eval_v1", "purpose": "robust conditioning and matched diversity", "provenance": meta,
        "policies": {"stable_record_key": "dataset_variant|split_role|image_id", "id_selection": "exact_squared_normalized_deficit_greedy_v1", "comp_selection": "sha256_2026_comp_robust_v1", "generation_seeds": [2026, 2027, 2028, 2029, 2030]},
        "source_artifact_sha256": source_hashes, "counts": {"ID_VAL": 100, "COMP_VAL": 100, "COMP_family": {family: 25 for family in FAMILIES}, "future_generations_per_model": 1000},
        "diversity": {"lpips_package": "lpips==0.1.4", "network": "AlexNet", "lpips_version": "0.1", "rgb_resolution": "32x32", "input_range": "[-1,1]", "unordered_pairs_per_prompt": 10, "bootstrap_unit": "prompt", "bootstrap_resamples": 10000, "bootstrap_seed": 202605, "comp_aggregation": "prompt_mean_then_family_mean_then_equal_family_macro"}, "entries": robust_entries,
    }
    selection = {
        "schema": "tiny_avatar_diffusion.ca16_checkpoint_selection_manifest", "version": "ca16_eval_v1", "purpose": "checkpoint selection and COMP KID fourth tie-break", "provenance": meta,
        "policies": {"stable_record_key": "dataset_variant|split_role|image_id", "id_selection": "sha256_2026_id_selection_v1", "comp_selection": "all_eligible_after_robust_and_ambiguous_exclusion", "comp_generation_seed": "2026+canonical_manifest_index"}, "source_artifact_sha256": source_hashes,
        "counts": {"ID_VAL": 256, "COMP_VAL": 237, "ID_VAL_unselected": 110},
        "id_entries": [public_record(record) for record in selection_id],
        "comp_entries": [{**public_record(record), "canonical_manifest_index": index, "generation_seed": 2026 + index, "comp_families": families_for(record)} for index, record in enumerate(selection_comp)],
    }
    quality_records = sorted([*id_records, *comp_records], key=lambda record: (record["split_role"], stable_record_key(record)))
    quality = {
        "schema": "tiny_avatar_diffusion.ca16_quality_guardrail_manifest", "version": "ca16_eval_v1", "purpose": "prospective distribution-preserving quality guardrail", "provenance": meta,
        "policies": {"stable_record_key": "dataset_variant|split_role|image_id", "order": "ascending_split_role_then_stable_record_key", "generation_seed": "2026+canonical_manifest_index", "source_overlap": "allowed_with_separate_prospective_generation_manifest"}, "source_artifact_sha256": source_hashes,
        "counts": {"ID_VAL": 466, "COMP_VAL": 338, "GLOBAL": 804},
        "entries": [{**public_record(record), "comp_families": families_for(record) if record["split_role"] == "COMP_VAL" else [], "canonical_manifest_index": index, "generation_seed": 2026 + index} for index, record in enumerate(quality_records)],
    }
    robust_path, selection_path, quality_path = (output_root / name for name in ("robust_conditioning_manifest.json", "checkpoint_selection_manifest.json", "quality_guardrail_manifest.json"))
    hashes = {"robust_conditioning_manifest_sha256": write_canonical(robust_path, robust), "checkpoint_selection_manifest_sha256": write_canonical(selection_path, selection), "quality_guardrail_manifest_sha256": write_canonical(quality_path, quality)}
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["datasets"]["10k"]["final_split_counts"]
    audit = {
        "schema": "tiny_avatar_diffusion.ca16_manifest_audit", "version": "ca16_eval_v1", "provenance": meta, "manifest_sha256": hashes, "source_artifact_sha256": source_hashes,
        "split_counts_from_frozen_summary": summary, "robust_id": {"count": 100, "unique_stable_keys": len(robust_id_keys), "classes": {attribute: len(values) for attribute, values in classes.items()}, "class_coverage": _coverage(robust_id, classes)},
        "robust_comp": {"eligible_after_ambiguous_exclusion": eligibility, "selected_per_family": dict(Counter(family for _, family in robust_comp_pairs)), "unique_stable_keys": len(robust_comp_keys), "ambiguous_image_id": AMBIGUOUS_IMAGE_ID, "ambiguous_absent": all(record["image_id"] != AMBIGUOUS_IMAGE_ID for record in robust_comp)},
        "selection": {"id_count": len(selection_id), "id_disjoint_robust": not robust_id_keys.intersection(stable_record_key(record) for record in selection_id), "comp_count": len(selection_comp), "comp_disjoint_robust": not robust_comp_keys.intersection(stable_record_key(record) for record in selection_comp), "ambiguous_absent": all(record["image_id"] != AMBIGUOUS_IMAGE_ID for record in selection_comp)},
        "generation": {"selection_comp_seed_range": [2026, 2262], "robust_prompts": 200, "robust_seeds_per_prompt": 5, "robust_future_generations_per_model": 1000, "quality_seed_range": [2026, 2829]},
        "quality": {"ID_VAL": 466, "COMP_VAL": 338, "GLOBAL": 804, "source_overlap_explicitly_allowed": True}, "safety": {"allowed_roles": sorted(ALLOWED_ROLES), "denied_roles": sorted(DENIED_ROLES), "records_from_denied_roles": 0},
    }
    audit_path = output_root / "manifest_audit.json"
    hashes["manifest_audit_sha256"] = write_canonical(audit_path, audit)
    return {"paths": {"robust": robust_path, "selection": selection_path, "quality": quality_path, "audit": audit_path}, "hashes": hashes, "audit": audit}
