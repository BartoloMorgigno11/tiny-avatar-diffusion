from __future__ import annotations

import json

import pytest
import tiny_avatar_diffusion.dr04_evaluator_splits as dr04_evaluator_splits

from tiny_avatar_diffusion.dr04_evaluator_splits import (
    PROCESSED_ROOT,
    _allocate_unique_slots,
    _quota_by_replacement,
    build_assignments,
    load_100k_manifests,
    write_outputs,
)
from tiny_avatar_diffusion.dr04_support_audit import MetadataRow, matching_frozen_ood


def _definitions() -> list[dict]:
    return json.loads((PROCESSED_ROOT / "heldout_combinations.json").read_text(encoding="utf-8"))["final_ood_definitions"]


def test_quota_remainders_are_balanced_and_deterministic() -> None:
    first = _quota_by_replacement([0, 1, 2, 3, 4, 5, 6], "OOD1", "A_FIXED", 2026)
    second = _quota_by_replacement([0, 1, 2, 3, 4, 5, 6], "OOD1", "A_FIXED", 2026)
    assert first == second
    assert sum(first.values()) == 50
    assert max(first.values()) - min(first.values()) <= 1


def test_impossible_global_unique_allocation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Infeasible"):
        _allocate_unique_slots({"one": ["same"], "two": ["same"]}, 2026)


def test_real_100k_split_invariants_and_order_independence() -> None:
    rows, errors = load_100k_manifests(PROCESSED_ROOT / "100k")
    assert not errors and len(rows) == 100_000
    definitions = _definitions()
    assignments = build_assignments(rows, definitions)
    reversed_assignments = build_assignments(list(reversed(rows)), definitions)
    assert assignments == reversed_assignments
    assert len(assignments) == 100_000
    assert len({row["image_id"] for row in assignments}) == 100_000

    counts = {split: sum(row["split"] == split for row in assignments) for split in {row["split"] for row in assignments}}
    assert counts == {"REAL_GENERATOR_OOD_CEILING": 400, "REAL_ID_CEILING": 400, "EVAL_VAL": 5000, "EVAL_TRAIN": 94200}

    by_name = {definition["name"]: definition for definition in definitions}
    ood_rows = [row for row in assignments if row["split"] == "REAL_GENERATOR_OOD_CEILING"]
    assert {name: sum(row["source_ood"] == name for row in ood_rows) for name in by_name} == {name: 100 for name in by_name}
    for row in ood_rows:
        metadata = MetadataRow(row["image_id"], {key: row[key] for key in ("face_shape", "glasses", "hair_color", "face_color")})
        assert matching_frozen_ood(metadata, definitions) == [row["source_ood"]]

    id_rows = [row for row in assignments if row["split"] == "REAL_ID_CEILING"]
    assert len(id_rows) == len({row["image_id"] for row in id_rows}) == 400
    for name, definition in by_name.items():
        source_rows = [row for row in id_rows if row["source_ood"] == name]
        assert len(source_rows) == 100
        for direction in ("A_FIXED", "B_FIXED"):
            direction_rows = [row for row in source_rows if row["control_direction"] == direction]
            assert len(direction_rows) == 50
            quotas = [sum(row["replacement_value"] == value for row in direction_rows) for value in {row["replacement_value"] for row in direction_rows}]
            assert max(quotas) - min(quotas) <= 1
            if direction == "A_FIXED":
                changed, fixed, fixed_value, original = definition["attribute_b"], definition["attribute_a"], definition["value_a"], definition["value_b"]
            else:
                changed, fixed, fixed_value, original = definition["attribute_a"], definition["attribute_b"], definition["value_b"], definition["value_a"]
            eligible_values = {
                row.values[changed] for row in rows
                if row.values[fixed] == fixed_value and row.values[changed] != original and not matching_frozen_ood(row, definitions)
            }
            assert {row["replacement_value"] for row in direction_rows} == eligible_values
            for row in direction_rows:
                metadata = MetadataRow(row["image_id"], {key: row[key] for key in ("face_shape", "glasses", "hair_color", "face_color")})
                assert not matching_frozen_ood(metadata, definitions)
                if direction == "A_FIXED":
                    assert row[definition["attribute_a"]] == definition["value_a"]
                    assert row[definition["attribute_b"]] != definition["value_b"]
                else:
                    assert row[definition["attribute_a"]] != definition["value_a"]
                    assert row[definition["attribute_b"]] == definition["value_b"]

    validation_quota = [row for row in assignments if row["selection_stratum"] == "exclusive_ood_validation_quota"]
    assert len(validation_quota) == 200
    assert {name: sum(row["source_ood"] == name for row in validation_quota) for name in by_name} == {name: 50 for name in by_name}
    assert sum(row["selection_stratum"] == "natural_validation_remainder" for row in assignments) == 4800
    multi_by_split = {split: 0 for split in counts}
    for row in assignments:
        metadata = MetadataRow(row["image_id"], {key: row[key] for key in ("face_shape", "glasses", "hair_color", "face_color")})
        if len(matching_frozen_ood(metadata, definitions)) > 1:
            multi_by_split[row["split"]] += 1
    assert multi_by_split["REAL_GENERATOR_OOD_CEILING"] == multi_by_split["REAL_ID_CEILING"] == 0


def test_manifest_and_config_hashes_are_stable_across_repeated_writes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows, errors = load_100k_manifests(PROCESSED_ROOT / "100k")
    assert not errors
    definitions = _definitions()
    assignments = build_assignments(rows, definitions)
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    monkeypatch.setattr(dr04_evaluator_splits, "ROOT", repository_root)
    first = write_outputs(assignments, definitions, repository_root / "one", 2026)
    second = write_outputs(assignments, definitions, repository_root / "two", 2026)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["config_sha256"] == second["config_sha256"]
