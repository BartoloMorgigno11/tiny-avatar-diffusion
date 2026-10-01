from __future__ import annotations

import json

from tiny_avatar_diffusion.dr04_support_audit import MetadataRow, enumerate_matched_controls, load_100k_manifests, marginal_support, matching_frozen_ood, ood_overlap_audit, pair_count


def test_counts_and_matched_control_enumeration() -> None:
    rows = [
        MetadataRow("one", {"face_shape": 2, "glasses": 3, "hair_color": 0, "face_color": 0}),
        MetadataRow("two", {"face_shape": 2, "glasses": 4, "hair_color": 0, "face_color": 0}),
        MetadataRow("three", {"face_shape": 1, "glasses": 3, "hair_color": 0, "face_color": 0}),
    ]
    assert marginal_support(rows)["glasses"] == {3: 2, 4: 1}
    assert pair_count(rows, "face_shape", 2, "glasses", 3) == 1
    definition = {"name": "OOD1", "attribute_a": "face_shape", "value_a": 2, "attribute_b": "glasses", "value_b": 3}
    ood4 = {"name": "OOD4", "attribute_a": "hair_color", "value_a": 0, "attribute_b": "face_color", "value_b": 0}
    controls = enumerate_matched_controls(rows, definition, [definition, ood4])
    assert {(row["changed_attribute"], row["changed_to_value"], row["unchanged_attribute"], row["unchanged_value"], row["raw_candidate_support"]) for row in controls} == {("glasses", 4, "face_shape", 2, 1), ("face_shape", 1, "glasses", 3, 1)}
    assert all(row["globally_ordinary_support"] == 0 for row in controls)
    assert all(json.loads(row["cross_ood_exclusions_json"]) == {"OOD4": 1} for row in controls)
    assert matching_frozen_ood(rows[0], [definition]) == ["OOD1"]


def test_manifest_loader_reports_malformed_missing_and_duplicates(tmp_path) -> None:
    names = ("train.jsonl", "id_val.jsonl", "id_test.jsonl", "comp_val.jsonl", "final_ood.jsonl")
    valid = {"image_id": "ok", "face_shape": 1, "glasses": 2, "hair_color": 3, "face_color": 4}
    (tmp_path / names[0]).write_text(json.dumps(valid) + "\n" + json.dumps({"image_id": "missing"}) + "\n", encoding="utf-8")
    (tmp_path / names[1]).write_text(json.dumps(valid) + "\n", encoding="utf-8")
    for name in names[2:]:
        (tmp_path / name).write_text("", encoding="utf-8")
    rows, errors = load_100k_manifests(tmp_path)
    assert [row.image_id for row in rows] == ["ok"]
    assert len(errors) == 2


def test_ood_overlap_partition_includes_zero_one_two_and_three_matches() -> None:
    definitions = [
        {"name": "OOD1", "attribute_a": "face_shape", "value_a": 2, "attribute_b": "glasses", "value_b": 3},
        {"name": "OOD2", "attribute_a": "face_shape", "value_a": 4, "attribute_b": "hair_color", "value_b": 6},
        {"name": "OOD3", "attribute_a": "glasses", "value_a": 10, "attribute_b": "face_color", "value_b": 10},
        {"name": "OOD4", "attribute_a": "hair_color", "value_a": 0, "attribute_b": "face_color", "value_b": 0},
    ]
    rows = [
        MetadataRow("ordinary", {"face_shape": 0, "glasses": 0, "hair_color": 1, "face_color": 1}),
        MetadataRow("one", {"face_shape": 2, "glasses": 3, "hair_color": 1, "face_color": 1}),
        MetadataRow("two", {"face_shape": 4, "glasses": 10, "hair_color": 6, "face_color": 10}),
        MetadataRow("three", {"face_shape": 2, "glasses": 3, "hair_color": 0, "face_color": 0}),
    ]
    audit = ood_overlap_audit(rows, definitions)
    assert audit["pattern_counts"] == {"0000": 1, "0110": 1, "1000": 1, "1001": 1}
    assert audit["exclusive_support"] == {"OOD1": 1, "OOD2": 0, "OOD3": 0, "OOD4": 0}
    assert audit["pairwise_intersections"]["OOD2__OOD3"] == 1
    assert audit["category_counts"] == {"0": 1, "1": 1, "2": 2, "3": 0, "4": 0}
    assert sum(audit["category_counts"].values()) == len(rows)


def test_ood_overlap_partition_counts_three_or_more_matches() -> None:
    definitions = [
        {"name": "OOD1", "attribute_a": "face_shape", "value_a": 1, "attribute_b": "glasses", "value_b": 1},
        {"name": "OOD2", "attribute_a": "face_shape", "value_a": 1, "attribute_b": "hair_color", "value_b": 1},
        {"name": "OOD3", "attribute_a": "face_shape", "value_a": 1, "attribute_b": "face_color", "value_b": 1},
        {"name": "OOD4", "attribute_a": "glasses", "value_a": 2, "attribute_b": "hair_color", "value_b": 2},
    ]
    row = MetadataRow("three", {"face_shape": 1, "glasses": 1, "hair_color": 1, "face_color": 1})
    audit = ood_overlap_audit([row], definitions)
    assert audit["pattern_counts"] == {"1110": 1}
    assert audit["category_counts"] == {"0": 0, "1": 0, "2": 0, "3": 1, "4": 0}
