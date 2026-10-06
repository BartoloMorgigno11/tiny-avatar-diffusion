from __future__ import annotations

import json

import pytest

from tiny_avatar_diffusion.ca16_eval_manifests import (
    AMBIGUOUS_IMAGE_ID,
    DENIED_ROLES,
    assert_allowed_role,
    build,
    digest,
    load_records,
    select_robust_comp,
    select_robust_id,
    stable_record_key,
)


def test_stable_key_and_sha_ordering_are_deterministic():
    record = load_records("ID_VAL")[0]
    assert stable_record_key(record) == f"10k|ID_VAL|{record['image_id']}"
    assert digest("ID_SELECTION", stable_record_key(record)) == digest("ID_SELECTION", stable_record_key(record))


def test_denied_roles_fail_loudly():
    for role in DENIED_ROLES:
        with pytest.raises(ValueError):
            assert_allowed_role(role)


def test_robust_panels_follow_frozen_geometry():
    robust_id, classes = select_robust_id(load_records("ID_VAL"))
    robust_comp, eligibility = select_robust_comp(load_records("COMP_VAL"))
    assert len(robust_id) == 100 and len({stable_record_key(record) for record in robust_id}) == 100
    assert all(classes.values())
    assert eligibility == {"CV1": 65, "CV2": 139, "CV3": 52, "CV4": 81}
    assert len(robust_comp) == 100 and all(record["image_id"] != AMBIGUOUS_IMAGE_ID for record, _ in robust_comp)


def test_build_is_byte_identical_and_complete(tmp_path):
    first, second = build(tmp_path / "first"), build(tmp_path / "second")
    for name in ("robust_conditioning_manifest.json", "checkpoint_selection_manifest.json", "quality_guardrail_manifest.json", "manifest_audit.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert first["hashes"] == second["hashes"]
    audit = json.loads((tmp_path / "first" / "manifest_audit.json").read_text())
    assert audit["selection"] == {"ambiguous_absent": True, "comp_count": 237, "comp_disjoint_robust": True, "id_count": 256, "id_disjoint_robust": True}
    assert audit["quality"] == {"COMP_VAL": 338, "GLOBAL": 804, "ID_VAL": 466, "source_overlap_explicitly_allowed": True}
    assert audit["generation"]["selection_comp_seed_range"] == [2026, 2262]
    assert audit["generation"]["quality_seed_range"] == [2026, 2829]
