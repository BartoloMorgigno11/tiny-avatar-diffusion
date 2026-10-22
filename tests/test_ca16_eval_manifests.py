from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

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


CA16_ROOT = Path(__file__).parents[1] / "data" / "processed" / "ca16_eval_v1"
EXPECTED_SHA256 = {
    "robust_conditioning_manifest.json": "e3319f65f67c19ae8b458f7450ff8e6b86e5913e0fd542e9c0c3009802451aec",
    "checkpoint_selection_manifest.json": "192191235af7992b8c332019b7e948414ba0b87c5f8df8d70f044b74e85e3d40",
    "quality_guardrail_manifest.json": "e34db5251d8949a5fb37a905f6e9e4791602047f3153a9a9908d14d9cd85e84f",
    "manifest_audit.json": "b693521cc5e1430b6f5d44acd3bca4bae11eb67746fc06e6e43b22f21b69c35f",
}


def _git_metadata_available() -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    ).returncode == 0


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


def test_packaged_ca16_manifests_are_intact_without_git():
    for name, expected in EXPECTED_SHA256.items():
        assert hashlib.sha256((CA16_ROOT / name).read_bytes()).hexdigest() == expected
    audit = json.loads((CA16_ROOT / "manifest_audit.json").read_text())
    assert audit["selection"] == {"ambiguous_absent": True, "comp_count": 237, "comp_disjoint_robust": True, "id_count": 256, "id_disjoint_robust": True}
    assert audit["quality"] == {"COMP_VAL": 338, "GLOBAL": 804, "ID_VAL": 466, "source_overlap_explicitly_allowed": True}


def test_build_is_byte_identical_and_complete(tmp_path):
    if not _git_metadata_available():
        pytest.skip("CA16 rebuild provenance requires Git metadata; packaged-manifest integrity is covered separately")
    first, second = build(tmp_path / "first"), build(tmp_path / "second")
    for name in ("robust_conditioning_manifest.json", "checkpoint_selection_manifest.json", "quality_guardrail_manifest.json", "manifest_audit.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert first["hashes"] == second["hashes"]
    audit = json.loads((tmp_path / "first" / "manifest_audit.json").read_text())
    assert audit["selection"] == {"ambiguous_absent": True, "comp_count": 237, "comp_disjoint_robust": True, "id_count": 256, "id_disjoint_robust": True}
    assert audit["quality"] == {"COMP_VAL": 338, "GLOBAL": 804, "ID_VAL": 466, "source_overlap_explicitly_allowed": True}
    assert audit["generation"]["selection_comp_seed_range"] == [2026, 2262]
    assert audit["generation"]["quality_seed_range"] == [2026, 2829]
