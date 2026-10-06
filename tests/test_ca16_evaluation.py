from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import tiny_avatar_diffusion.ca16_evaluation as evaluation_module
from tiny_avatar_diffusion.ca16_evaluation import (
    ARCHITECTURE_ID,
    ATTRIBUTES,
    CANDIDATE_MILESTONES,
    CANONICAL_ATTRIBUTE_ORDER,
    CLASS_COUNTS,
    EVALUATOR_CHECKPOINT_RELATIVE,
    EVALUATOR_CHECKPOINT_SHA256,
    EVALUATOR_FREEZE,
    EVALUATOR_FREEZE_SHA256,
    PREPROCESSING_ID,
    QUALITY_MANIFEST,
    ROBUST_MANIFEST,
    SELECTION_MANIFEST,
    SEMANTIC_MAPPING,
    SampleArtifact,
    SelectionCandidate,
    attribute_metrics,
    build_measured_selection_result,
    build_robust_conditioning_plan,
    compute_comp_kid_from_artifacts,
    family_macro_metrics,
    load_checkpoint_selection_manifest,
    load_ema_checkpoint,
    load_evaluator_sample,
    load_robust_conditioning_manifest,
    rank_candidates,
    select_final_candidate,
    stable_sample_identity,
    validate_candidate_milestones,
    validate_sample_artifacts,
)
from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec
from tiny_avatar_diffusion.conditional_main_training import CA16_CHECKPOINT_SCHEMA
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet


def test_selection_manifest_frozen_panel_and_seeds() -> None:
    manifest = load_checkpoint_selection_manifest()
    assert len(manifest["id_entries"]) == 256
    assert len(manifest["comp_entries"]) == 237
    assert manifest["id_entries"][0]["generation_seed"] == 2026
    assert manifest["id_entries"][-1]["generation_seed"] == 2281
    assert [item["generation_seed"] for item in manifest["comp_entries"]] == list(range(2026, 2263))


def test_forbidden_manifest_roles_are_rejected(tmp_path: Path) -> None:
    manifest = json.loads(SELECTION_MANIFEST.read_text())
    manifest["comp_entries"][0]["split_role"] = "FINAL_OOD"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        from tiny_avatar_diffusion.ca16_evaluation import _validate_records
        _validate_records(manifest["comp_entries"], source="test")


def test_robust_panel_shape_and_selection_separation() -> None:
    robust = load_robust_conditioning_manifest()
    selection = load_checkpoint_selection_manifest()
    selection_keys = {entry["stable_record_key"] for entry in selection["id_entries"] + selection["comp_entries"]}
    robust_keys = {entry["stable_record_key"] for entry in robust["entries"]}
    assert not selection_keys & robust_keys
    assert build_robust_conditioning_plan()["generations_per_model"] == 1000


def test_exact_candidate_milestones() -> None:
    candidates = [
        SelectionCandidate("r", "ca8", 2026, step, .5, .2, .5)
        for step in CANDIDATE_MILESTONES
    ]
    validate_candidate_milestones(candidates)
    with pytest.raises(ValueError):
        validate_candidate_milestones(candidates[:-1])
    with pytest.raises(ValueError):
        validate_candidate_milestones([*candidates[:-1], candidates[0]])


def _record(family: str, index: int) -> dict[str, object]:
    return {"comp_families": [family], "ground_truth_attributes": {name: index for name in ("face_shape", "glasses", "hair_color", "face_color")}}


def _prediction(value: int) -> dict[str, int]:
    return {name: value for name in ("face_shape", "glasses", "hair_color", "face_color")}


def test_metrics_and_psr_require_all_attributes() -> None:
    records = [_record("CV1", 0), _record("CV1", 0)]
    metrics = attribute_metrics(records, [_prediction(0), {"face_shape": 0, "glasses": 1, "hair_color": 0, "face_color": 0}])
    assert metrics["maa"] == pytest.approx(7 / 8)
    assert metrics["psr"] == pytest.approx(.5)
    assert metrics["per_attribute_accuracy"]["glasses"] == pytest.approx(.5)
    with pytest.raises(ValueError):
        attribute_metrics(records, [_prediction(0)])


def test_family_macro_is_equal_weighted_not_pooled() -> None:
    records, predictions = [], []
    for family, count, correct in (("CV1", 1, 1), ("CV2", 3, 0), ("CV3", 1, 1), ("CV4", 1, 1)):
        for index in range(count):
            records.append(_record(family, index))
            predictions.append(_prediction(index if correct else index + 1))
    result = family_macro_metrics(records, predictions)
    assert result["family_macro_maa"] == pytest.approx(.75)
    assert result["family_macro_maa"] != pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        family_macro_metrics(records, predictions[:-1])


def test_ranking_is_lexicographic_and_kid_is_fourth() -> None:
    base = [SelectionCandidate("r", "ca8", 2026, step, .5, .5, .5, comp_kid=.1) for step in CANDIDATE_MILESTONES]
    ranked = rank_candidates(base)
    assert ranked[0].milestone_step == 5000
    better_primary = base[1].__class__("r", "ca8", 2026, 10000, .6, .1, .1, comp_kid=.0)
    assert rank_candidates([better_primary, *[item for item in base if item.milestone_step != 10000]])[0] == better_primary


def test_kid_never_overrides_adherence_and_epsilon_mse_is_ignored() -> None:
    candidates = [
        SelectionCandidate("r", "ca8", 2026, 5000, .8, .8, .8, comp_kid=10, epsilon_mse=99),
        SelectionCandidate("r", "ca8", 2026, 10000, .7, .7, .7, comp_kid=.01, epsilon_mse=.001),
        SelectionCandidate("r", "ca8", 2026, 15000, .6, .6, .6, comp_kid=.01),
        SelectionCandidate("r", "ca8", 2026, 20000, .5, .5, .5, comp_kid=.01),
    ]
    assert rank_candidates(candidates)[0].milestone_step == 5000


def test_stable_identity_and_robust_seed_contract() -> None:
    record = load_checkpoint_selection_manifest()["comp_entries"][0]
    identity = stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8, 2026), 5000, record)
    assert identity.weight_set == "EMA"
    assert "ca16_ablation_ca8_seed2026" in identity.as_key()
    assert str(record["generation_seed"]) in identity.as_key()


def test_id_seed_is_canonical_and_not_record_supplied() -> None:
    record = dict(load_checkpoint_selection_manifest()["id_entries"][3])
    record["generation_seed"] = 9999
    identity = stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8, 2026), 5000, record)
    assert identity.generation_seed == 2029


def test_selection_identity_rejects_non_panel_records() -> None:
    with pytest.raises(ValueError):
        stable_sample_identity(
            CA16RunSpec(ArchitectureArm.CA8, 2026),
            5000,
            {
                "split_role": "ID_VAL",
                "stable_record_key": "not-in-panel",
                "generation_seed": 2026,
            },
        )


def test_sample_identity_collision_dimensions() -> None:
    record = load_checkpoint_selection_manifest()["id_entries"][0]
    base = stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8, 2026), 5000, record)
    variants = {
        base.as_key(),
        stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8_CA16, 2026), 5000, record).as_key(),
        stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8, 2027), 5000, record).as_key(),
        stable_sample_identity(CA16RunSpec(ArchitectureArm.CA8, 2026), 10000, record).as_key(),
        base.as_key().replace("ID_VAL", "COMP_VAL"),
        base.as_key().replace(base.record_id, "other"),
        base.as_key().replace(str(base.generation_seed), "9999"),
    }
    assert len(variants) == 7


def test_metrics_explicit_four_of_four_and_three_of_four() -> None:
    record = _record("CV1", 0)
    correct = _prediction(0)
    three_correct = {**correct, "face_color": 1}
    assert attribute_metrics([record], [correct])["psr"] == 1
    metrics = attribute_metrics([record], [three_correct])
    assert metrics["psr"] == 0
    assert metrics["maa"] == pytest.approx(.75)


def test_family_macro_rejects_missing_family() -> None:
    records = [_record(family, 0) for family in ("CV1", "CV2", "CV3")]
    with pytest.raises(ValueError):
        family_macro_metrics(records, [_prediction(0)] * 3)


def test_ranking_each_hierarchy_level_and_precision() -> None:
    def candidates(values: list[tuple[float, float, float, float]]) -> list[SelectionCandidate]:
        return [
            SelectionCandidate("r", "ca8", 2026, step, *value)
            for step, value in zip(CANDIDATE_MILESTONES, values)
        ]

    assert select_final_candidate(candidates([(0.9, .1, .1, .9), (.8, .9, .9, .1), (.7, .8, .8, .1), (.6, .7, .7, .1)])).milestone_step == 5000
    assert select_final_candidate(candidates([(.9, .9, .1, .9), (.9, .8, .9, .1), (.8, .7, .8, .1), (.7, .6, .7, .1)])).milestone_step == 5000
    assert select_final_candidate(candidates([(.9, .9, .9, .9), (.9, .9, .8, .1), (.8, .8, .8, .1), (.7, .7, .7, .1)])).milestone_step == 5000
    assert select_final_candidate(candidates([(.9, .9, .9, .2), (.9, .9, .9, .1), (.8, .8, .8, .1), (.7, .7, .7, .1)])).milestone_step == 10000
    assert select_final_candidate(candidates([(.9, .9, .9, .1), (.9, .9, .9, .1), (.8, .8, .8, .1), (.7, .7, .7, .1)])).milestone_step == 5000
    assert select_final_candidate(candidates([(.9000001, .1, .1, .9), (.9, .9, .9, .1), (.8, .8, .8, .1), (.7, .7, .7, .1)])).milestone_step == 5000


def test_kid_is_lazy_and_only_called_for_criterion_ties() -> None:
    calls: list[int] = []
    def kid(candidate: SelectionCandidate) -> float:
        calls.append(candidate.milestone_step)
        return float(candidate.milestone_step)

    unique = [
        SelectionCandidate("r", "ca8", 2026, step, 1.0 if step == 5000 else 0.5, .1, .1)
        for step in CANDIDATE_MILESTONES
    ]
    assert select_final_candidate(unique, kid=kid).milestone_step == 5000
    assert calls == []

    tied = [
        SelectionCandidate("r", "ca8", 2026, 5000, .9, .9, .9),
        SelectionCandidate("r", "ca8", 2026, 10000, .9, .9, .9),
        SelectionCandidate("r", "ca8", 2026, 15000, .8, .8, .8),
        SelectionCandidate("r", "ca8", 2026, 20000, .7, .7, .7),
    ]
    select_final_candidate(tied, kid=kid)
    assert calls == [5000, 10000]


def _artifact_set(tmp_path: Path, spec: CA16RunSpec, step: int, role: str) -> list[SampleArtifact]:
    manifest = load_checkpoint_selection_manifest()
    records = manifest["id_entries"] if role == "ID_VAL" else manifest["comp_entries"]
    directory = tmp_path / role
    directory.mkdir()
    artifacts = []
    for index, record in enumerate(records):
        path = directory / f"{index}.png"
        path.write_bytes(b"fixture")
        identity = stable_sample_identity(spec, step, record)
        truth = {name: 0 for name in ("face_shape", "glasses", "hair_color", "face_color")}
        artifacts.append(SampleArtifact(identity, str(path), truth, truth, {name: True for name in truth}))
    return artifacts


def test_artifact_completeness_and_identity(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    artifacts = _artifact_set(tmp_path, spec, 5000, "ID_VAL")
    validate_sample_artifacts(artifacts, spec, 5000, role="ID_VAL")
    with pytest.raises(ValueError):
        validate_sample_artifacts(artifacts[:-1], spec, 5000, role="ID_VAL")
    wrong = list(artifacts)
    wrong[0] = SampleArtifact(
        wrong[0].identity.__class__(
            **{**wrong[0].identity.__dict__, "weight_set": "RAW"}
        ),
        wrong[0].artifact_path, wrong[0].ground_truth, wrong[0].predictions, wrong[0].correctness,
    )
    with pytest.raises(ValueError):
        validate_sample_artifacts(wrong, spec, 5000, role="ID_VAL")


def _checkpoint_fixture(tmp_path: Path, spec: CA16RunSpec, *, identity_update: dict[str, object] | None = None) -> Path:
    text = TextConditioningEncoder()
    unet = ConditionalTinyUNet(
        use_decoder16_cross_attention=spec.use_decoder16_cross_attention,
        dropout=0.0,
    )
    raw_text = {key: torch.zeros_like(value) for key, value in text.state_dict().items()}
    raw_unet = {key: torch.zeros_like(value) for key, value in unet.state_dict().items()}
    ema_text = {key: torch.ones_like(value) for key, value in text.state_dict().items()}
    ema_unet = {key: torch.ones_like(value) for key, value in unet.state_dict().items()}
    identity = spec.provenance(execution_git_sha="fixture-sha", branch="fixture-branch")
    identity.update(identity_update or {})
    checkpoint = {
        "schema": CA16_CHECKPOINT_SCHEMA,
        "config": asdict(spec.main_training_config()),
        "progress": {"step": 5000},
        "experiment": identity,
        "raw_state_dict": {"text_encoder": raw_text, "conditional_unet": raw_unet},
        "ema_state_dict": {"text_encoder": ema_text, "conditional_unet": ema_unet},
    }
    path = tmp_path / f"{spec.architecture_arm.value}.pt"
    torch.save(checkpoint, path)
    return path


def _measured_result_fixture(tmp_path: Path) -> dict[str, object]:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = {
        "schema": CA16_CHECKPOINT_SCHEMA,
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "progress": {"step": 5000},
        "ema_state_dict": {},
        "experiment": spec.provenance(execution_git_sha="fixture-sha", branch="fixture-branch"),
    }
    artifacts = _artifact_set(tmp_path, spec, 5000, "ID_VAL") + _artifact_set(tmp_path, spec, 5000, "COMP_VAL")
    per_attribute = {attribute: 1.0 for attribute in ATTRIBUTES}
    family_metrics = {
        family: {"per_attribute_accuracy": dict(per_attribute), "maa": 1.0, "psr": 1.0}
        for family in ("CV1", "CV2", "CV3", "CV4")
    }
    conditioning = {
        "id_count": 256,
        "id_metrics": {"maa": 1.0, "per_attribute_accuracy": dict(per_attribute)},
        "comp_count": 237,
        "comp_family_counts": {"CV1": 40, "CV2": 114, "CV3": 27, "CV4": 56},
        "comp_per_family": family_metrics,
        "family_macro_maa": 1.0,
        "macro_psr": 1.0,
    }
    evaluator = {
        "architecture": ARCHITECTURE_ID,
        "preprocessing": PREPROCESSING_ID,
        "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE,
        "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
        "freeze_manifest_path": str(EVALUATOR_FREEZE),
        "freeze_manifest_sha256": EVALUATOR_FREEZE_SHA256,
        "attribute_order": json.dumps(list(CANONICAL_ATTRIBUTE_ORDER)),
        "class_counts": json.dumps(CLASS_COUNTS, sort_keys=True),
        "semantic_mapping": json.dumps(
            SEMANTIC_MAPPING, sort_keys=True, default=lambda item: asdict(item)
        ),
    }
    rows = [
        {
            "milestone_step": step,
            "comp_family_macro_maa": 1.0 if step == 5000 else 0.9,
            "comp_macro_psr": 1.0,
            "id_maa": 1.0,
            "kid_status": "NOT_REQUIRED",
            "comp_kid": None,
        }
        for step in CANDIDATE_MILESTONES
    ]
    ranking = {
        "criteria_order": [
            "comp_family_macro_maa", "comp_macro_psr", "id_maa", "comp_kid", "milestone_step"
        ],
        "candidates": rows,
        "selected_milestone": 5000,
    }
    return {
        "candidate": SelectionCandidate("r", "ca8", 2026, 5000, 1.0, 1.0, 1.0),
        "spec": spec,
        "checkpoint": checkpoint,
        "conditioning": conditioning,
        "artifacts": artifacts,
        "evaluator": evaluator,
        "ranking": ranking,
    }


def test_ema_loader_uses_ema_and_preserves_architecture_namespace(tmp_path: Path) -> None:
    for arm in (ArchitectureArm.CA8, ArchitectureArm.CA8_CA16):
        spec = CA16RunSpec(arm, 2026)
        path = _checkpoint_fixture(tmp_path, spec)
        text, unet, _ = load_ema_checkpoint(path, spec, 5000)
        assert next(iter(text.state_dict().values())).eq(1).all()
        names = set(unet.state_dict())
        namespace = "decoder16_image_text_cross_attention"
        assert any(namespace in name for name in names) is (arm is ArchitectureArm.CA8_CA16)
        assert not text.training
        assert not unet.training


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "wrong"),
        ("training_seed", 9999),
        ("architecture_arm", "ca8_ca16"),
        ("use_decoder16_cross_attention", True),
        ("manifests", []),
    ],
)
def test_ema_loader_rejects_identity_mismatches(tmp_path: Path, field: str, value: object) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    with pytest.raises(ValueError):
        load_ema_checkpoint(
            _checkpoint_fixture(tmp_path, spec, identity_update={field: value}),
            spec,
            5000,
        )


def test_ema_identity_rejection_precedes_state_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    path = _checkpoint_fixture(tmp_path, spec, identity_update={"run_id": "wrong"})
    calls = {"text": 0, "unet": 0}

    def reject_text(*args: object, **kwargs: object) -> None:
        calls["text"] += 1
        raise AssertionError("text EMA state loading must not be reached")

    def reject_unet(*args: object, **kwargs: object) -> None:
        calls["unet"] += 1
        raise AssertionError("UNet EMA state loading must not be reached")

    monkeypatch.setattr(TextConditioningEncoder, "load_state_dict", reject_text)
    monkeypatch.setattr(ConditionalTinyUNet, "load_state_dict", reject_unet)
    with pytest.raises(ValueError, match="identity"):
        load_ema_checkpoint(path, spec, 5000)
    assert calls == {"text": 0, "unet": 0}


def test_ema_loader_rejects_main_training_config_mismatch(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    path = _checkpoint_fixture(tmp_path, spec)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"]["seed"] = 9999
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_ema_checkpoint(path, spec, 5000)


def test_measured_result_requires_complete_contract(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = {
        "schema": CA16_CHECKPOINT_SCHEMA,
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "progress": {"step": 5000},
        "ema_state_dict": {},
        "experiment": spec.provenance(execution_git_sha="fixture-sha", branch="fixture-branch"),
    }
    artifacts = _artifact_set(tmp_path, spec, 5000, "ID_VAL") + _artifact_set(tmp_path, spec, 5000, "COMP_VAL")
    per_attribute = {attribute: 1.0 for attribute in ATTRIBUTES}
    family_metrics = {
        family: {"per_attribute_accuracy": per_attribute, "maa": 1.0, "psr": 1.0}
        for family in ("CV1", "CV2", "CV3", "CV4")
    }
    conditioning = {
        "id_count": 256,
        "id_metrics": {"maa": 1.0, "per_attribute_accuracy": per_attribute},
        "comp_count": 237,
        "comp_family_counts": {"CV1": 40, "CV2": 114, "CV3": 27, "CV4": 56},
        "comp_per_family": family_metrics,
        "family_macro_maa": 1.0,
        "macro_psr": 1.0,
    }
    evaluator = {
        "architecture": ARCHITECTURE_ID,
        "preprocessing": PREPROCESSING_ID,
        "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE,
        "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
        "freeze_manifest_path": str(EVALUATOR_FREEZE),
        "freeze_manifest_sha256": EVALUATOR_FREEZE_SHA256,
        "attribute_order": json.dumps(list(CANONICAL_ATTRIBUTE_ORDER)),
        "class_counts": json.dumps(CLASS_COUNTS, sort_keys=True),
        "semantic_mapping": json.dumps(
            SEMANTIC_MAPPING, sort_keys=True, default=lambda item: asdict(item)
        ),
    }
    rows = [
        {
            "milestone_step": step,
            "comp_family_macro_maa": 1.0 if step == 5000 else 0.9,
            "comp_macro_psr": 1.0,
            "id_maa": 1.0,
            "kid_status": "NOT_REQUIRED",
            "comp_kid": None,
        }
        for step in CANDIDATE_MILESTONES
    ]
    ranking = {
        "criteria_order": [
            "comp_family_macro_maa", "comp_macro_psr", "id_maa", "comp_kid", "milestone_step"
        ],
        "candidates": rows,
        "selected_milestone": 5000,
    }
    candidate = SelectionCandidate("r", "ca8", 2026, 5000, 1.0, 1.0, 1.0)
    result = build_measured_selection_result(
        candidate=candidate,
        spec=spec,
        checkpoint=checkpoint,
        conditioning=conditioning,
        artifacts=artifacts,
        evaluator=evaluator,
        selected=True,
        ranking=ranking,
    )
    assert result.measured is True
    assert result.as_dict() == result.as_dict()
    with pytest.raises(ValueError):
        build_measured_selection_result(
            candidate=candidate,
            spec=spec,
            checkpoint=checkpoint,
            conditioning={**conditioning, "id_count": 255},
            artifacts=artifacts,
            evaluator=evaluator,
            selected=True,
            ranking=ranking,
        )
    with pytest.raises(ValueError):
        build_measured_selection_result(
            candidate=candidate,
            spec=spec,
            checkpoint=checkpoint,
            conditioning=conditioning,
            artifacts=artifacts,
            evaluator={**evaluator, "checkpoint_sha256": "wrong"},
            selected=True,
            ranking=ranking,
        )


def _assert_measured_result_rejected(
    fixture: dict[str, object], **updates: object
) -> None:
    payload = dict(fixture)
    payload.update(updates)
    build_measured_selection_result(
        candidate=payload["candidate"],
        spec=payload["spec"],
        checkpoint=payload["checkpoint"],
        conditioning=payload["conditioning"],
        artifacts=payload["artifacts"],
        evaluator=payload["evaluator"],
        selected=True,
        ranking=payload["ranking"],
        kid=payload.get("kid"),
    )


def test_measured_result_rejects_wrong_comp_count(tmp_path: Path) -> None:
    fixture = _measured_result_fixture(tmp_path)
    conditioning = deepcopy(fixture["conditioning"])
    conditioning["comp_count"] = 236
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(fixture, conditioning=conditioning)


def test_measured_result_rejects_wrong_family_count(tmp_path: Path) -> None:
    fixture = _measured_result_fixture(tmp_path)
    conditioning = deepcopy(fixture["conditioning"])
    conditioning["comp_family_counts"]["CV2"] = 113
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(fixture, conditioning=conditioning)


def test_measured_result_rejects_missing_attribute_metric(tmp_path: Path) -> None:
    fixture = _measured_result_fixture(tmp_path)
    conditioning = deepcopy(fixture["conditioning"])
    del conditioning["id_metrics"]["per_attribute_accuracy"]["glasses"]
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(fixture, conditioning=conditioning)


def test_measured_result_rejects_inconsistent_not_required_kid(tmp_path: Path) -> None:
    fixture = _measured_result_fixture(tmp_path)
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(
            fixture, kid={"status": "NOT_REQUIRED", "value": 0.25}
        )


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_measured_result_rejects_inconsistent_computed_kid(
    tmp_path: Path, value: float | None
) -> None:
    fixture = _measured_result_fixture(tmp_path)
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(
            fixture, kid={"status": "COMPUTED", "value": value}
        )


def test_measured_result_rejects_inconsistent_selected_milestone(tmp_path: Path) -> None:
    fixture = _measured_result_fixture(tmp_path)
    ranking = deepcopy(fixture["ranking"])
    ranking["selected_milestone"] = 10_000
    with pytest.raises(ValueError):
        _assert_measured_result_rejected(fixture, ranking=ranking)


def test_dry_run_plans_have_no_model_evaluator_or_generation_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run plan invoked a forbidden side effect")

    for target in (
        "TextConditioningEncoder",
        "ConditionalTinyUNet",
        "load_frozen_evaluator",
        "predict_attributes",
        "compute_comp_kid_from_artifacts",
        "conditional_sample",
        "torch.cuda.init",
    ):
        if target == "torch.cuda.init":
            monkeypatch.setattr(torch.cuda, "init", forbidden)
        else:
            monkeypatch.setattr(evaluation_module, target, forbidden)
    manifest = load_checkpoint_selection_manifest()
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    selection_plan = evaluation_module.build_generation_plan(manifest, spec, 5000)
    robust_plan = build_robust_conditioning_plan()
    assert selection_plan["measured"] is False
    assert robust_plan["measured"] is False
    assert "ID_TEST" not in json.dumps(selection_plan)
    assert "FINAL_OOD" not in json.dumps(selection_plan)
    assert "ID_TEST" not in json.dumps(robust_plan)
    assert "FINAL_OOD" not in json.dumps(robust_plan)


def test_evaluator_sample_uses_frozen_preprocessing(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), (255, 0, 128)).save(path)
    sample = load_evaluator_sample(path)
    assert tuple(sample.shape) == (3, 32, 32)
    assert str(sample.dtype) == "torch.float32"
    assert float(sample[0, 0, 0]) == pytest.approx(1.0)
    assert float(sample[1, 0, 0]) == pytest.approx(-1.0)


def test_kid_protocol_and_rng_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    artifacts = _artifact_set(tmp_path, spec, 5000, "COMP_VAL")
    real = tmp_path / "real"
    real.mkdir()
    for index in range(237):
        (real / f"{index}.png").write_bytes(b"real")
    calls: dict[str, object] = {}
    fake_fid = types.SimpleNamespace(
        build_feature_extractor=lambda **kwargs: calls.setdefault("build", kwargs),
        get_folder_features=lambda *args, **kwargs: np.zeros((237, 4)),
        kernel_distance=lambda *args, **kwargs: calls.update(kernel=kwargs) or .25,
    )
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.1.35")
    monkeypatch.setitem(sys.modules, "cleanfid", types.SimpleNamespace(fid=fake_fid))
    monkeypatch.setattr(
        "tiny_avatar_diffusion.ca16_evaluation.conditional_sample",
        lambda *args, **kwargs: pytest.fail("KID must not regenerate samples"),
    )
    np.random.seed(77)
    before = np.random.get_state()
    result = compute_comp_kid_from_artifacts(artifacts, real, spec=spec, milestone_step=5000)
    after = np.random.get_state()
    assert all(np.array_equal(left, right) for left, right in zip(before, after))
    assert result["num_subsets"] == 100
    assert result["subset_size"] == 237
    assert result["subset_seed"] == 2026
    assert calls["build"]["mode"] == "clean"
    assert calls["kernel"] == {"num_subsets": 100, "max_subset_size": 237}


def test_kid_rng_restores_after_kernel_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    artifacts = _artifact_set(tmp_path, spec, 5000, "COMP_VAL")
    real = tmp_path / "real"
    real.mkdir()
    for index in range(237):
        (real / f"{index}.png").write_bytes(b"real")
    def raising_kernel(*args: object, **kwargs: object) -> float:
        np.random.random()
        raise RuntimeError("kernel failure")
    fake_fid = types.SimpleNamespace(
        build_feature_extractor=lambda **kwargs: object(),
        get_folder_features=lambda *args, **kwargs: np.zeros((237, 4)),
        kernel_distance=raising_kernel,
    )
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.1.35")
    monkeypatch.setitem(sys.modules, "cleanfid", types.SimpleNamespace(fid=fake_fid))
    np.random.seed(123)
    before = np.random.get_state()
    with pytest.raises(RuntimeError):
        compute_comp_kid_from_artifacts(artifacts, real, spec=spec, milestone_step=5000)
    after = np.random.get_state()
    assert all(np.array_equal(left, right) for left, right in zip(before, after))


def test_manifests_are_role_specific() -> None:
    assert ROBUST_MANIFEST.name == "robust_conditioning_manifest.json"
    assert QUALITY_MANIFEST.name == "quality_guardrail_manifest.json"
