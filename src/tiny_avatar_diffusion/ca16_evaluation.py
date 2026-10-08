"""Deterministic CA16 checkpoint-selection and conditioning-evaluation plans.

This module is intentionally evaluation-only.  Building a plan validates frozen
artifacts and metadata but never trains a model or generates an image.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_experiment import (
    ROOT,
    CA16RunSpec,
)
from tiny_avatar_diffusion.conditional_main_training import CA16_CHECKPOINT_SCHEMA
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.evaluator import (
    ARCHITECTURE_ID,
    CLASS_COUNTS,
    PREPROCESSING_ID,
    AttributeEvaluator,
    preprocess,
)
from tiny_avatar_diffusion.semantic_mapping import (
    CANONICAL_ATTRIBUTE_ORDER,
    SEMANTIC_MAPPING,
)
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

SELECTION_MANIFEST = ROOT / "data" / "processed" / "ca16_eval_v1" / "checkpoint_selection_manifest.json"
ROBUST_MANIFEST = ROOT / "data" / "processed" / "ca16_eval_v1" / "robust_conditioning_manifest.json"
QUALITY_MANIFEST = ROOT / "data" / "processed" / "ca16_eval_v1" / "quality_guardrail_manifest.json"
EVALUATOR_FREEZE = ROOT / "data" / "processed" / "dr04_evaluator_v1" / "evaluator_checkpoint_freeze.json"
SELECTION_MANIFEST_SHA256 = "192191235af7992b8c332019b7e948414ba0b87c5f8df8d70f044b74e85e3d40"
ROBUST_MANIFEST_SHA256 = "e3319f65f67c19ae8b458f7450ff8e6b86e5913e0fd542e9c0c3009802451aec"
QUALITY_MANIFEST_SHA256 = "e34db5251d8949a5fb37a905f6e9e4791602047f3153a9a9908d14d9cd85e84f"
CANDIDATE_MILESTONES = (5_000, 10_000, 15_000, 20_000)
ALLOWED_ROLES = frozenset(("ID_VAL", "COMP_VAL"))
DENIED_ROLES = frozenset(("TRAIN", "ID_TEST", "FINAL_OOD"))
ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
FAMILIES = ("CV1", "CV2", "CV3", "CV4")
EXPECTED_SELECTION_FAMILY_COUNTS = {"CV1": 40, "CV2": 114, "CV3": 27, "CV4": 56}
AMBIGUOUS_COMP_IMAGE_ID = "cs10387983567686797037"
EVALUATOR_FREEZE_SHA256 = "85b0cef08bb5a298442d7d863c4f76b114ecf023335d312f319c6feef168f13f"
EVALUATOR_CHECKPOINT_RELATIVE = "results/runs/dr04_evaluator_main_100k_seed2026_r1/best.pt"
EVALUATOR_CHECKPOINT_SHA256 = "3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f"
FROZEN_GENERATION = {
    "timesteps": 1_000,
    "pixel_space_ddpm": True,
    "clip_denoised": True,
    "cfg": False,
    "conditioning_dropout": False,
    "sampler": "pixel_ddpm_ancestral",
    "weight_set": "EMA",
}
CLEAN_FID_VERSION = "0.1.35"
KID_SUBSET_SEED = 2026
KID_NUM_SUBSETS = 100


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"manifest must contain an object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=lambda item: asdict(item))


def _validate_records(records: Iterable[Mapping[str, Any]], *, source: str) -> tuple[dict[str, Any], ...]:
    materialized = tuple(dict(record) for record in records)
    for record in materialized:
        role = record.get("split_role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{source} contains forbidden split role: {role!r}")
        if not record.get("stable_record_key"):
            raise ValueError(f"{source} record lacks stable_record_key")
        if not record.get("image_id"):
            raise ValueError(f"{source} record lacks image_id")
    return materialized


def _record_seed(record: Mapping[str, Any], *, index: int, role: str) -> int:
    if role == "ID_VAL":
        return 2026 + index
    seed = record.get("generation_seed")
    if not isinstance(seed, int):
        raise TypeError("COMP selection record lacks its frozen generation_seed")
    if seed != 2026 + index:
        raise ValueError("COMP selection seed is not the frozen canonical 2026+i mapping")
    return seed


def _validate_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"frozen manifest hash mismatch: expected {expected}, got {actual}")


def load_checkpoint_selection_manifest(path: Path = SELECTION_MANIFEST) -> dict[str, Any]:
    """Load and validate the immutable checkpoint-selection panel."""
    if path.resolve() != SELECTION_MANIFEST.resolve():
        raise ValueError("checkpoint selection requires the frozen R2 selection manifest")
    _validate_hash(path, SELECTION_MANIFEST_SHA256)
    manifest = _read_json(path)
    if manifest.get("schema") != "tiny_avatar_diffusion.ca16_checkpoint_selection_manifest":
        raise ValueError("unexpected checkpoint-selection manifest schema")
    ids = _validate_records(manifest.get("id_entries", ()), source="ID selection")
    comps = _validate_records(manifest.get("comp_entries", ()), source="COMP selection")
    if len(ids) != 256 or len(comps) != 237:
        raise ValueError("checkpoint-selection panel must contain 256 ID and 237 COMP records")
    ids = [{**record, "generation_seed": _record_seed(record, index=i, role="ID_VAL")}
           for i, record in enumerate(ids)]
    comps = [{**record, "generation_seed": _record_seed(record, index=i, role="COMP_VAL")}
             for i, record in enumerate(comps)]
    family_counts: dict[str, int] = {family: 0 for family in FAMILIES}
    for record in comps:
        families = record.get("comp_families", ())
        if len(families) != 1 or families[0] not in FAMILIES:
            raise ValueError("each COMP selection record must have exactly one known family")
        family_counts[families[0]] += 1
    if family_counts != EXPECTED_SELECTION_FAMILY_COUNTS:
        raise ValueError(f"unexpected COMP selection family counts: {family_counts}")
    if any(record.get("image_id") == AMBIGUOUS_COMP_IMAGE_ID for record in comps):
        raise ValueError("ambiguous COMP record must not be in selection panel")
    return {
        **manifest,
        "_path": str(path),
        "id_entries": ids,
        "comp_entries": comps,
        "family_counts": family_counts,
    }


def load_robust_conditioning_manifest(path: Path = ROBUST_MANIFEST) -> dict[str, Any]:
    """Load the post-selection robust panel, never as a selection input."""
    if path.resolve() != ROBUST_MANIFEST.resolve():
        raise ValueError("robust evaluation requires the frozen R2 robust manifest")
    _validate_hash(path, ROBUST_MANIFEST_SHA256)
    manifest = _read_json(path)
    if manifest.get("schema") != "tiny_avatar_diffusion.ca16_robust_conditioning_manifest":
        raise ValueError("unexpected robust-conditioning manifest schema")
    entries = _validate_records(manifest.get("entries", ()), source="robust conditioning")
    if len(entries) != 200:
        raise ValueError("robust-conditioning panel must contain 200 prompts")
    if sum(record.get("panel") == "robust_id" for record in entries) != 100:
        raise ValueError("robust ID panel must contain 100 prompts")
    comp = [record for record in entries if record.get("panel") == "robust_comp"]
    if len(comp) != 100 or any(len(record.get("generation_seeds", ())) != 5 for record in entries):
        raise ValueError("robust panel must contain five seeds per prompt")
    if {seed for record in entries for seed in record["generation_seeds"]} != {2026, 2027, 2028, 2029, 2030}:
        raise ValueError("robust panel uses unexpected generation seeds")
    counts = {
        family: sum(record.get("comp_family") == family for record in comp)
        for family in FAMILIES
    }
    if counts != {family: 25 for family in FAMILIES}:
        raise ValueError("robust COMP panel must contain 25 prompts per family")
    return {**manifest, "_path": str(path), "entries": list(entries)}


def _reject_non_selection_manifest(path: Path) -> None:
    if path.resolve() in {ROBUST_MANIFEST.resolve(), QUALITY_MANIFEST.resolve()}:
        raise ValueError("robust and quality manifests cannot be checkpoint-selection inputs")


def build_robust_conditioning_plan() -> dict[str, Any]:
    manifest = load_robust_conditioning_manifest()
    return {
        "kind": "dry_run_generation_plan",
        "manifest_path": str(ROBUST_MANIFEST),
        "manifest_sha256": ROBUST_MANIFEST_SHA256,
        "prompts": len(manifest["entries"]),
        "generations_per_model": 1000,
        "seeds": [2026, 2027, 2028, 2029, 2030],
        "measured": False,
    }


class PanelKind(str, Enum):
    SELECTION = "checkpoint_selection"
    ROBUST = "robust_conditioning"
    QUALITY = "quality_guardrail"


@dataclass(frozen=True)
class SelectionPanel:
    """Typed checkpoint-selection records; other panel kinds cannot be ranked."""

    records: tuple[Mapping[str, Any], ...]
    manifest_path: str
    manifest_sha256: str
    kind: PanelKind = PanelKind.SELECTION


def load_selection_panel() -> SelectionPanel:
    manifest = load_checkpoint_selection_manifest()
    return SelectionPanel(
        records=tuple(manifest["id_entries"] + manifest["comp_entries"]),
        manifest_path=str(SELECTION_MANIFEST),
        manifest_sha256=SELECTION_MANIFEST_SHA256,
    )


@dataclass(frozen=True)
class SampleIdentity:
    experiment_id: str
    run_id: str
    architecture_arm: str
    training_seed: int
    milestone_step: int
    split_role: str
    record_id: str
    generation_seed: int
    weight_set: str = "EMA"

    def as_key(self) -> str:
        return "|".join(
            (self.experiment_id, self.run_id, self.architecture_arm, str(self.training_seed),
             str(self.milestone_step), self.split_role, self.record_id,
             str(self.generation_seed), self.weight_set)
        )


@dataclass(frozen=True)
class SampleArtifact:
    identity: SampleIdentity
    artifact_path: str
    ground_truth: Mapping[str, int]
    predictions: Mapping[str, int]
    correctness: Mapping[str, bool]


def _expected_panel_records(role: str) -> dict[str, Mapping[str, Any]]:
    manifest = load_checkpoint_selection_manifest()
    entries = manifest["id_entries"] if role == "ID_VAL" else manifest["comp_entries"]
    return {str(entry["stable_record_key"]): entry for entry in entries}


def validate_sample_artifacts(
    artifacts: Sequence[SampleArtifact],
    spec: CA16RunSpec,
    milestone_step: int,
    *,
    role: str,
) -> tuple[SampleArtifact, ...]:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"sample artifact validation rejects split role: {role}")
    expected = _expected_panel_records(role)
    if len(artifacts) != len(expected):
        raise ValueError(f"{role} requires exactly {len(expected)} sample artifacts")
    seen: set[str] = set()
    validated: list[SampleArtifact] = []
    for artifact in artifacts:
        identity = artifact.identity
        if identity.record_id in seen:
            raise ValueError(f"duplicate sample identity: {identity.record_id}")
        seen.add(identity.record_id)
        record = expected.get(identity.record_id)
        if record is None:
            raise ValueError(f"unknown {role} sample identity: {identity.record_id}")
        expected_identity = stable_sample_identity(spec, milestone_step, record)
        if identity != expected_identity:
            raise ValueError(f"sample identity mismatch: {identity.as_key()}")
        path = Path(artifact.artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"sample artifact is missing: {path}")
        validated.append(artifact)
    if seen != set(expected):
        raise ValueError(f"{role} sample artifacts are incomplete")
    return tuple(validated)


def validate_selection_artifacts(
    artifacts: Sequence[SampleArtifact],
    spec: CA16RunSpec,
    milestone_step: int,
) -> tuple[SampleArtifact, ...]:
    id_artifacts = [artifact for artifact in artifacts if artifact.identity.split_role == "ID_VAL"]
    comp_artifacts = [artifact for artifact in artifacts if artifact.identity.split_role == "COMP_VAL"]
    if len(id_artifacts) + len(comp_artifacts) != len(artifacts):
        raise ValueError("selection artifacts contain a forbidden split role")
    return (
        *validate_sample_artifacts(id_artifacts, spec, milestone_step, role="ID_VAL"),
        *validate_sample_artifacts(comp_artifacts, spec, milestone_step, role="COMP_VAL"),
    )


def load_evaluator_sample(path: Path, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Read one generated image using the frozen V1 evaluator preprocessing."""
    with Image.open(path) as image:
        return preprocess(image).to(device)


@dataclass(frozen=True)
class SelectionCandidate:
    run_id: str
    architecture_arm: str
    training_seed: int
    milestone_step: int
    comp_family_macro_maa: float
    comp_macro_psr: float
    id_maa: float
    comp_kid: float | None = None
    epsilon_mse: float | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Machine-readable measured selection output, distinct from a dry-run plan."""

    experiment_id: str
    run_id: str
    architecture_arm: str
    training_seed: int
    milestone_step: int
    checkpoint_path: str
    checkpoint_sha256: str
    ema: bool
    selection_manifest_path: str
    selection_manifest_sha256: str
    evaluator: Mapping[str, str]
    generation: Mapping[str, Any]
    conditioning: Mapping[str, Any]
    kid: float | None
    kid_status: str
    selected: bool
    checkpoint_schema: str = CA16_CHECKPOINT_SCHEMA
    execution_git_sha: str = ""
    execution_branch: str | None = None
    ranking: Mapping[str, Any] | None = None
    per_record: tuple[Mapping[str, Any], ...] = ()
    measured: bool = True

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({
            "schema": "tiny_avatar_diffusion.ca16_selection_result_v1",
            **self.__dict__,
        }, sort_keys=True, default=list))


def validate_candidate_milestones(candidates: Iterable[SelectionCandidate]) -> None:
    candidates = tuple(candidates)
    steps = [candidate.milestone_step for candidate in candidates]
    if len(steps) != len(set(steps)):
        raise ValueError("candidate milestones contain duplicates")
    if set(steps) != set(CANDIDATE_MILESTONES) or len(steps) != len(CANDIDATE_MILESTONES):
        raise ValueError(f"candidate set must contain exactly {CANDIDATE_MILESTONES}")


def _candidate_key(candidate: SelectionCandidate) -> tuple[float, float, float, float, int]:
    """Return descending adherence/KID and ascending milestone ranking values."""
    kid = float("inf") if candidate.comp_kid is None else candidate.comp_kid
    return (
        -candidate.comp_family_macro_maa,
        -candidate.comp_macro_psr,
        -candidate.id_maa,
        kid,
        candidate.milestone_step,
    )


def rank_candidates(candidates: Sequence[SelectionCandidate], *, kid: Callable[[SelectionCandidate], float] | None = None) -> list[SelectionCandidate]:
    """Rank candidates by the frozen adherence-first hierarchy."""
    validate_candidate_milestones(candidates)
    groups: dict[tuple[float, float, float], list[SelectionCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(
            (
                candidate.comp_family_macro_maa,
                candidate.comp_macro_psr,
                candidate.id_maa,
            ),
            [],
        ).append(candidate)
    best_group_key = min(
        groups,
        key=lambda values: (-values[0], -values[1], -values[2]),
    )
    enriched: list[SelectionCandidate] = []
    for group_key, group in groups.items():
        if (
            len(group) > 1
            and group_key == best_group_key
            and any(candidate.comp_kid is None for candidate in group)
        ):
            if kid is None:
                raise ValueError("KID is required to resolve a conditioning tie")
            group = [
                candidate if candidate.comp_kid is not None else SelectionCandidate(
                    **{**candidate.__dict__, "comp_kid": kid(candidate)}
                )
                for candidate in group
            ]
        enriched.extend(group)
    if len(enriched) != len({candidate.milestone_step for candidate in enriched}):
        raise ValueError("candidate milestones must be unique")
    return sorted(enriched, key=_candidate_key)


def select_final_candidate(
    candidates: Sequence[SelectionCandidate],
    *,
    kid: Callable[[SelectionCandidate], float] | None = None,
) -> SelectionCandidate:
    """Select a frozen four-milestone candidate with lazy fourth-level KID."""
    if any(not isinstance(candidate, SelectionCandidate) for candidate in candidates):
        raise TypeError("final selection accepts SelectionCandidate values only")
    return rank_candidates(candidates, kid=kid)[0]


def attribute_metrics(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    """Compute frozen MAA, PSR, and per-attribute/class accuracy summaries."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    if not records:
        raise ValueError("at least one record is required")
    per_attribute: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = {}
    for attribute in ATTRIBUTES:
        pairs = [(int(record["ground_truth_attributes"][attribute]), int(prediction[attribute])) for record, prediction in zip(records, predictions)]
        per_attribute[attribute] = sum(actual == predicted for actual, predicted in pairs) / len(pairs)
        classes: dict[str, int] = {}
        for actual, predicted in pairs:
            key = f"{actual}->{predicted}"
            classes[key] = classes.get(key, 0) + 1
        confusion[attribute] = classes
    psr = sum(
        all(int(record["ground_truth_attributes"][attribute]) == int(prediction[attribute]) for attribute in ATTRIBUTES)
        for record, prediction in zip(records, predictions)
    ) / len(records)
    return {"maa": sum(per_attribute.values()) / len(ATTRIBUTES), "psr": psr, "per_attribute_accuracy": per_attribute, "confusion": confusion, "sample_count": len(records)}


def family_macro_metrics(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    groups: dict[str, tuple[list[Mapping[str, Any]], list[Mapping[str, int]]]] = {}
    for record, prediction in zip(records, predictions):
        families = record.get("comp_families", ())
        if not families and record.get("comp_family"):
            families = (record["comp_family"],)
        if len(families) != 1:
            raise ValueError("family-macro metrics require one unambiguous family per record")
        family = families[0]
        if family not in FAMILIES:
            raise ValueError(f"unknown COMP family: {family}")
        if family not in groups:
            groups[family] = ([], [])
        groups[family][0].append(record)
        groups[family][1].append(prediction)
    if set(groups) != set(FAMILIES):
        raise ValueError("all four COMP families are required")
    per_family = {family: attribute_metrics(*groups[family]) for family in FAMILIES}
    return {
        "per_family": per_family,
        "family_macro_maa": sum(value["maa"] for value in per_family.values()) / 4,
        "macro_psr": sum(value["psr"] for value in per_family.values()) / 4,
    }


def stable_sample_identity(spec: CA16RunSpec, milestone_step: int, record: Mapping[str, Any]) -> SampleIdentity:
    if milestone_step not in CANDIDATE_MILESTONES:
        raise ValueError("sample identity requires a frozen candidate milestone")
    canonical = _canonical_selection_record(record)
    role = canonical["split_role"]
    seed = canonical["generation_seed"]
    return SampleIdentity(
        experiment_id="ca16_ablation",
        run_id=spec.run_id,
        architecture_arm=spec.architecture_arm.value,
        training_seed=spec.training_seed,
        milestone_step=milestone_step,
        split_role=str(role),
        record_id=str(record["stable_record_key"]),
        generation_seed=seed,
    )


def load_ema_checkpoint(path: Path, spec: CA16RunSpec, milestone_step: int, *, device: torch.device | str = "cpu") -> tuple[TextConditioningEncoder, ConditionalTinyUNet, dict[str, Any]]:
    """Load only the EMA model states after validating CA16 identity."""
    if milestone_step not in CANDIDATE_MILESTONES:
        raise ValueError("checkpoint is not a frozen candidate milestone")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != CA16_CHECKPOINT_SCHEMA:
        raise ValueError("scientific CA16 evaluation requires the CA16 checkpoint schema")
    if checkpoint.get("progress", {}).get("step") != milestone_step:
        raise ValueError("checkpoint milestone does not match requested step")
    identity = checkpoint.get("experiment")
    if not isinstance(identity, dict):
        raise TypeError("checkpoint lacks CA16 experiment identity")
    expected = spec.provenance(
        execution_git_sha=str(identity.get("git_commit", "")),
        branch=identity.get("branch"),
    )
    required_identity = (
        "experiment_id", "run_id", "architecture_arm",
        "use_decoder16_cross_attention", "training_seed",
        "frozen_training", "text_encoder", "manifests",
    )
    if any(identity.get(key) != expected.get(key) for key in required_identity):
        raise ValueError("checkpoint CA16 identity does not match requested run spec")
    if checkpoint.get("config") != asdict(spec.main_training_config()):
        raise ValueError("checkpoint frozen MainTrainingConfig does not match requested run")
    if identity.get("frozen_training") != dict(spec.frozen_protocol):
        raise ValueError("checkpoint frozen protocol metadata does not match requested run")
    if identity.get("git_commit") == "":
        raise ValueError("checkpoint CA16 identity lacks execution Git SHA")
    ema = checkpoint.get("ema_state_dict")
    if not isinstance(ema, dict) or set(ema) != {"text_encoder", "conditional_unet"}:
        raise ValueError("checkpoint lacks complete EMA state")
    text_encoder = TextConditioningEncoder().to(device)
    denoiser = ConditionalTinyUNet(use_decoder16_cross_attention=spec.use_decoder16_cross_attention, dropout=0.0).to(device)
    text_encoder.load_state_dict(ema["text_encoder"])
    denoiser.load_state_dict(ema["conditional_unet"])
    text_encoder.eval()
    denoiser.eval()
    return text_encoder, denoiser, checkpoint


def evaluator_identity(
    freeze_path: Path = EVALUATOR_FREEZE,
    checkpoint_path: Path | None = None,
) -> dict[str, str]:
    if freeze_path.resolve() != EVALUATOR_FREEZE.resolve():
        raise ValueError("scientific evaluation requires the canonical DR04 freeze manifest")
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "FROZEN":
        raise ValueError("DR04 evaluator freeze is not marked FROZEN")
    if freeze.get("selected_checkpoint") != EVALUATOR_CHECKPOINT_RELATIVE:
        raise ValueError("canonical DR04 freeze manifest selects an unexpected checkpoint")
    checkpoint = ROOT / EVALUATOR_CHECKPOINT_RELATIVE
    if checkpoint_path is not None and checkpoint_path.resolve() != checkpoint.resolve():
        raise ValueError("scientific evaluation requires the canonical frozen checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "frozen DR04 evaluator checkpoint is unavailable: "
            f"{checkpoint}; measured evaluation is disabled"
        )
    actual = sha256_file(checkpoint)
    if actual != EVALUATOR_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen DR04 evaluator checkpoint hash mismatch: "
            f"expected {EVALUATOR_CHECKPOINT_SHA256}, got {actual}"
        )
    return {
        "architecture": ARCHITECTURE_ID,
        "preprocessing": PREPROCESSING_ID,
        "run_id": str(freeze["run_id"]),
        "checkpoint": str(freeze["selected_checkpoint"]),
        "checkpoint_sha256": actual,
        "freeze_manifest_path": str(EVALUATOR_FREEZE),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "attribute_order": json.dumps(CANONICAL_ATTRIBUTE_ORDER),
        "class_counts": json.dumps(CLASS_COUNTS, sort_keys=True),
        "semantic_mapping": _canonical_json(SEMANTIC_MAPPING),
    }


def _validate_measured_evaluator_identity(evaluator: Mapping[str, str]) -> None:
    required = {
        "architecture", "preprocessing", "checkpoint", "checkpoint_sha256",
        "freeze_manifest_path", "freeze_manifest_sha256", "attribute_order",
        "class_counts", "semantic_mapping",
    }
    if set(evaluator) != required:
        raise ValueError("measured result evaluator identity is incomplete")
    if Path(evaluator["freeze_manifest_path"]).resolve() != EVALUATOR_FREEZE.resolve():
        raise ValueError("measured result evaluator must use the canonical freeze manifest")
    if evaluator["freeze_manifest_sha256"] != EVALUATOR_FREEZE_SHA256:
        raise ValueError("measured result evaluator freeze hash is not canonical")
    if evaluator["checkpoint"] != EVALUATOR_CHECKPOINT_RELATIVE:
        raise ValueError("measured result evaluator checkpoint path is not canonical")
    if evaluator["checkpoint_sha256"] != EVALUATOR_CHECKPOINT_SHA256:
        raise ValueError("measured result evaluator checkpoint hash is not canonical")
    if evaluator["architecture"] != ARCHITECTURE_ID or evaluator["preprocessing"] != PREPROCESSING_ID:
        raise ValueError("measured result evaluator architecture is not canonical")
    if json.loads(evaluator["attribute_order"]) != list(CANONICAL_ATTRIBUTE_ORDER):
        raise ValueError("measured result evaluator attribute order is not canonical")
    if json.loads(evaluator["class_counts"]) != CLASS_COUNTS:
        raise ValueError("measured result evaluator class mapping is not canonical")
    if json.loads(evaluator["semantic_mapping"]) != json.loads(_canonical_json(SEMANTIC_MAPPING)):
        raise ValueError("measured result evaluator semantic mapping is not canonical")


def _validate_conditioning_metrics(conditioning: Mapping[str, Any]) -> None:
    required_attributes = set(ATTRIBUTES)
    if conditioning.get("id_count") != 256 or conditioning.get("comp_count") != 237:
        raise ValueError("conditioning counts must be 256 ID and 237 COMP")
    id_metrics = conditioning.get("id_metrics")
    if not isinstance(id_metrics, Mapping):
        raise TypeError("ID metrics are missing")
    if set(id_metrics.get("per_attribute_accuracy", {})) != required_attributes:
        raise ValueError("ID per-attribute metrics are incomplete")
    families = conditioning.get("comp_per_family")
    counts = conditioning.get("comp_family_counts")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("COMP family metrics must contain exactly CV1-CV4")
    if counts != EXPECTED_SELECTION_FAMILY_COUNTS:
        raise ValueError("COMP family counts do not match the frozen panel")
    for family in FAMILIES:
        metrics = families[family]
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics missing for {family}")
        if set(metrics.get("per_attribute_accuracy", {})) != required_attributes:
            raise ValueError(f"{family} per-attribute metrics are incomplete")
    if "family_macro_maa" not in conditioning or "macro_psr" not in conditioning:
        raise ValueError("COMP aggregate family metrics are missing")


def _validate_ranking_metadata(
    ranking: Mapping[str, Any],
    candidate: SelectionCandidate,
) -> None:
    expected_order = [
        "comp_family_macro_maa", "comp_macro_psr", "id_maa", "comp_kid", "milestone_step"
    ]
    if ranking.get("criteria_order") != expected_order:
        raise ValueError("ranking criteria order is not frozen")
    rows = ranking.get("candidates")
    if not isinstance(rows, Sequence) or len(rows) != len(CANDIDATE_MILESTONES):
        raise ValueError("ranking must contain all four candidate milestones")
    milestones = [row.get("milestone_step") for row in rows if isinstance(row, Mapping)]
    if len(milestones) != len(set(milestones)) or set(milestones) != set(CANDIDATE_MILESTONES):
        raise ValueError("ranking candidate milestones are incomplete or duplicated")
    if ranking.get("selected_milestone") != candidate.milestone_step:
        raise ValueError("ranking selected milestone disagrees with result")
    for row in rows:
        if not isinstance(row, Mapping) or "epsilon_mse" in row:
            raise ValueError("ranking contains an unsupported criterion")
        if row.get("kid_status") not in {"NOT_REQUIRED", "COMPUTED"}:
            raise ValueError("ranking KID status is invalid")
        if row["kid_status"] == "COMPUTED" and not isinstance(row.get("comp_kid"), (int, float)):
            raise ValueError("ranking computed KID value is missing")
        if row["kid_status"] == "NOT_REQUIRED" and row.get("comp_kid") is not None:
            raise ValueError("ranking NOT_REQUIRED KID has a value")


def load_frozen_evaluator(
    *,
    device: torch.device | str = "cpu",
    freeze_path: Path = EVALUATOR_FREEZE,
    checkpoint_path: Path | None = None,
) -> tuple[AttributeEvaluator, dict[str, str]]:
    """Load the frozen real-only evaluator, failing closed if unavailable."""
    if freeze_path.resolve() != EVALUATOR_FREEZE.resolve():
        raise ValueError("scientific evaluation requires the canonical DR04 freeze manifest")
    _validate_hash(EVALUATOR_FREEZE, EVALUATOR_FREEZE_SHA256)
    freeze = _read_json(EVALUATOR_FREEZE)
    if freeze.get("parameter_count") != 293064:
        raise ValueError("canonical DR04 freeze manifest parameter count is not 293064")
    if freeze.get("selected_epoch") != 41:
        raise ValueError("canonical DR04 freeze manifest selected epoch is not 41")
    path = ROOT / EVALUATOR_CHECKPOINT_RELATIVE if checkpoint_path is None else checkpoint_path
    identity = evaluator_identity(EVALUATOR_FREEZE, path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen evaluator checkpoint is unavailable: {path}")
    if sha256_file(path) != EVALUATOR_CHECKPOINT_SHA256:
        raise ValueError("frozen evaluator checkpoint hash mismatch")
    # This legacy frozen V1 checkpoint contains NumPy objects unsupported by
    # weights_only=True; historical V1 loaded this trusted, hash-verified
    # artifact with weights_only=False.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("frozen evaluator checkpoint has an unexpected schema")
    if payload.get("best_epoch") != freeze["selected_epoch"]:
        raise ValueError("frozen evaluator checkpoint best epoch disagrees with freeze manifest")
    if "completed_epoch" in payload and payload["completed_epoch"] != freeze["selected_epoch"]:
        raise ValueError("frozen evaluator checkpoint completed epoch disagrees with freeze manifest")
    evaluator = AttributeEvaluator().to(device)
    parameter_count = sum(parameter.numel() for parameter in evaluator.parameters())
    if parameter_count != freeze["parameter_count"]:
        raise ValueError("instantiated evaluator parameter count disagrees with freeze manifest")
    evaluator.load_state_dict(payload["model_state_dict"])
    evaluator.eval()
    return evaluator, identity


def predict_attributes(evaluator: AttributeEvaluator, samples: torch.Tensor) -> list[dict[str, int]]:
    """Run frozen evaluator inference only; never trains or generates."""
    if samples.ndim != 4 or samples.shape[1:] != (3, 32, 32):
        raise ValueError("samples must have shape [B, 3, 32, 32]")
    with torch.inference_mode():
        logits = evaluator(samples)
    return [
        {attribute: int(logits[attribute][index].argmax().item())
         for attribute in CANONICAL_ATTRIBUTE_ORDER}
        for index in range(samples.shape[0])
    ]


def predict_artifact_attributes(
    evaluator: AttributeEvaluator,
    artifact_paths: Sequence[str],
    *,
    device: torch.device | str = "cpu",
) -> list[dict[str, int]]:
    samples = torch.stack(
        [load_evaluator_sample(Path(path), device=device) for path in artifact_paths]
    )
    return predict_attributes(evaluator, samples)


def evaluate_generated_samples(
    records: Sequence[Mapping[str, Any]],
    samples: torch.Tensor,
    evaluator: AttributeEvaluator,
    identities: Sequence[SampleIdentity],
    artifact_paths: Sequence[str],
    *,
    spec: CA16RunSpec,
    milestone_step: int,
) -> tuple[dict[str, Any], tuple[SampleArtifact, ...]]:
    if len(records) != len(identities) or len(records) != len(artifact_paths):
        raise ValueError("records, identities, and artifact_paths must have equal length")
    if len(samples) != len(records):
        raise ValueError("records and samples must have equal length")
    predictions = predict_attributes(evaluator, samples)
    metrics = attribute_metrics(records, predictions)
    artifacts = tuple(
        SampleArtifact(
            identity=identity,
            artifact_path=artifact_path,
            ground_truth=dict(record["ground_truth_attributes"]),
            predictions=prediction,
            correctness={
                attribute: int(record["ground_truth_attributes"][attribute]) == prediction[attribute]
                for attribute in ATTRIBUTES
            },
        )
        for record, identity, artifact_path, prediction
        in zip(records, identities, artifact_paths, predictions)
    )
    validate_sample_artifacts(
        artifacts,
        spec,
        milestone_step,
        role=identities[0].split_role if identities else "",
    )
    return metrics, artifacts


def build_generation_plan(manifest: Mapping[str, Any], spec: CA16RunSpec, milestone_step: int) -> dict[str, Any]:
    if manifest.get("schema") != "tiny_avatar_diffusion.ca16_checkpoint_selection_manifest":
        raise ValueError("generation plans for checkpoint selection require the selection manifest")
    _reject_non_selection_manifest(Path(str(manifest.get("_path", SELECTION_MANIFEST))))
    records = list(manifest.get("id_entries", ())) + list(manifest.get("comp_entries", ()))
    return {
        "kind": "dry_run_generation_plan",
        "measured": False,
        "run_id": spec.run_id,
        "milestone_step": milestone_step,
        "weight_set": "EMA",
        "generation": dict(FROZEN_GENERATION),
        "records": [stable_sample_identity(spec, milestone_step, record).as_key() for record in records],
    }


def _canonical_selection_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = load_checkpoint_selection_manifest()
    key = record.get("stable_record_key")
    for candidate in panel["id_entries"] + panel["comp_entries"]:
        if candidate["stable_record_key"] == key:
            if candidate["split_role"] != record.get("split_role"):
                raise ValueError("sample record role does not match the frozen panel")
            return candidate
    raise ValueError("sample record is not in the frozen checkpoint-selection panel")


def sample_ema_record(
    schedule: LinearDDPMSchedule,
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    record: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Generate one deterministic sample using an explicit per-record generator."""
    canonical = _canonical_selection_record(record)
    token_ids = record.get("token_ids")
    attention_mask = record.get("attention_mask")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(value) is not int or value < 0 for value in token_ids)
    ):
        raise ValueError("hydrated record has invalid token_ids")
    if (
        not isinstance(attention_mask, list)
        or len(attention_mask) != len(token_ids)
        or any(type(value) is not int or value not in (0, 1) for value in attention_mask)
    ):
        raise ValueError("hydrated record has invalid attention_mask")
    generator = torch.Generator(device=device).manual_seed(int(canonical["generation_seed"]))
    input_ids = torch.tensor([token_ids], device=device)
    mask = torch.tensor([attention_mask], device=device)
    return conditional_sample(schedule, text_encoder, denoiser, input_ids, mask, generator=generator, clip_denoised=True)[0]


def sample_ema_records_batch(
    schedule: LinearDDPMSchedule,
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Sample canonical records in parallel without sharing their RNG streams."""
    if not records:
        raise ValueError("batched sampling requires at least one record")
    canonical = [_canonical_selection_record(record) for record in records]
    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for record in records:
        token_ids = record.get("token_ids")
        attention_mask = record.get("attention_mask")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(value) is not int or value < 0 for value in token_ids)
        ):
            raise ValueError("hydrated record has invalid token_ids")
        if (
            not isinstance(attention_mask, list)
            or len(attention_mask) != len(token_ids)
            or any(type(value) is not int or value not in (0, 1) for value in attention_mask)
        ):
            raise ValueError("hydrated record has invalid attention_mask")
        token_rows.append(token_ids)
        mask_rows.append(attention_mask)
    if len({len(row) for row in token_rows}) != 1:
        raise ValueError("batched sampling requires equal token sequence lengths")
    generators = [
        torch.Generator(device=device).manual_seed(int(record["generation_seed"]))
        for record in canonical
    ]

    def draw_noise() -> torch.Tensor:
        return torch.cat(
            [
                torch.randn(
                    (1, 3, 32, 32),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                for generator in generators
            ],
            dim=0,
        )

    return conditional_sample(
        schedule,
        text_encoder,
        denoiser,
        torch.tensor(token_rows, device=device),
        torch.tensor(mask_rows, device=device),
        x_T=draw_noise(),
        reverse_noise_provider=lambda _timestep, _x_t: draw_noise(),
        clip_denoised=True,
    )


def compute_comp_kid_from_artifacts(
    artifacts: Sequence[SampleArtifact],
    real_directory: Path,
    *,
    spec: CA16RunSpec | None = None,
    milestone_step: int | None = None,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Compute the V1 clean-fid KID using already-existing sample artifacts.

    This function never invokes a generator.  The caller must provide the
    exact COMP selection samples already used for adherence inference.
    """
    if spec is None or milestone_step is None:
        raise ValueError("KID requires the candidate run spec and milestone")
    comp_artifacts = list(
        validate_sample_artifacts(
            [artifact for artifact in artifacts if artifact.identity.split_role == "COMP_VAL"],
            spec,
            milestone_step,
            role="COMP_VAL",
        )
    )
    if len(comp_artifacts) != len(artifacts):
        raise ValueError("KID tie-break input must contain only COMP_VAL samples")
    if not comp_artifacts:
        raise ValueError("KID tie-break requires at least one generated sample")
    identities = {artifact.identity for artifact in comp_artifacts}
    if len({(identity.run_id, identity.architecture_arm, identity.training_seed, identity.milestone_step)
            for identity in identities}) != 1:
        raise ValueError("KID tie-break artifacts must belong to one candidate")
    if len({identity.record_id for identity in identities}) != len(identities):
        raise ValueError("KID tie-break artifacts contain duplicate sample identities")
    generated_paths = {Path(artifact.artifact_path).resolve() for artifact in comp_artifacts}
    if any(not path.is_file() for path in generated_paths):
        raise FileNotFoundError("KID input contains a missing generated sample artifact")
    generated_directories = {path.parent for path in generated_paths}
    if len(generated_directories) != 1:
        raise ValueError("KID artifacts must be stored in one canonical directory")
    generated_directory = next(iter(generated_directories))
    if {
        path.resolve() for path in generated_directory.iterdir() if path.is_file()
    } != generated_paths:
        raise ValueError("KID directory contains extra or missing sample artifacts")
    if not real_directory.is_dir():
        raise FileNotFoundError(f"KID real COMP directory is missing: {real_directory}")
    try:
        installed_version = importlib.metadata.version("clean-fid")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("clean-fid 0.1.35 is required for measured KID") from error
    if installed_version != CLEAN_FID_VERSION:
        raise RuntimeError(
            f"clean-fid must equal {CLEAN_FID_VERSION}, got {installed_version}"
        )
    try:
        from cleanfid import fid
    except ImportError as error:
        raise RuntimeError("clean-fid 0.1.35 is required for measured KID") from error
    real_count = len([path for path in real_directory.iterdir() if path.is_file()])
    generated_count = len(generated_paths)
    subset_size = min(1000, real_count, generated_count)
    if real_count != 237 or generated_count != 237 or subset_size != 237:
        raise ValueError("COMP KID requires exactly 237 real and generated samples")
    model = fid.build_feature_extractor(
        mode="clean", device=device, use_dataparallel=False
    )
    real_features = fid.get_folder_features(
        str(real_directory), model=model, num_workers=0, batch_size=batch_size,
        device=device, mode="clean", verbose=False,
    )
    generated_features = fid.get_folder_features(
        str(generated_directory), model=model, num_workers=0, batch_size=batch_size,
        device=device, mode="clean", verbose=False,
    )
    caller_state = np.random.get_state()
    try:
        np.random.seed(KID_SUBSET_SEED)
        result = fid.kernel_distance(
            real_features,
            generated_features,
            num_subsets=KID_NUM_SUBSETS,
            max_subset_size=subset_size,
        )
    finally:
        np.random.set_state(caller_state)
    return {
        "status": "COMPUTED",
        "value": float(result),
        "clean_fid_version": CLEAN_FID_VERSION,
        "mode": "clean",
        "feature_extractor": "standard Inception-v3",
        "num_subsets": KID_NUM_SUBSETS,
        "subset_size": subset_size,
        "subset_seed": KID_SUBSET_SEED,
        "sampling": "without_replacement_within_subset",
        "sample_keys": [artifact.identity.as_key() for artifact in comp_artifacts],
    }


def build_measured_selection_result(
    *,
    candidate: SelectionCandidate,
    spec: CA16RunSpec,
    checkpoint: Mapping[str, Any],
    conditioning: Mapping[str, Any],
    artifacts: Sequence[SampleArtifact],
    evaluator: Mapping[str, str],
    selected: bool,
    ranking: Mapping[str, Any],
    kid: Mapping[str, Any] | None = None,
) -> SelectionResult:
    identity = checkpoint.get("experiment")
    if not isinstance(identity, Mapping):
        raise TypeError("measured result requires checkpoint experiment provenance")
    if identity.get("git_commit") is None or not identity.get("branch"):
        raise ValueError("measured result requires checkpoint execution Git provenance")
    if checkpoint.get("schema") != CA16_CHECKPOINT_SCHEMA:
        raise ValueError("measured result requires a CA16 checkpoint schema")
    expected_identity = spec.provenance(
        execution_git_sha=str(identity["git_commit"]),
        branch=str(identity["branch"]),
    )
    if any(identity.get(key) != expected_identity.get(key) for key in (
        "experiment_id", "run_id", "architecture_arm",
        "use_decoder16_cross_attention", "training_seed", "frozen_training", "manifests",
    )):
        raise ValueError("measured result checkpoint identity does not match run spec")
    checkpoint_path = checkpoint.get("path", checkpoint.get("checkpoint_path"))
    checkpoint_sha256 = checkpoint.get("sha256", checkpoint.get("checkpoint_sha256"))
    if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_sha256, str):
        raise TypeError("measured result checkpoint path/hash are required")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"measured result checkpoint is missing: {checkpoint_file}")
    if sha256_file(checkpoint_file) != checkpoint_sha256:
        raise ValueError("measured result checkpoint hash does not match file")
    if (
        candidate.checkpoint_path is not None
        and Path(candidate.checkpoint_path).resolve() != checkpoint_file.resolve()
    ):
        raise ValueError("candidate checkpoint path disagrees with checkpoint provenance")
    if candidate.checkpoint_sha256 is not None and candidate.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("candidate checkpoint hash disagrees with checkpoint provenance")
    if checkpoint.get("progress", {}).get("step") != candidate.milestone_step:
        raise ValueError("measured result checkpoint milestone is inconsistent")
    if checkpoint.get("ema_state_dict") is None:
        raise ValueError("measured result requires EMA checkpoint state")
    _validate_measured_evaluator_identity(evaluator)
    if not artifacts:
        raise ValueError("measured result requires per-record sample artifacts")
    validated_artifacts = validate_selection_artifacts(artifacts, spec, candidate.milestone_step)
    if sum(item.identity.split_role == "ID_VAL" for item in validated_artifacts) != 256:
        raise ValueError("measured result requires 256 ID references")
    if sum(item.identity.split_role == "COMP_VAL" for item in validated_artifacts) != 237:
        raise ValueError("measured result requires 237 COMP references")
    _validate_conditioning_metrics(conditioning)
    if (
        set(FROZEN_GENERATION) != {
            "timesteps", "pixel_space_ddpm", "clip_denoised",
            "cfg", "conditioning_dropout", "sampler", "weight_set",
        }
        or FROZEN_GENERATION["sampler"] != "pixel_ddpm_ancestral"
        or FROZEN_GENERATION["weight_set"] != "EMA"
    ):
        raise ValueError("frozen generation metadata is incomplete")
    _validate_ranking_metadata(ranking, candidate)
    ranking_candidates = [
        SelectionCandidate(
            spec.run_id,
            spec.architecture_arm.value,
            spec.training_seed,
            int(row["milestone_step"]),
            float(row["comp_family_macro_maa"]),
            float(row["comp_macro_psr"]),
            float(row["id_maa"]),
            None if row["kid_status"] == "NOT_REQUIRED" else float(row["comp_kid"]),
        )
        for row in ranking["candidates"]
    ]
    if select_final_candidate(ranking_candidates).milestone_step != candidate.milestone_step:
        raise ValueError("ranking criteria do not select the reported milestone")
    if kid is None:
        kid = {"status": "NOT_REQUIRED", "value": None}
    if kid.get("status") == "COMPUTED" and (
        not isinstance(kid.get("value"), (int, float))
        or not math.isfinite(float(kid["value"]))
    ):
        raise ValueError("computed KID must include a value")
    if kid.get("status") == "NOT_REQUIRED" and kid.get("value") is not None:
        raise ValueError("NOT_REQUIRED KID must not include a value")
    if kid.get("status") not in {"NOT_REQUIRED", "COMPUTED"}:
        raise ValueError("unknown KID status")
    per_record = tuple(
        {
            "sample_identity": artifact.identity.as_key(),
            "artifact_path": artifact.artifact_path,
            "ground_truth": dict(artifact.ground_truth),
            "predictions": dict(artifact.predictions),
            "correctness": dict(artifact.correctness),
        }
        for artifact in artifacts
    )
    return SelectionResult(
        experiment_id="ca16_ablation",
        run_id=spec.run_id,
        architecture_arm=spec.architecture_arm.value,
        training_seed=spec.training_seed,
        milestone_step=candidate.milestone_step,
        checkpoint_path=str(checkpoint_file),
        checkpoint_sha256=checkpoint_sha256,
        ema=True,
        selection_manifest_path=str(SELECTION_MANIFEST),
        selection_manifest_sha256=SELECTION_MANIFEST_SHA256,
        evaluator=evaluator,
        generation=dict(FROZEN_GENERATION),
        conditioning=dict(conditioning),
        kid=kid.get("value"),
        kid_status=str(kid["status"]),
        selected=selected,
        checkpoint_schema=CA16_CHECKPOINT_SCHEMA,
        execution_git_sha=str(identity.get("git_commit", "")),
        execution_branch=identity.get("branch"),
        ranking=dict(ranking),
        per_record=per_record,
    )
