"""Fail-closed, resumable orchestration for frozen CA16 R11 selection.

Scientific primitives remain in :mod:`ca16_evaluation`; this module binds their
inputs and outputs to durable, hash-verified artifacts.  Importing it has no
filesystem, model-loading, evaluator, sampling, or KID side effects.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import socket
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_evaluation import (
    ATTRIBUTES,
    CANDIDATE_MILESTONES,
    CLEAN_FID_VERSION,
    EVALUATOR_CHECKPOINT_RELATIVE,
    EVALUATOR_CHECKPOINT_SHA256,
    EVALUATOR_FREEZE,
    EVALUATOR_FREEZE_SHA256,
    EXPECTED_SELECTION_FAMILY_COUNTS,
    FROZEN_GENERATION,
    QUALITY_MANIFEST,
    QUALITY_MANIFEST_SHA256,
    ROBUST_MANIFEST,
    ROBUST_MANIFEST_SHA256,
    SELECTION_MANIFEST,
    SELECTION_MANIFEST_SHA256,
    SampleArtifact,
    SelectionCandidate,
    attribute_metrics,
    compute_comp_kid_from_artifacts,
    family_macro_metrics,
    load_checkpoint_selection_manifest,
    load_ema_checkpoint,
    load_frozen_evaluator,
    predict_artifact_attributes,
    rank_candidates,
    sample_ema_record,
    sample_ema_records_batch,
    sha256_file,
    stable_sample_identity,
)
from tiny_avatar_diffusion.ca16_experiment import ROOT, CA16RunSpec, ca16_run_matrix
from tiny_avatar_diffusion.conditional_main_training import CA16_CHECKPOINT_SCHEMA
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule

PROTOCOL_ID = "ca16_r11_checkpoint_selection_v1"
BATCHED_PROTOCOL_ID = "ca16_r11_checkpoint_selection_batched_v1"
OUTPUT_SCHEMA = "tiny_avatar_diffusion.ca16_r11_output_v1"
TRAINING_GIT_SHA = "48583669097560ed308ab8d2d5e7104173cc99b4"
INVENTORY_PATH = ROOT / "data/processed/ca16_eval_v1/r11_checkpoint_inventory.json"
RUN_ORDER = tuple(spec.run_id for spec in ca16_run_matrix())
MILESTONE_ORDER = tuple(CANDIDATE_MILESTONES)
ALLOWED_ROLES = ("ID_VAL", "COMP_VAL")
EXPECTED_COUNTS = {"ID_VAL": 256, "COMP_VAL": 237}
TOTAL_SAMPLES = 493
LOCK_NAME = ".r11.lock"
TOKEN_SOURCE_PATHS = {
    role: ROOT
    / "data"
    / "processed"
    / "dr01_v1"
    / "text"
    / "10k"
    / f"{role.lower()}.jsonl"
    for role in ALLOWED_ROLES
}


class OutputState(str, Enum):
    NEW = "NEW"
    RESUMABLE_VALID = "RESUMABLE_VALID"
    INVALID_OR_UNPROVEN = "INVALID_OR_UNPROVEN"


@dataclass(frozen=True)
class GitState:
    head: str
    branch: str
    tracked_clean: bool


@dataclass(frozen=True)
class CandidatePaths:
    output_root: Path
    run_id: str
    milestone: int

    @property
    def root(self) -> Path:
        return self.output_root / "candidates" / self.run_id / f"{self.milestone:05d}"

    @property
    def sample_manifest(self) -> Path:
        return self.root / "sample_manifest.jsonl"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "candidate_metrics.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _operational_temp(path: Path) -> Path:
    """Return the single, deterministic staging name for an atomic artifact."""
    return path.with_name(f".{path.name}.operational.tmp")


def atomic_json(
    path: Path, payload: Any, *, replace: Callable[[Path, Path], None] = os.replace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _operational_temp(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _operational_temp(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        for attempt, delay in enumerate((0.05, 0.15, 0.45), start=1):
            try:
                replace(temporary, path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) != 5 or attempt == 3:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def git_state(*, root: Path = ROOT) -> GitState:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    return GitState(
        head=run("rev-parse", "HEAD"),
        branch=run("branch", "--show-current"),
        # Normal Git status semantics include staged, modified, and untracked
        # non-ignored files while continuing to exclude ignored files.
        tracked_clean=run("status", "--porcelain", "--untracked-files=normal") == "",
    )


def verify_evaluation_git(
    expected_sha: str,
    state: GitState,
    *,
    expected_tag: str | None = None,
    root: Path = ROOT,
) -> None:
    if not expected_sha or state.head != expected_sha:
        raise RuntimeError(
            f"evaluation HEAD mismatch: expected {expected_sha}, got {state.head}"
        )
    if not state.tracked_clean:
        raise RuntimeError(
            "repository must be clean (including staged and untracked non-ignored files) for scientific execution"
        )
    if state.branch != "exp/ca16-ablation":
        raise RuntimeError(f"unexpected evaluation branch: {state.branch}")
    if expected_tag is not None:
        tagged = subprocess.run(
            ["git", "rev-parse", f"{expected_tag}^{{commit}}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if tagged != expected_sha:
            raise RuntimeError("evaluation freeze tag does not identify expected HEAD")


def expected_plan() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (spec.run_id, spec.architecture_arm.value, spec.training_seed, milestone)
        for spec in ca16_run_matrix()
        for milestone in MILESTONE_ORDER
    )


def load_checkpoint_inventory(
    path: Path = INVENTORY_PATH, *, root: Path = ROOT, verify_files: bool = True
) -> dict[str, Any]:
    payload = read_json(path)
    if (
        payload.get("schema")
        != "tiny_avatar_diffusion.ca16_r11_checkpoint_inventory_v1"
    ):
        raise ValueError("unexpected R11 checkpoint inventory schema")
    if payload.get("expected_training_git_sha") != TRAINING_GIT_SHA:
        raise ValueError("inventory training Git SHA is not frozen R10 SHA")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("R11 inventory must contain exactly 24 checkpoints")
    observed: list[tuple[str, str, int, int]] = []
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("checkpoint inventory entry must be an object")
        observed.append(
            (
                entry.get("run_id"),
                entry.get("architecture_arm"),
                entry.get("training_seed"),
                entry.get("milestone"),
            )
        )
        relative = entry.get("checkpoint_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or relative in seen_paths
        ):
            raise ValueError("checkpoint inventory paths must be unique relative paths")
        seen_paths.add(relative)
        if (
            entry.get("checkpoint_schema") != CA16_CHECKPOINT_SCHEMA
            or entry.get("training_git_sha") != TRAINING_GIT_SHA
        ):
            raise ValueError("checkpoint inventory provenance mismatch")
        if verify_files:
            checkpoint = root / relative
            if not checkpoint.is_file() or sha256_file(checkpoint) != entry.get(
                "sha256"
            ):
                raise ValueError(f"checkpoint inventory hash mismatch: {relative}")
    if tuple(observed) != expected_plan():
        raise ValueError("checkpoint inventory does not match canonical 6x4 plan order")
    return payload


def inventory_entry(
    inventory: Mapping[str, Any], spec: CA16RunSpec, milestone: int
) -> dict[str, Any]:
    matches = [
        entry
        for entry in inventory["entries"]
        if entry["run_id"] == spec.run_id and entry["milestone"] == milestone
    ]
    if len(matches) != 1:
        raise ValueError("candidate has no unique checkpoint inventory entry")
    return dict(matches[0])


def canonical_records() -> tuple[dict[str, Any], ...]:
    manifest = load_checkpoint_selection_manifest()
    records = tuple(
        dict(row) for row in manifest["id_entries"] + manifest["comp_entries"]
    )
    counts = {
        role: sum(row.get("split_role") == role for row in records)
        for role in ALLOWED_ROLES
    }
    if counts != EXPECTED_COUNTS or len(records) != TOTAL_SAMPLES:
        raise ValueError("selection panel is not the frozen 256/237 contract")
    if any(row.get("split_role") not in ALLOWED_ROLES for row in records):
        raise ValueError("selection panel contains forbidden role")
    if len({row["stable_record_key"] for row in records}) != TOTAL_SAMPLES:
        raise ValueError("selection panel has duplicate canonical records")
    return records


def _index_tokenized_source_records(
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for role in ALLOWED_ROLES:
        for record in records_by_role.get(role, ()):
            if record.get("split_role") != role:
                raise ValueError(
                    "token-source record split differs from its authorized source"
                )
            key = "|".join(
                str(record.get(field))
                for field in ("dataset_variant", "split_role", "image_id")
            )
            if key in indexed:
                raise ValueError("duplicate token-source stable_record_key")
            indexed[key] = dict(record)
    return indexed


def load_tokenized_source_index() -> dict[str, dict[str, Any]]:
    expected_hashes = load_checkpoint_selection_manifest()["source_artifact_sha256"]
    for role, path in TOKEN_SOURCE_PATHS.items():
        if not path.is_file() or sha256_file(path) != expected_hashes.get(role):
            raise ValueError(f"frozen token-source hash mismatch: {role}")
    return _index_tokenized_source_records(
        {role: read_jsonl(path) for role, path in TOKEN_SOURCE_PATHS.items()}
    )


def hydrate_selection_record(
    record: Mapping[str, Any], source_by_key: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    role = record.get("split_role")
    if role not in ALLOWED_ROLES:
        raise ValueError("selection record has a forbidden split role")
    key = str(record.get("stable_record_key"))
    source = source_by_key.get(key)
    if source is None:
        raise ValueError("selection record has no matching token-source record")
    for field in ("dataset_variant", "split_role", "image_id", "caption"):
        if source.get(field) != record.get(field):
            raise ValueError(f"token-source {field} differs from selection record")
    for field in ("token_ids", "attention_mask"):
        if field not in source:
            raise ValueError(f"token-source record lacks {field}")
    return {
        **dict(record),
        "token_ids": source["token_ids"],
        "attention_mask": source["attention_mask"],
    }


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "clean_fid": importlib.metadata.version("clean-fid"),
    }


def build_run_manifest(
    *,
    evaluation: GitState,
    inventory_sha256: str,
    environment: Mapping[str, Any],
    timestamp: str | None = None,
    protocol_id: str = PROTOCOL_ID,
    sampling_batch_size: int = 1,
) -> dict[str, Any]:
    if environment.get("clean_fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")
    if protocol_id not in (PROTOCOL_ID, BATCHED_PROTOCOL_ID):
        raise ValueError("unknown R11 execution protocol")
    if sampling_batch_size < 1:
        raise ValueError("sampling batch size must be positive")
    if protocol_id == PROTOCOL_ID and sampling_batch_size != 1:
        raise ValueError("legacy R11 protocol requires sampling batch size one")
    if protocol_id == BATCHED_PROTOCOL_ID and sampling_batch_size == 1:
        raise ValueError("batched R11 protocol requires sampling batch size above one")
    return {
        "schema": OUTPUT_SCHEMA,
        "experiment_id": "ca16_ablation",
        "protocol_id": protocol_id,
        "execution": {
            "profile": protocol_id,
            "sampling_batch_size": sampling_batch_size,
            "rng_policy": "per_record_cuda_generator_legacy_shape_then_concatenate",
        },
        "evaluation_git_sha": evaluation.head,
        "expected_training_git_sha": TRAINING_GIT_SHA,
        "branch": evaluation.branch,
        "tracked_working_tree": "clean" if evaluation.tracked_clean else "dirty",
        "environment": dict(environment),
        "selection_manifest": {
            "path": str(SELECTION_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": SELECTION_MANIFEST_SHA256,
        },
        "robust_manifest": {
            "path": str(ROBUST_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": ROBUST_MANIFEST_SHA256,
        },
        "quality_manifest": {
            "path": str(QUALITY_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": QUALITY_MANIFEST_SHA256,
        },
        "evaluator_freeze_sha256": EVALUATOR_FREEZE_SHA256,
        "evaluator_checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
        "checkpoint_inventory": {
            "path": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": inventory_sha256,
        },
        "run_order": list(RUN_ORDER),
        "milestone_order": list(MILESTONE_ORDER),
        "selection_counts": dict(EXPECTED_COUNTS),
        "generation_seed_policy": "frozen selection manifest: ID and COMP seed=2026+i within split",
        "generation": dict(FROZEN_GENERATION),
        "timestamp_utc": timestamp or utc_now(),
    }


def _known_operational_target(output_root: Path, temporary: Path) -> Path | None:
    prefix = "."
    suffix = ".operational.tmp"
    if not temporary.name.startswith(prefix) or not temporary.name.endswith(suffix):
        return None
    target = temporary.with_name(temporary.name[len(prefix) : -len(suffix)])
    try:
        relative = target.relative_to(output_root)
    except ValueError:
        return None
    parts = relative.parts
    top_level = {
        "run_manifest.json",
        "progress.json",
        "real_comp_manifest.jsonl",
        "r11_summary.json",
        "R11_COMPLETE.json",
    }
    if len(parts) == 1 and parts[0] in top_level:
        return target
    if len(parts) == 2 and parts[0] == "selections" and target.suffix == ".json":
        return target
    if (
        len(parts) == 4
        and parts[0] == "candidates"
        and parts[3]
        in {
            "sample_manifest.jsonl",
            "predictions.jsonl",
            "candidate_metrics.json",
        }
    ):
        return target
    if (
        len(parts) == 5
        and parts[0] == "candidates"
        and parts[3] == "transactions"
        and target.name.endswith(".transaction.json")
    ):
        return target
    return None


def _validate_operational_payload(path: Path, target: Path) -> None:
    if target.suffix == ".jsonl":
        read_jsonl(path)
        return
    payload = read_json(path)
    expected_schemas = {
        "run_manifest.json": OUTPUT_SCHEMA,
        "progress.json": "tiny_avatar_diffusion.ca16_r11_progress_v1",
        "candidate_metrics.json": "tiny_avatar_diffusion.ca16_r11_candidate_metrics_v1",
        "r11_summary.json": "tiny_avatar_diffusion.ca16_r11_summary_v1",
        "R11_COMPLETE.json": "tiny_avatar_diffusion.ca16_r11_complete_v1",
    }
    expected = expected_schemas.get(target.name)
    if target.name.endswith(".transaction.json"):
        expected = "tiny_avatar_diffusion.ca16_r11_sample_transaction_v1"
    if target.parent.name == "selections":
        expected = "tiny_avatar_diffusion.ca16_r11_run_selection_v1"
    if expected is not None and payload.get("schema") != expected:
        raise ValueError(f"operational temporary has wrong schema for {target}")


def recover_operational_temps(output_root: Path, *, mutate: bool) -> None:
    if not output_root.exists():
        return
    for temporary in output_root.rglob("*.tmp"):
        target = _known_operational_target(output_root, temporary)
        if target is None:
            raise ValueError(f"unknown temporary file: {temporary}")
        _validate_operational_payload(temporary, target)
        if target.exists():
            _validate_operational_payload(target, target)
            if mutate:
                temporary.unlink()
        elif mutate:
            os.replace(temporary, target)


def _transaction_generated_references(output_root: Path) -> dict[Path, dict[str, Any]]:
    references: dict[Path, dict[str, Any]] = {}
    candidate_root = output_root / "candidates"
    if not candidate_root.exists():
        return references
    known: dict[Path, tuple[Path, CA16RunSpec, int, int, Mapping[str, Any]]] = {}
    records = canonical_records()
    for spec in ca16_run_matrix():
        for milestone in MILESTONE_ORDER:
            for index, record in enumerate(records):
                transaction, temporary = _transaction_paths(
                    output_root,
                    spec,
                    milestone,
                    index,
                    str(record["stable_record_key"]),
                )
                known[transaction.resolve()] = (
                    temporary.resolve(),
                    spec,
                    milestone,
                    index,
                    record,
                )
    known_temporaries = {entry[0] for entry in known.values()}
    for path in candidate_root.glob("*/*/transactions/*"):
        if not path.is_file():
            raise ValueError("sample transaction directory contains a non-file")
        if path.name.endswith(".operational.tmp"):
            if _known_operational_target(output_root, path) is None:
                raise ValueError("unknown transaction operational temporary")
            continue
        resolved = path.resolve()
        if path.name.endswith(".png.pending"):
            if resolved not in known_temporaries:
                raise ValueError("unknown sample transaction temporary")
            continue
        if resolved not in known:
            raise ValueError("unknown sample transaction record")
    for path in candidate_root.glob("*/*/transactions/*.transaction.json"):
        transaction = read_json(path)
        if (
            transaction.get("schema")
            != "tiny_avatar_diffusion.ca16_r11_sample_transaction_v1"
        ):
            raise ValueError("unknown sample transaction schema")
        temporary_expected, spec, milestone, index, record = known[path.resolve()]
        sample = transaction.get("sample")
        if not isinstance(sample, dict):
            raise TypeError("sample transaction has no canonical sample identity")
        expected = expected_sample_record(
            output_root=output_root,
            spec=spec,
            milestone=milestone,
            checkpoint_sha256=str(sample.get("checkpoint_sha256")),
            canonical_record=record,
            sample_index=index,
        )
        final = output_root / str(transaction.get("final_relative_path"))
        temporary = output_root / str(transaction.get("temporary_relative_path"))
        for item in (final, temporary):
            if (
                item.resolve() == output_root.resolve()
                or output_root.resolve() not in item.resolve().parents
            ):
                raise ValueError("sample transaction path escapes output root")
        digest = transaction.get("png_sha256")
        if (
            not isinstance(digest, str)
            or temporary.resolve() != temporary_expected
            or transaction
            != _expected_transaction(output_root, expected, path, temporary, digest)
        ):
            raise ValueError("sample transaction does not match canonical identity")
        if temporary.exists():
            _validate_png(temporary)
            if sha256_file(temporary) != digest:
                raise ValueError("sample transaction temporary hash mismatch")
        if final.exists():
            _validate_png(final)
            if sha256_file(final) != digest:
                raise ValueError("sample transaction final hash mismatch")
            references[final.resolve()] = {"png_sha256": digest}
    return references


def classify_output_root(
    output_root: Path, expected_manifest: Mapping[str, Any] | None = None
) -> OutputState:
    if not output_root.exists():
        return OutputState.NEW
    if not output_root.is_dir():
        return OutputState.INVALID_OR_UNPROVEN
    entries = [path for path in output_root.iterdir() if path.name != LOCK_NAME]
    if not entries:
        return OutputState.NEW
    try:
        recover_operational_temps(output_root, mutate=False)
        manifest_path = output_root / "run_manifest.json"
        staged_manifest = _operational_temp(manifest_path)
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
        elif staged_manifest.is_file():
            manifest = read_json(staged_manifest)
        else:
            return OutputState.INVALID_OR_UNPROVEN
    except (OSError, ValueError, json.JSONDecodeError):
        return OutputState.INVALID_OR_UNPROVEN
    if manifest.get("schema") != OUTPUT_SCHEMA:
        return OutputState.INVALID_OR_UNPROVEN
    if expected_manifest is not None and manifest != dict(expected_manifest):
        return OutputState.INVALID_OR_UNPROVEN
    allowed = {
        "run_manifest.json",
        "progress.json",
        "r11.log",
        LOCK_NAME,
        "generated",
        "candidates",
        "selections",
        "real_comp",
        "real_comp_manifest.jsonl",
        "r11_summary.json",
        "R11_COMPLETE.json",
    }
    if any(
        path.name not in allowed
        and _known_operational_target(output_root, path) is None
        for path in output_root.iterdir()
    ):
        return OutputState.INVALID_OR_UNPROVEN
    try:
        transaction_references = _transaction_generated_references(output_root)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return OutputState.INVALID_OR_UNPROVEN
    generated = output_root / "generated"
    if generated.exists():
        try:
            referenced = {
                (output_root / row["png_relative_path"]).resolve(): row
                for manifest_path in (output_root / "candidates").glob(
                    "*/*/sample_manifest.jsonl"
                )
                for row in read_jsonl(manifest_path)
            }
            referenced.update(transaction_references)
            actual = {path.resolve() for path in generated.rglob("*") if path.is_file()}
            if actual != set(referenced):
                return OutputState.INVALID_OR_UNPROVEN
            for path in actual:
                _validate_png(path)
                if sha256_file(path) != referenced[path].get("png_sha256"):
                    return OutputState.INVALID_OR_UNPROVEN
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return OutputState.INVALID_OR_UNPROVEN
    return OutputState.RESUMABLE_VALID


def initialize_or_resume(
    output_root: Path, expected_manifest: Mapping[str, Any]
) -> OutputState:
    state = classify_output_root(output_root, expected_manifest)
    if state is OutputState.INVALID_OR_UNPROVEN:
        raise RuntimeError(
            "output root is INVALID_OR_UNPROVEN; refusing adoption or repair"
        )
    if state is OutputState.NEW:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / "run_manifest.json", expected_manifest)
    else:
        recover_operational_temps(output_root, mutate=True)
    return state


@contextmanager
def exclusive_run_lock(output_root: Path) -> Iterator[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / LOCK_NAME
    payload = {
        "schema": "ca16_r11_lock_v1",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_utc": utc_now(),
        "token": uuid.uuid4().hex,
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(
            "R11 output lock exists; live or stale locks require explicit forensic resolution"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield payload
    finally:
        try:
            current = read_json(path)
            if current.get("token") == payload["token"]:
                path.unlink()
        except FileNotFoundError:
            pass


def configure_logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger(f"ca16_r11.{output_root.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(output_root / "r11.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    return logger


def candidate_paths(
    output_root: Path, spec: CA16RunSpec, milestone: int
) -> CandidatePaths:
    if milestone not in MILESTONE_ORDER:
        raise ValueError("candidate milestone is not frozen")
    return CandidatePaths(output_root, spec.run_id, milestone)


def _sample_filename(index: int, stable_record_key: str) -> str:
    digest = hashlib.sha256(stable_record_key.encode("utf-8")).hexdigest()[:16]
    return f"{index:03d}_{digest}.png"


def expected_sample_record(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    canonical_record: Mapping[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    role = canonical_record.get("split_role")
    if role not in ALLOWED_ROLES:
        raise ValueError(f"R11 rejects forbidden split role: {role}")
    identity = stable_sample_identity(spec, milestone, canonical_record)
    relative = (
        Path("generated")
        / spec.run_id
        / f"{milestone:05d}"
        / str(role)
        / _sample_filename(sample_index, identity.record_id)
    )
    return {
        "schema": "tiny_avatar_diffusion.ca16_r11_sample_v1",
        "sample_index": sample_index,
        "sample_identity": identity.as_key(),
        "run_id": spec.run_id,
        "architecture_arm": spec.architecture_arm.value,
        "training_seed": spec.training_seed,
        "milestone": milestone,
        "checkpoint_sha256": checkpoint_sha256,
        "split_role": role,
        "image_id": canonical_record["image_id"],
        "stable_record_key": canonical_record["stable_record_key"],
        "generation_seed": canonical_record["generation_seed"],
        "ema": True,
        "png_relative_path": relative.as_posix(),
    }


def _validate_png(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty scientific PNG: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("scientific PNG must be readable RGB 32x32")


def validate_sample_record(
    output_root: Path, row: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    required = {**dict(expected), "png_sha256": row.get("png_sha256")}
    if not isinstance(required["png_sha256"], str) or any(
        row.get(key) != value for key, value in required.items()
    ):
        raise ValueError(
            "persisted sample provenance does not match canonical identity"
        )
    relative = Path(str(row["png_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sample PNG path must stay inside output root")
    path = output_root / relative
    _validate_png(path)
    if sha256_file(path) != row["png_sha256"]:
        raise ValueError("persisted PNG SHA256 mismatch")
    return dict(row)


def tensor_to_pixels(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.shape != (3, 32, 32):
            raise ValueError("generator tensor must have shape [3,32,32]")
        value = (
            ((value.detach().cpu().to(torch.float32).clamp(-1, 1) + 1) * 127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
    pixels = np.ascontiguousarray(value, dtype=np.uint8)
    if pixels.shape != (32, 32, 3):
        raise ValueError("generator output must be RGB uint8 32x32")
    return pixels


def persist_new_png(path: Path, pixels: Any) -> str:
    if path.exists():
        raise FileExistsError(
            "scientific PNG already exists and will not be overwritten"
        )
    array = tensor_to_pixels(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
        _validate_png(temporary)
        # A hard-link finalization is atomic on the same volume and, unlike
        # os.replace, fails if another writer has created the final name.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _transaction_paths(
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    sample_index: int,
    stable_record_key: str,
) -> tuple[Path, Path]:
    token = _sample_filename(sample_index, stable_record_key).removesuffix(".png")
    directory = candidate_paths(output_root, spec, milestone).root / "transactions"
    return directory / f"{token}.transaction.json", directory / f"{token}.png.pending"


def _write_pending_png(path: Path, pixels: Any) -> str:
    if path.exists():
        raise FileExistsError("sample transaction temporary already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_pixels(pixels), mode="RGB").save(path, format="PNG")
    _validate_png(path)
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
    return sha256_file(path)


def _expected_transaction(
    output_root: Path,
    expected: Mapping[str, Any],
    transaction_path: Path,
    temporary_path: Path,
    png_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "tiny_avatar_diffusion.ca16_r11_sample_transaction_v1",
        "transaction_id": transaction_path.stem,
        "sample": dict(expected),
        "temporary_relative_path": temporary_path.relative_to(output_root).as_posix(),
        "final_relative_path": expected["png_relative_path"],
        "png_sha256": png_sha256,
    }


def _recover_sample_transaction(
    *,
    output_root: Path,
    expected: Mapping[str, Any],
    transaction_path: Path,
    temporary_path: Path,
    sample_manifest: Path,
    rows: list[dict[str, Any]],
    after_boundary: Callable[[str], None] | None = None,
) -> bool:
    """Recover one persisted sample transaction; return whether it completed work."""
    key = str(expected["stable_record_key"])
    existing = next((row for row in rows if row.get("stable_record_key") == key), None)
    final_path = output_root / str(expected["png_relative_path"])
    if not transaction_path.exists():
        if temporary_path.exists():
            if final_path.exists() or existing is not None:
                raise ValueError(
                    "unproven sample transaction temporary conflicts with authoritative state"
                )
            # The process died before transaction provenance became durable.  This
            # deterministic staging file is safe to discard and regenerate.
            temporary_path.unlink()
        if final_path.exists() and existing is None:
            raise ValueError(
                "orphan scientific PNG exists without canonical transaction provenance"
            )
        return False

    transaction = read_json(transaction_path)
    digest = transaction.get("png_sha256")
    if not isinstance(digest, str) or transaction != _expected_transaction(
        output_root,
        expected,
        transaction_path,
        temporary_path,
        digest,
    ):
        raise ValueError("sample transaction identity or provenance mismatch")
    if not temporary_path.exists() and not final_path.exists():
        raise ValueError("sample transaction has neither temporary nor final PNG")
    if temporary_path.exists():
        _validate_png(temporary_path)
        if sha256_file(temporary_path) != digest:
            raise ValueError("sample transaction temporary PNG SHA256 mismatch")
    if final_path.exists():
        _validate_png(final_path)
        if sha256_file(final_path) != digest:
            raise ValueError("sample transaction final PNG SHA256 mismatch")
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(temporary_path, final_path)  # atomic no-clobber publication
        _validate_png(final_path)
        if sha256_file(final_path) != digest:
            raise ValueError("finalized sample PNG SHA256 mismatch")
        if after_boundary is not None:
            after_boundary("FINALIZED")

    authoritative = {**dict(expected), "png_sha256": digest}
    if existing is None:
        rows.append(authoritative)
        atomic_jsonl(sample_manifest, rows)
        if after_boundary is not None:
            after_boundary("MANIFEST_PERSISTED")
    elif dict(existing) != authoritative:
        raise ValueError("sample transaction conflicts with authoritative manifest")
    validate_sample_record(output_root, authoritative, expected)
    temporary_path.unlink(missing_ok=True)
    if after_boundary is not None:
        after_boundary("TEMP_CLEANED")
    # Remove the authoritative transaction last.  A crash after temporary
    # cleanup therefore leaves recoverable CASE C (transaction + final + row),
    # never an unproven temporary without its transaction record.
    transaction_path.unlink()
    return True


def _validate_candidate_samples(
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    canonical: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    canonical = tuple(canonical)
    paths = candidate_paths(output_root, spec, milestone)
    rows = read_jsonl(paths.sample_manifest)
    keys = [row.get("stable_record_key") for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate canonical record in sample manifest")
    expected_by_key = {
        row["stable_record_key"]: expected_sample_record(
            output_root=output_root,
            spec=spec,
            milestone=milestone,
            checkpoint_sha256=checkpoint_sha256,
            canonical_record=row,
            sample_index=index,
        )
        for index, row in enumerate(canonical)
    }
    if any(key not in expected_by_key for key in keys):
        raise ValueError("sample manifest contains a noncanonical record")
    if keys != [row["stable_record_key"] for row in canonical[: len(keys)]]:
        raise ValueError("sample manifest is not a canonical ordered prefix")
    validated = [
        validate_sample_record(
            output_root, row, expected_by_key[str(row["stable_record_key"])]
        )
        for row in rows
    ]
    generated_root = output_root / "generated" / spec.run_id / f"{milestone:05d}"
    actual_pngs = (
        {path.resolve() for path in generated_root.rglob("*") if path.is_file()}
        if generated_root.exists()
        else set()
    )
    provenanced_pngs = {
        (output_root / row["png_relative_path"]).resolve() for row in validated
    }
    if actual_pngs != provenanced_pngs:
        raise ValueError("candidate contains orphan or missing PNG files")
    if require_complete:
        if len(validated) != TOTAL_SAMPLES or set(keys) != set(expected_by_key):
            raise ValueError(
                "candidate is incomplete; exactly 493 canonical samples are required"
            )
        counts = {
            role: sum(row["split_role"] == role for row in validated)
            for role in ALLOWED_ROLES
        }
        if counts != EXPECTED_COUNTS:
            raise ValueError("candidate split counts are not 256/237")
    return validated


def validate_candidate_samples(
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    return _validate_candidate_samples(
        output_root,
        spec,
        milestone,
        checkpoint_sha256,
        canonical_records(),
        require_complete=require_complete,
    )


def _resume_candidate_generation(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    generate: Callable[[Mapping[str, Any]], Any] | None,
    canonical: Sequence[Mapping[str, Any]],
    on_progress: Callable[[int], None] | None = None,
    after_boundary: Callable[[str], None] | None = None,
    generate_batch: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]]
    | None = None,
    sampling_batch_size: int = 1,
) -> dict[str, Any]:
    if sampling_batch_size < 1:
        raise ValueError("sampling batch size must be positive")
    if generate is None and generate_batch is None:
        raise ValueError("candidate generation requires a generator")
    canonical = tuple(canonical)
    paths = candidate_paths(output_root, spec, milestone)
    rows = read_jsonl(paths.sample_manifest)
    pending: list[tuple[int, Mapping[str, Any], dict[str, Any], Path, Path]] = []
    for index, record in enumerate(canonical):
        expected = expected_sample_record(
            output_root=output_root,
            spec=spec,
            milestone=milestone,
            checkpoint_sha256=checkpoint_sha256,
            canonical_record=record,
            sample_index=index,
        )
        transaction_path, temporary_path = _transaction_paths(
            output_root,
            spec,
            milestone,
            index,
            str(record["stable_record_key"]),
        )
        _recover_sample_transaction(
            output_root=output_root,
            expected=expected,
            transaction_path=transaction_path,
            temporary_path=temporary_path,
            sample_manifest=paths.sample_manifest,
            rows=rows,
            after_boundary=after_boundary,
        )
    transaction_directory = paths.root / "transactions"
    if transaction_directory.exists() and any(transaction_directory.iterdir()):
        raise ValueError(
            "candidate contains an unknown transaction or operational file"
        )
    rows = _validate_candidate_samples(
        output_root,
        spec,
        milestone,
        checkpoint_sha256,
        canonical,
        require_complete=False,
    )
    by_key = {row["stable_record_key"]: row for row in rows}
    generated: list[str] = []
    reused: list[str] = []
    for index, record in enumerate(canonical):
        expected = expected_sample_record(
            output_root=output_root,
            spec=spec,
            milestone=milestone,
            checkpoint_sha256=checkpoint_sha256,
            canonical_record=record,
            sample_index=index,
        )
        key = str(record["stable_record_key"])
        png = output_root / expected["png_relative_path"]
        if key in by_key:
            validate_sample_record(output_root, by_key[key], expected)
            reused.append(key)
            continue
        if png.exists():
            raise RuntimeError(
                "orphan scientific PNG exists without canonical provenance"
            )
        transaction_path, temporary_path = _transaction_paths(
            output_root, spec, milestone, index, key
        )
        pending.append((index, record, expected, transaction_path, temporary_path))
    for start in range(0, len(canonical), sampling_batch_size):
        group = [
            item for item in pending if start <= item[0] < start + sampling_batch_size
        ]
        if not group:
            continue
        original_group = canonical[start : start + sampling_batch_size]
        if generate_batch is None:
            assert generate is not None
            generated_pixels = [generate(record) for _, record, *_ in group]
        else:
            generated_pixels = list(generate_batch(original_group))
        if len(generated_pixels) != (
            len(group) if generate_batch is None else len(original_group)
        ):
            raise ValueError("batch generator returned an unexpected number of samples")
        for offset, record, expected, transaction_path, temporary_path in group:
            pixels = (
                generated_pixels[offset - start]
                if generate_batch is not None
                else generated_pixels[
                    group.index(
                        (offset, record, expected, transaction_path, temporary_path)
                    )
                ]
            )
            digest = _write_pending_png(temporary_path, pixels)
            if after_boundary is not None:
                after_boundary("TEMP_WRITTEN")
            atomic_json(
                transaction_path,
                _expected_transaction(
                    output_root, expected, transaction_path, temporary_path, digest
                ),
            )
            if after_boundary is not None:
                after_boundary("TRANSACTION_PERSISTED")
            _recover_sample_transaction(
                output_root=output_root,
                expected=expected,
                transaction_path=transaction_path,
                temporary_path=temporary_path,
                sample_manifest=paths.sample_manifest,
                rows=rows,
                after_boundary=after_boundary,
            )
            generated.append(str(record["stable_record_key"]))
            if on_progress is not None:
                on_progress(len(rows))
    _validate_candidate_samples(
        output_root, spec, milestone, checkpoint_sha256, canonical
    )
    return {"generated": generated, "reused": reused, "completed": len(rows)}


def resume_candidate_generation(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    generate: Callable[[Mapping[str, Any]], Any],
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Resume generation using only the frozen canonical selection manifest."""
    return _resume_candidate_generation(
        output_root=output_root,
        spec=spec,
        milestone=milestone,
        checkpoint_sha256=checkpoint_sha256,
        generate=generate,
        canonical=canonical_records(),
        on_progress=on_progress,
    )


def resume_candidate_generation_batched(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    generate_batch: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]],
    sampling_batch_size: int,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Resume canonical generation while computing missing records in batches."""
    return _resume_candidate_generation(
        output_root=output_root,
        spec=spec,
        milestone=milestone,
        checkpoint_sha256=checkpoint_sha256,
        generate=None,
        generate_batch=generate_batch,
        sampling_batch_size=sampling_batch_size,
        canonical=canonical_records(),
        on_progress=on_progress,
    )


def expected_prediction_record(
    sample: Mapping[str, Any],
    canonical_record: Mapping[str, Any],
    prediction: Mapping[str, int],
    evaluator_identity: Mapping[str, str],
) -> dict[str, Any]:
    if evaluator_identity.get("checkpoint_sha256") != EVALUATOR_CHECKPOINT_SHA256:
        raise ValueError("prediction evaluator checkpoint hash is not frozen")
    if evaluator_identity.get("freeze_manifest_sha256") != EVALUATOR_FREEZE_SHA256:
        raise ValueError("prediction evaluator freeze hash is not frozen")
    if set(prediction) != set(ATTRIBUTES) or any(
        not isinstance(prediction[name], int) for name in ATTRIBUTES
    ):
        raise ValueError("evaluator prediction must contain four integer attributes")
    return {
        "schema": "tiny_avatar_diffusion.ca16_r11_prediction_v1",
        "sample_index": sample["sample_index"],
        "sample_identity": sample["sample_identity"],
        "stable_record_key": sample["stable_record_key"],
        "png_relative_path": sample["png_relative_path"],
        "png_sha256": sample["png_sha256"],
        "split_role": sample["split_role"],
        "evaluator_checkpoint_sha256": evaluator_identity["checkpoint_sha256"],
        "evaluator_freeze_manifest_sha256": evaluator_identity[
            "freeze_manifest_sha256"
        ],
        "ground_truth": dict(canonical_record["ground_truth_attributes"]),
        "predictions": {name: int(prediction[name]) for name in ATTRIBUTES},
    }


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    evaluator_identity: Mapping[str, str],
    *,
    require_complete: bool,
) -> list[dict[str, Any]]:
    sample_by_key = {row["stable_record_key"]: row for row in samples}
    record_by_key = {row["stable_record_key"]: row for row in records}
    keys = [row.get("stable_record_key") for row in rows]
    if len(keys) != len(set(keys)) or any(key not in sample_by_key for key in keys):
        raise ValueError("prediction rows are duplicate or noncanonical")
    validated: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["stable_record_key"])
        expected = expected_prediction_record(
            sample_by_key[key],
            record_by_key[key],
            row.get("predictions", {}),
            evaluator_identity,
        )
        if dict(row) != expected:
            raise ValueError(
                "prediction is not bound to canonical PNG/evaluator/ground truth"
            )
        validated.append(dict(row))
    if require_complete and (
        len(validated) != TOTAL_SAMPLES or set(keys) != set(sample_by_key)
    ):
        raise ValueError("candidate predictions are incomplete")
    return validated


def _recompute_candidate_metrics(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    evaluator_identity: Mapping[str, str],
    predict: Callable[[Path], Mapping[str, int]],
    canonical: Sequence[Mapping[str, Any]],
    persist_missing: bool = True,
) -> dict[str, Any]:
    canonical = tuple(canonical)
    samples = _validate_candidate_samples(
        output_root, spec, milestone, checkpoint_sha256, canonical
    )
    paths = candidate_paths(output_root, spec, milestone)
    predictions = validate_prediction_rows(
        read_jsonl(paths.predictions),
        samples,
        canonical,
        evaluator_identity,
        require_complete=False,
    )
    predicted_keys = {row["stable_record_key"] for row in predictions}
    sample_by_key = {row["stable_record_key"]: row for row in samples}
    for record in canonical:
        key = str(record["stable_record_key"])
        if key in predicted_keys:
            continue
        sample = sample_by_key[key]
        prediction = predict(output_root / sample["png_relative_path"])
        predictions.append(
            expected_prediction_record(sample, record, prediction, evaluator_identity)
        )
        atomic_jsonl(paths.predictions, predictions)
    predictions = validate_prediction_rows(
        predictions, samples, canonical, evaluator_identity, require_complete=True
    )
    prediction_by_key = {
        row["stable_record_key"]: row["predictions"] for row in predictions
    }
    id_records = [row for row in canonical if row["split_role"] == "ID_VAL"]
    comp_records = [row for row in canonical if row["split_role"] == "COMP_VAL"]
    id_metrics = attribute_metrics(
        id_records, [prediction_by_key[row["stable_record_key"]] for row in id_records]
    )
    comp_predictions = [
        prediction_by_key[row["stable_record_key"]] for row in comp_records
    ]
    comp_metrics = attribute_metrics(comp_records, comp_predictions)
    family = family_macro_metrics(comp_records, comp_predictions)
    metrics = {
        "schema": "tiny_avatar_diffusion.ca16_r11_candidate_metrics_v1",
        "run_id": spec.run_id,
        "architecture_arm": spec.architecture_arm.value,
        "training_seed": spec.training_seed,
        "milestone": milestone,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_manifest_sha256": sha256_file(paths.sample_manifest),
        "predictions_sha256": sha256_file(paths.predictions),
        "evaluator_checkpoint_sha256": evaluator_identity["checkpoint_sha256"],
        "evaluator_freeze_manifest_sha256": evaluator_identity[
            "freeze_manifest_sha256"
        ],
        "id_count": len(id_records),
        "comp_count": len(comp_records),
        "id_metrics": id_metrics,
        "comp_metrics": comp_metrics,
        "comp_per_family": family["per_family"],
        "comp_family_counts": dict(EXPECTED_SELECTION_FAMILY_COUNTS),
        "comp_family_macro_maa": family["family_macro_maa"],
        "comp_macro_psr": family["macro_psr"],
        "id_maa": id_metrics["maa"],
    }
    if paths.metrics.exists() and read_json(paths.metrics) != metrics:
        raise ValueError(
            "persisted aggregate metrics do not match canonical recomputation"
        )
    if not paths.metrics.exists() and not persist_missing:
        raise ValueError("candidate metrics artifact is required for completion")
    if not paths.metrics.exists():
        atomic_json(paths.metrics, metrics)
    return metrics


def recompute_candidate_metrics(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    evaluator_identity: Mapping[str, str],
    predict: Callable[[Path], Mapping[str, int]],
) -> dict[str, Any]:
    return _recompute_candidate_metrics(
        output_root=output_root,
        spec=spec,
        milestone=milestone,
        checkpoint_sha256=checkpoint_sha256,
        evaluator_identity=evaluator_identity,
        predict=predict,
        canonical=canonical_records(),
    )


def _sample_artifacts_from_persisted(
    output_root: Path,
    spec: CA16RunSpec,
    milestone: int,
    checkpoint_sha256: str,
    evaluator_identity: Mapping[str, str],
    *,
    role: str,
) -> list[SampleArtifact]:
    if role not in ALLOWED_ROLES:
        raise ValueError("artifact conversion rejects forbidden role")
    records = canonical_records()
    samples = _validate_candidate_samples(
        output_root, spec, milestone, checkpoint_sha256, records
    )
    predictions = validate_prediction_rows(
        read_jsonl(candidate_paths(output_root, spec, milestone).predictions),
        samples,
        records,
        evaluator_identity,
        require_complete=True,
    )
    prediction_by_key = {row["stable_record_key"]: row for row in predictions}
    record_by_key = {row["stable_record_key"]: row for row in records}
    artifacts: list[SampleArtifact] = []
    for sample in samples:
        if sample["split_role"] != role:
            continue
        key = sample["stable_record_key"]
        prediction = prediction_by_key[key]["predictions"]
        ground_truth = record_by_key[key]["ground_truth_attributes"]
        artifacts.append(
            SampleArtifact(
                identity=stable_sample_identity(spec, milestone, record_by_key[key]),
                artifact_path=str(output_root / sample["png_relative_path"]),
                ground_truth=dict(ground_truth),
                predictions=dict(prediction),
                correctness={
                    name: int(ground_truth[name]) == int(prediction[name])
                    for name in ATTRIBUTES
                },
            )
        )
    return artifacts


def materialize_real_comp(output_root: Path) -> Path:
    """Create the KID real panel from canonical COMP records, never from sealed data."""
    from tiny_avatar_diffusion.debug_tiny_overfit import preload_images

    directory = output_root / "real_comp"
    manifest_path = output_root / "real_comp_manifest.jsonl"
    records = [row for row in canonical_records() if row["split_role"] == "COMP_VAL"]
    rows = read_jsonl(manifest_path)
    by_key = {row.get("stable_record_key"): row for row in rows}
    if len(by_key) != len(rows) or any(
        key not in {r["stable_record_key"] for r in records} for key in by_key
    ):
        raise ValueError("real COMP provenance is duplicate or noncanonical")
    for index, record in enumerate(records):
        key = record["stable_record_key"]
        relative = Path("real_comp") / _sample_filename(index, key)
        path = output_root / relative
        expected = {
            "schema": "tiny_avatar_diffusion.ca16_r11_real_comp_v1",
            "sample_index": index,
            "stable_record_key": key,
            "image_id": record["image_id"],
            "split_role": "COMP_VAL",
            "png_relative_path": relative.as_posix(),
        }
        if key in by_key:
            row = by_key[key]
            if any(row.get(name) != value for name, value in expected.items()):
                raise ValueError("real COMP provenance mismatch")
            _validate_png(path)
            if row.get("png_sha256") != sha256_file(path):
                raise ValueError("real COMP PNG hash mismatch")
            continue
        if path.exists():
            raise RuntimeError("orphan real COMP PNG exists without provenance")
        pixels = tensor_to_pixels(
            preload_images([{"source_image_path": record["source_image_path"]}])[0]
        )
        rows.append({**expected, "png_sha256": persist_new_png(path, pixels)})
        atomic_jsonl(manifest_path, rows)
    if len(rows) != 237 or len(list(directory.glob("*.png"))) != 237:
        raise ValueError(
            "real COMP KID panel must contain exactly 237 provenanced PNGs"
        )
    return directory


def select_run(
    *,
    output_root: Path,
    spec: CA16RunSpec,
    inventory: Mapping[str, Any],
    evaluator_identity: Mapping[str, str],
    kid_compute: Callable[
        [SelectionCandidate, Sequence[SampleArtifact]], Mapping[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    metrics_by_step: dict[int, dict[str, Any]] = {}
    entries: dict[int, dict[str, Any]] = {}
    candidates: list[SelectionCandidate] = []
    for milestone in MILESTONE_ORDER:
        entry = inventory_entry(inventory, spec, milestone)
        entries[milestone] = entry
        paths = candidate_paths(output_root, spec, milestone)
        if not paths.metrics.is_file():
            raise ValueError(
                "all four adherence metric artifacts are required before ranking"
            )
        metrics = read_json(paths.metrics)
        # Recompute from persisted canonical predictions; the callback cannot be reached
        # because completed candidates must already contain all predictions.
        recomputed = recompute_candidate_metrics(
            output_root=output_root,
            spec=spec,
            milestone=milestone,
            checkpoint_sha256=entry["sha256"],
            evaluator_identity=evaluator_identity,
            predict=lambda _path: (_ for _ in ()).throw(
                RuntimeError("unexpected evaluator reuse gap")
            ),
        )
        if metrics != recomputed:
            raise ValueError("candidate metric artifact is not canonical")
        metrics_by_step[milestone] = metrics
        candidates.append(
            SelectionCandidate(
                run_id=spec.run_id,
                architecture_arm=spec.architecture_arm.value,
                training_seed=spec.training_seed,
                milestone_step=milestone,
                comp_family_macro_maa=float(metrics["comp_family_macro_maa"]),
                comp_macro_psr=float(metrics["comp_macro_psr"]),
                id_maa=float(metrics["id_maa"]),
                checkpoint_path=entry["checkpoint_path"],
                checkpoint_sha256=entry["sha256"],
            )
        )

    kid_results: dict[int, dict[str, Any]] = {}

    def resolve_kid(candidate: SelectionCandidate) -> float:
        artifacts = _sample_artifacts_from_persisted(
            output_root,
            spec,
            candidate.milestone_step,
            entries[candidate.milestone_step]["sha256"],
            evaluator_identity,
            role="COMP_VAL",
        )
        if kid_compute is None:
            result = compute_comp_kid_from_artifacts(
                artifacts,
                materialize_real_comp(output_root),
                spec=spec,
                milestone_step=candidate.milestone_step,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            result = dict(kid_compute(candidate, artifacts))
        if result.get("status") != "COMPUTED" or not isinstance(
            result.get("value"), (int, float)
        ):
            raise ValueError("KID callback must return a frozen COMPUTED result")
        kid_results[candidate.milestone_step] = result
        return float(result["value"])

    ranked = rank_candidates(candidates, kid=resolve_kid)
    rows = []
    for rank, candidate in enumerate(ranked, start=1):
        kid = kid_results.get(candidate.milestone_step)
        rows.append(
            {
                "rank": rank,
                "milestone_step": candidate.milestone_step,
                "checkpoint_sha256": entries[candidate.milestone_step]["sha256"],
                "comp_family_macro_maa": candidate.comp_family_macro_maa,
                "comp_macro_psr": candidate.comp_macro_psr,
                "id_maa": candidate.id_maa,
                "comp_kid": None if kid is None else float(kid["value"]),
                "kid_status": "NOT_REQUIRED" if kid is None else "COMPUTED",
                "selected": rank == 1,
            }
        )
    if sum(row["selected"] for row in rows) != 1:
        raise AssertionError("R11 must select exactly one milestone per run")
    payload = {
        "schema": "tiny_avatar_diffusion.ca16_r11_run_selection_v1",
        "run_id": spec.run_id,
        "architecture_arm": spec.architecture_arm.value,
        "training_seed": spec.training_seed,
        "criteria_order": [
            "comp_family_macro_maa",
            "comp_macro_psr",
            "id_maa",
            "comp_kid",
            "milestone_step",
        ],
        "candidates": rows,
        "selected_milestone": rows[0]["milestone_step"],
    }
    path = output_root / "selections" / f"{spec.run_id}.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("persisted run selection disagrees with canonical ranking")
    if not path.exists():
        atomic_json(path, payload)
    return payload


def update_progress(output_root: Path, **updates: Any) -> dict[str, Any]:
    path = output_root / "progress.json"
    progress = (
        read_json(path)
        if path.exists()
        else {
            "schema": "tiny_avatar_diffusion.ca16_r11_progress_v1",
            "completed_candidates": [],
            "completed_run_selections": [],
        }
    )
    progress.update(updates)
    progress["last_successful_action_utc"] = utc_now()
    atomic_json(path, progress)
    return progress


def validate_preflight(
    *,
    output_root: Path,
    expected_eval_sha: str,
    expected_tag: str | None = None,
    state: GitState | None = None,
    environment: Mapping[str, Any] | None = None,
    root: Path = ROOT,
    protocol_id: str = PROTOCOL_ID,
    sampling_batch_size: int = 1,
) -> dict[str, Any]:
    evaluation = state or git_state(root=root)
    verify_evaluation_git(
        expected_eval_sha, evaluation, expected_tag=expected_tag, root=root
    )
    selection = canonical_records()
    for path, expected in (
        (SELECTION_MANIFEST, SELECTION_MANIFEST_SHA256),
        (ROBUST_MANIFEST, ROBUST_MANIFEST_SHA256),
        (QUALITY_MANIFEST, QUALITY_MANIFEST_SHA256),
        (EVALUATOR_FREEZE, EVALUATOR_FREEZE_SHA256),
        (ROOT / EVALUATOR_CHECKPOINT_RELATIVE, EVALUATOR_CHECKPOINT_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen preflight hash mismatch: {path}")
    inventory = load_checkpoint_inventory(root=root)
    env = dict(environment or runtime_environment())
    existing_manifest = output_root / "run_manifest.json"
    staged_manifest = _operational_temp(existing_manifest)
    manifest_source = (
        existing_manifest if existing_manifest.is_file() else staged_manifest
    )
    timestamp = (
        read_json(manifest_source).get("timestamp_utc")
        if manifest_source.is_file()
        else "PREFLIGHT_TIMESTAMP"
    )
    manifest = build_run_manifest(
        evaluation=evaluation,
        inventory_sha256=sha256_file(INVENTORY_PATH),
        environment=env,
        timestamp=timestamp,
        protocol_id=protocol_id,
        sampling_batch_size=sampling_batch_size,
    )
    output_state = classify_output_root(output_root, manifest)
    if output_state is OutputState.INVALID_OR_UNPROVEN:
        raise RuntimeError("output root is INVALID_OR_UNPROVEN")
    return {
        "status": "PREFLIGHT_PASS",
        "scientific_execution": False,
        "output_state": output_state.value,
        "evaluation_git_sha": evaluation.head,
        "training_git_sha": TRAINING_GIT_SHA,
        "checkpoint_inventory_sha256": sha256_file(INVENTORY_PATH),
        "checkpoints": len(inventory["entries"]),
        "candidates": len(expected_plan()),
        "samples_per_candidate": len(selection),
        "manifest": manifest,
    }


def verify_training_checkpoint_before_load(
    entry: Mapping[str, Any], *, root: Path = ROOT
) -> Path:
    if entry.get("training_git_sha") != TRAINING_GIT_SHA:
        raise ValueError("training provenance is not frozen R10 SHA")
    path = root / str(entry["checkpoint_path"])
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ValueError("training checkpoint SHA256 mismatch before model loading")
    return path


def validate_completion_state(
    *,
    output_root: Path,
    inventory: Mapping[str, Any],
    evaluator_identity: Mapping[str, str],
    evaluation: GitState,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    if evaluator_identity.get("checkpoint_sha256") != EVALUATOR_CHECKPOINT_SHA256:
        raise ValueError("completion evaluator checkpoint identity mismatch")
    if evaluator_identity.get("freeze_manifest_sha256") != EVALUATOR_FREEZE_SHA256:
        raise ValueError("completion evaluator freeze identity mismatch")
    completed_candidates: list[str] = []
    selection_hashes: dict[str, str] = {}
    selected: dict[str, int] = {}
    if verify_checkpoint_files:
        for entry in inventory["entries"]:
            verify_training_checkpoint_before_load(entry)
    for spec in ca16_run_matrix():
        for milestone in MILESTONE_ORDER:
            entry = inventory_entry(inventory, spec, milestone)
            samples = validate_candidate_samples(
                output_root, spec, milestone, entry["sha256"]
            )
            records = canonical_records()
            predictions = validate_prediction_rows(
                read_jsonl(candidate_paths(output_root, spec, milestone).predictions),
                samples,
                records,
                evaluator_identity,
                require_complete=True,
            )
            if len(predictions) != TOTAL_SAMPLES:
                raise ValueError(
                    "completion requires 493 bound predictions per candidate"
                )
            _recompute_candidate_metrics(
                output_root=output_root,
                spec=spec,
                milestone=milestone,
                checkpoint_sha256=entry["sha256"],
                evaluator_identity=evaluator_identity,
                predict=lambda _path: (_ for _ in ()).throw(
                    RuntimeError("prediction gap during completion")
                ),
                canonical=records,
                persist_missing=False,
            )
            completed_candidates.append(f"{spec.run_id}:{milestone}")
        selection_path = output_root / "selections" / f"{spec.run_id}.json"
        selection = read_json(selection_path)
        rows = selection.get("candidates")
        if (
            selection.get("schema") != "tiny_avatar_diffusion.ca16_r11_run_selection_v1"
            or not isinstance(rows, list)
            or len(rows) != 4
            or sum(bool(row.get("selected")) for row in rows) != 1
            or {row.get("milestone_step") for row in rows} != set(MILESTONE_ORDER)
        ):
            raise ValueError("completion requires one canonical selection per run")
        winner = next(row for row in rows if row["selected"])
        if selection.get("selected_milestone") != winner["milestone_step"]:
            raise ValueError("selection winner metadata mismatch")
        selection_hashes[spec.run_id] = sha256_file(selection_path)
        selected[spec.run_id] = int(winner["milestone_step"])
    expected_candidate_count = len(tuple(ca16_run_matrix())) * len(MILESTONE_ORDER)
    expected_selection_count = len(tuple(ca16_run_matrix()))
    if (
        len(completed_candidates) != expected_candidate_count
        or len(selection_hashes) != expected_selection_count
    ):
        raise ValueError("completion requires 24 candidates and six selections")
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    if (
        run_manifest.get("evaluation_git_sha") != evaluation.head
        or not evaluation.tracked_clean
    ):
        raise ValueError("completion evaluation Git identity is no longer valid")
    return {
        "schema": "tiny_avatar_diffusion.ca16_r11_summary_v1",
        "status": "COMPLETE",
        "candidate_count": expected_candidate_count,
        "samples_per_candidate": TOTAL_SAMPLES,
        "selection_count": expected_selection_count,
        "selected_milestones": selected,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "selection_sha256": selection_hashes,
        "checkpoint_inventory_sha256": sha256_file(INVENTORY_PATH),
        "evaluation_git_sha": evaluation.head,
        "training_git_sha": TRAINING_GIT_SHA,
        "evaluator_checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
        "evaluator_freeze_manifest_sha256": EVALUATOR_FREEZE_SHA256,
    }


def write_completion_marker(
    *,
    output_root: Path,
    inventory: Mapping[str, Any],
    evaluator_identity: Mapping[str, str],
    evaluation: GitState,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    if (output_root / "R11_COMPLETE.json").exists():
        raise FileExistsError("R11 completion marker already exists")
    summary = validate_completion_state(
        output_root=output_root,
        inventory=inventory,
        evaluator_identity=evaluator_identity,
        evaluation=evaluation,
        verify_checkpoint_files=verify_checkpoint_files,
    )
    summary_path = output_root / "r11_summary.json"
    if summary_path.exists() and read_json(summary_path) != summary:
        raise ValueError("existing R11 summary is invalid and will not be repaired")
    if not summary_path.exists():
        atomic_json(summary_path, summary)
    marker = {
        "schema": "tiny_avatar_diffusion.ca16_r11_complete_v1",
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "run_manifest_sha256": summary["run_manifest_sha256"],
        "r11_summary_sha256": sha256_file(summary_path),
        "selection_sha256": summary["selection_sha256"],
        "checkpoint_inventory_sha256": summary["checkpoint_inventory_sha256"],
    }
    atomic_json(output_root / "R11_COMPLETE.json", marker)
    return marker


def validate_existing_completion(
    *,
    output_root: Path,
    inventory: Mapping[str, Any],
    evaluator_identity: Mapping[str, str],
    evaluation: GitState,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    marker_path = output_root / "R11_COMPLETE.json"
    summary_path = output_root / "r11_summary.json"
    marker = read_json(marker_path)
    expected_summary = validate_completion_state(
        output_root=output_root,
        inventory=inventory,
        evaluator_identity=evaluator_identity,
        evaluation=evaluation,
        verify_checkpoint_files=verify_checkpoint_files,
    )
    if read_json(summary_path) != expected_summary:
        raise ValueError("completed R11 summary no longer validates")
    expected_marker_fields = {
        "schema": "tiny_avatar_diffusion.ca16_r11_complete_v1",
        "status": "COMPLETE",
        "run_manifest_sha256": expected_summary["run_manifest_sha256"],
        "r11_summary_sha256": sha256_file(summary_path),
        "selection_sha256": expected_summary["selection_sha256"],
        "checkpoint_inventory_sha256": expected_summary["checkpoint_inventory_sha256"],
    }
    if not isinstance(marker.get("completed_utc"), str) or any(
        marker.get(key) != value for key, value in expected_marker_fields.items()
    ):
        raise ValueError("R11 completion marker is invalid")
    return marker


def execute_r11(
    *,
    output_root: Path,
    expected_eval_sha: str,
    expected_tag: str | None = None,
    device: torch.device | None = None,
    protocol_id: str = PROTOCOL_ID,
    sampling_batch_size: int = 1,
) -> dict[str, Any]:
    """Execute R11.  The CLI is the only intended production caller."""
    preflight = validate_preflight(
        output_root=output_root,
        expected_eval_sha=expected_eval_sha,
        expected_tag=expected_tag,
        protocol_id=protocol_id,
        sampling_batch_size=sampling_batch_size,
    )
    evaluation = git_state()
    inventory = load_checkpoint_inventory()
    environment = runtime_environment()
    manifest = dict(preflight["manifest"])
    if preflight["output_state"] == OutputState.NEW.value:
        manifest = build_run_manifest(
            evaluation=evaluation,
            inventory_sha256=sha256_file(INVENTORY_PATH),
            environment=environment,
            protocol_id=protocol_id,
            sampling_batch_size=sampling_batch_size,
        )
    target_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if target_device.type != "cuda":
        raise RuntimeError("scientific R11 execution requires CUDA")

    with exclusive_run_lock(output_root):
        initialize_or_resume(output_root, manifest)
        logger = configure_logger(output_root)
        logger.info("R11 process start pid=%s output=%s", os.getpid(), output_root)
        if preflight["output_state"] == OutputState.RESUMABLE_VALID.value:
            logger.info(
                "validated resume event: existing artifacts will be reused only after hash checks"
            )
        try:
            logger.info("frozen evaluator load start")
            evaluator, evaluator_identity = load_frozen_evaluator(device=target_device)
            logger.info("frozen evaluator load complete")
            if (output_root / "R11_COMPLETE.json").exists():
                marker = validate_existing_completion(
                    output_root=output_root,
                    inventory=inventory,
                    evaluator_identity=evaluator_identity,
                    evaluation=evaluation,
                )
                logger.info(
                    "validated existing R11 completion marker; no scientific work performed"
                )
                return marker

            token_source_index = load_tokenized_source_index()

            def predict(path: Path) -> Mapping[str, int]:
                return predict_artifact_attributes(
                    evaluator, [str(path)], device=target_device
                )[0]

            completed_candidates: list[str] = []
            completed_selections: list[str] = []
            for spec in ca16_run_matrix():
                for milestone in MILESTONE_ORDER:
                    entry = inventory_entry(inventory, spec, milestone)
                    checkpoint_path = verify_training_checkpoint_before_load(entry)
                    logger.info(
                        "candidate start run=%s milestone=%s", spec.run_id, milestone
                    )
                    update_progress(
                        output_root,
                        current_run=spec.run_id,
                        current_milestone=milestone,
                        current_phase="GENERATION",
                        completed_candidates=completed_candidates,
                        completed_run_selections=completed_selections,
                    )
                    text_encoder, denoiser, checkpoint = load_ema_checkpoint(
                        checkpoint_path,
                        spec,
                        milestone,
                        device=target_device,
                    )
                    identity = checkpoint.get("experiment", {})
                    if identity.get("git_commit") != TRAINING_GIT_SHA:
                        raise ValueError(
                            "checkpoint training Git SHA is not authorized frozen R10 SHA"
                        )
                    schedule = LinearDDPMSchedule(num_timesteps=1_000).to(target_device)

                    def generate(
                        record: Mapping[str, Any],
                        schedule: LinearDDPMSchedule = schedule,
                        text_encoder: Any = text_encoder,
                        denoiser: Any = denoiser,
                    ) -> torch.Tensor:
                        hydrated = hydrate_selection_record(record, token_source_index)
                        return sample_ema_record(
                            schedule,
                            text_encoder,
                            denoiser,
                            hydrated,
                            device=target_device,
                        )

                    def generate_batch(
                        records: Sequence[Mapping[str, Any]],
                        schedule: LinearDDPMSchedule = schedule,
                        text_encoder: Any = text_encoder,
                        denoiser: Any = denoiser,
                    ) -> Sequence[torch.Tensor]:
                        hydrated = [
                            hydrate_selection_record(record, token_source_index)
                            for record in records
                        ]
                        return list(
                            sample_ema_records_batch(
                                schedule,
                                text_encoder,
                                denoiser,
                                hydrated,
                                device=target_device,
                            )
                        )

                    def record_generation_progress(
                        count: int, s: CA16RunSpec = spec, m: int = milestone
                    ) -> None:
                        update_progress(
                            output_root,
                            current_run=s.run_id,
                            current_milestone=m,
                            current_phase="GENERATION",
                            completed_sample_count=count,
                            completed_candidates=completed_candidates,
                            completed_run_selections=completed_selections,
                        )
                        if count == 1 or count % 25 == 0 or count == TOTAL_SAMPLES:
                            logger.info(
                                "generation progress run=%s milestone=%s samples=%s/%s",
                                s.run_id,
                                m,
                                count,
                                TOTAL_SAMPLES,
                            )

                    if protocol_id == BATCHED_PROTOCOL_ID:
                        resume_candidate_generation_batched(
                            output_root=output_root,
                            spec=spec,
                            milestone=milestone,
                            checkpoint_sha256=entry["sha256"],
                            generate_batch=generate_batch,
                            sampling_batch_size=sampling_batch_size,
                            on_progress=record_generation_progress,
                        )
                    else:
                        resume_candidate_generation(
                            output_root=output_root,
                            spec=spec,
                            milestone=milestone,
                            checkpoint_sha256=entry["sha256"],
                            generate=generate,
                            on_progress=record_generation_progress,
                        )
                    update_progress(
                        output_root,
                        current_run=spec.run_id,
                        current_milestone=milestone,
                        current_phase="EVALUATION",
                        completed_sample_count=TOTAL_SAMPLES,
                        completed_candidates=completed_candidates,
                        completed_run_selections=completed_selections,
                    )
                    logger.info(
                        "evaluator start run=%s milestone=%s", spec.run_id, milestone
                    )
                    recompute_candidate_metrics(
                        output_root=output_root,
                        spec=spec,
                        milestone=milestone,
                        checkpoint_sha256=entry["sha256"],
                        evaluator_identity=evaluator_identity,
                        predict=predict,
                    )
                    logger.info(
                        "evaluator complete run=%s milestone=%s", spec.run_id, milestone
                    )
                    candidate_key = f"{spec.run_id}:{milestone}"
                    if candidate_key not in completed_candidates:
                        completed_candidates.append(candidate_key)
                    logger.info(
                        "candidate complete run=%s milestone=%s", spec.run_id, milestone
                    )
                update_progress(
                    output_root,
                    current_run=spec.run_id,
                    current_milestone=None,
                    current_phase="KID_OR_RANKING",
                    completed_candidates=completed_candidates,
                    completed_run_selections=completed_selections,
                )
                logger.info("ranking start run=%s", spec.run_id)
                select_run(
                    output_root=output_root,
                    spec=spec,
                    inventory=inventory,
                    evaluator_identity=evaluator_identity,
                )
                if spec.run_id not in completed_selections:
                    completed_selections.append(spec.run_id)
                logger.info("ranking complete run=%s", spec.run_id)
            marker = write_completion_marker(
                output_root=output_root,
                inventory=inventory,
                evaluator_identity=evaluator_identity,
                evaluation=evaluation,
            )
            update_progress(
                output_root,
                current_run=None,
                current_milestone=None,
                current_phase="COMPLETE",
                completed_sample_count=24 * TOTAL_SAMPLES,
                completed_candidates=completed_candidates,
                completed_run_selections=completed_selections,
            )
            logger.info("R11 clean successful completion")
            return marker
        except Exception as error:
            logger.error("R11 failed: %s\n%s", error, traceback.format_exc())
            try:
                update_progress(
                    output_root, current_phase="FAILED", last_error=repr(error)
                )
            except (OSError, TypeError, ValueError):
                logger.error(
                    "failed to update progress after exception\n%s",
                    traceback.format_exc(),
                )
            raise
