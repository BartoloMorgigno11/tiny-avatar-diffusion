"""D25 DR05 generative-quality pipeline, with a non-sealed ID_VAL dry-run.

The only executable mode used by this task is ``dry-run``.  The final plan is
encoded here so that a later explicitly authorised sealed evaluation uses the
same allocation, RNG, serialization, manifest, and Clean-FID contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import tarfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy
import torch
from PIL import Image

try:  # Supports both ``python scripts/...`` and package imports in tests.
    from scripts.dr05_materialize_canonical_checkpoints import (
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
except ModuleNotFoundError:
    from dr05_materialize_canonical_checkpoints import (  # type: ignore[no-redef]
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer
from tiny_avatar_diffusion.dr05_final_executor import ExecutionIdentity, FinalQualityExecutor, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
ID_VAL = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_val.jsonl"
ID_TEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_test.jsonl"
FINAL_OOD = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "final_ood.jsonl"
VOCAB = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "vocab.json"
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
BINDINGS = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_execution_bindings.json"
CONDITIONAL = ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt"
UNCONDITIONAL = ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt"
EVALUATOR = ROOT / "artifacts" / "checkpoints" / "dr04_evaluator_epoch41.pt"
RUNTIME_BENCHMARK = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_runtime_benchmark.json"
OUTPUT = ROOT / "results" / "runs" / "dr05_quality_pipeline_dryrun_v1"
MAIN_PLAN = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_main_quality_generation_plan.jsonl"
MAIN_PLAN_META = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_main_quality_generation_plan_meta.json"
FINAL_OUTPUT = ROOT / "results" / "runs" / "dr05_final_quality_v1"

PROTOCOL_SHA256 = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
BINDINGS_SHA256 = "603a9121f980b769621b628a0cf393dda17dd10a5e6aeca6e6e94bcc3e671ebf"
CONDITIONAL_SHA256 = "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"
UNCONDITIONAL_SHA256 = "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e"
EVALUATOR_SHA256 = "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"
RUNTIME_BENCHMARK_SHA256 = "5b59c7c5662f1b93170175fe0bc945ee15a5fd68317fb36f2af21d6578120e5a"
ID_TEST_METADATA_SHA256 = "6b23a54ef90d74a0b6bdff0ac8054d4e8c4ec92ee6e6311f07316715e268adde"
FINAL_OOD_METADATA_SHA256 = "8414c64ac13781c148fd38154bb04b684b841088f2e4243b7e03afb524bf070d"
CONDITIONAL_PARAMETERS = 4_723_907
UNCONDITIONAL_PARAMETERS = 4_595_779

FINAL_ALLOCATIONS = {
    5_000: {"id": 2_884, "ood": 2_116},
    10_000: {"id": 5_767, "ood": 4_233},
    25_000: {"id": 14_418, "ood": 10_582},
    50_000: {"id": 28_837, "ood": 21_163},
}
FINAL_NAMESPACE = "quality_generation"
FINAL_SEED_BASE = 502_026
DRY_RUN_NAMESPACE = "quality_pipeline_dryrun"
DRY_RUN_SEED_BASE = 702_026
BATCH_SIZE = 16
DRY_RUN_COUNT = 32
KID_SUBSET_SEED = 2026
CLEAN_FID_VERSION = "0.1.35"
MAIN_ID_COUNT = 5_767
MAIN_OOD_COUNT = 4_233
FINAL_LAYOUT = {
    "root": "results/runs/dr05_final_quality_v1",
    "directories": [
        "conditional/id", "conditional/ood", "unconditional/id", "unconditional/ood",
        "real/id", "real/ood", "manifests", "metrics",
    ],
}
GENERATED_MANIFEST_FIELDS = {
    "sample_id", "model", "checkpoint_path", "checkpoint_sha256", "evaluation_role", "split",
    "reference_sample_id", "prompt", "generation_namespace", "batch_index", "batch_seed",
    "sample_index_in_batch", "image_shape", "image_mode", "dtype", "png_relative_path",
    "png_sha256", "pixel_sha256", "status", "conditioning_consumed",
}
REAL_MANIFEST_FIELDS = {
    "reference_sample_id", "source_split", "source_image_reference", "canonical_shape", "mode",
    "dtype", "canonical_png_relative_path", "png_sha256", "pixel_sha256", "status",
}
RECOVERY_TASK = "DR05-I20"
RETAINED_I19_MANIFEST_SHA256 = {
    "real": "28131e82b82180ae674be39bdb7d7de615e4b4edd6b817c776a2341653a285cc",
    "conditional": "f5a42c77fb3400d2888e64375338396a63154702977d070aa39b3fbcbdf06d3a",
    "unconditional": "9ee0a9cd22b6a13e712d5247402d0f8e2b1c1e0104f2b56c00935c653159beed",
}

MAIN_PLAN_FIELDS = {
    "planned_sample_index", "evaluation_role", "split", "reference_sample_id", "prompt",
    "reference_occurrence_index", "reference_repeat_count", "batch_index", "batch_seed",
    "sample_index_in_batch", "generation_namespace",
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_final_metadata(path: Path, *, expected_sha256: str, expected_count: int, split: str) -> list[dict[str, Any]]:
    """Read sealed metadata only; source-image paths are deliberately never opened here."""
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"unexpected frozen {split} metadata SHA256")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != expected_count:
        raise ValueError(f"unexpected {split} metadata record count")
    if any(record.get("split_role") != split for record in records):
        raise ValueError(f"unexpected split role in {split} metadata")
    identifiers = [record.get("image_id") for record in records]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{split} metadata must have unique stable image_id values")
    if any(not isinstance(record.get("caption"), str) for record in records):
        raise ValueError(f"{split} metadata must contain canonical captions")
    return sorted(records, key=lambda record: str(record["image_id"]))


def _expand_final_split(records: list[dict[str, Any]], *, split: str, target: int, start_index: int) -> list[dict[str, Any]]:
    cycles, remainder = divmod(target, len(records))
    expanded = records * cycles + records[:remainder]
    occurrences: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for offset, record in enumerate(expanded):
        reference_id = str(record["image_id"])
        sample_index = start_index + offset
        repeat_count = cycles + int(records.index(record) < remainder)
        rows.append({
            "planned_sample_index": sample_index, "evaluation_role": "main_quality", "split": split,
            "reference_sample_id": reference_id, "prompt": record["caption"],
            "reference_occurrence_index": occurrences[reference_id], "reference_repeat_count": repeat_count,
            "batch_index": sample_index // BATCH_SIZE, "batch_seed": FINAL_SEED_BASE + sample_index // BATCH_SIZE,
            "sample_index_in_batch": sample_index % BATCH_SIZE, "generation_namespace": FINAL_NAMESPACE,
        })
        occurrences[reference_id] += 1
    return rows


def build_main_quality_generation_plan() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Build the D30 plan solely from frozen JSONL metadata, without image I/O or RNG."""
    id_records = _load_final_metadata(ID_TEST, expected_sha256=ID_TEST_METADATA_SHA256, expected_count=466, split="ID_TEST")
    ood_records = _load_final_metadata(FINAL_OOD, expected_sha256=FINAL_OOD_METADATA_SHA256, expected_count=342, split="FINAL_OOD")
    rows = _expand_final_split(id_records, split="ID", target=MAIN_ID_COUNT, start_index=0)
    rows.extend(_expand_final_split(ood_records, split="OOD", target=MAIN_OOD_COUNT, start_index=MAIN_ID_COUNT))
    audit_main_quality_generation_plan(rows, {"ID": id_records, "OOD": ood_records})
    return rows, {"ID": id_records, "OOD": ood_records}


def audit_main_quality_generation_plan(rows: list[dict[str, Any]], sources: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Validate D30 structure from parsed plan rows; this function never rebuilds a plan."""
    if len(rows) != MAIN_ID_COUNT + MAIN_OOD_COUNT or any(set(row) != MAIN_PLAN_FIELDS for row in rows):
        raise ValueError("final plan schema or length mismatch")
    if [row["planned_sample_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("final plan indices are not contiguous zero-based values")
    if any(row["evaluation_role"] != "main_quality" or row["generation_namespace"] != FINAL_NAMESPACE for row in rows):
        raise ValueError("final plan role or namespace mismatch")
    if [row["split"] for row in rows[:MAIN_ID_COUNT]] != ["ID"] * MAIN_ID_COUNT or [row["split"] for row in rows[MAIN_ID_COUNT:]] != ["OOD"] * MAIN_OOD_COUNT:
        raise ValueError("final plan must contain all ID rows before OOD rows")
    for row in rows:
        index = row["planned_sample_index"]
        if row["batch_index"] != index // BATCH_SIZE or row["sample_index_in_batch"] != index % BATCH_SIZE or row["batch_seed"] != FINAL_SEED_BASE + index // BATCH_SIZE:
            raise ValueError("final plan batch indexing or seed mismatch")
    if rows[MAIN_ID_COUNT]["batch_index"] != 360 or rows[MAIN_ID_COUNT]["sample_index_in_batch"] != 7 or rows[MAIN_ID_COUNT]["batch_seed"] != 502386:
        raise ValueError("frozen ID/OOD boundary mismatch")
    if len({row["batch_index"] for row in rows}) != 625 or rows[0]["batch_seed"] != 502026 or rows[-1]["batch_seed"] != 502650:
        raise ValueError("final plan batch schedule mismatch")
    result: dict[str, Any] = {}
    for split, target, remainder in (("ID", MAIN_ID_COUNT, 175), ("OOD", MAIN_OOD_COUNT, 129)):
        source = sources[split]
        split_rows = [row for row in rows if row["split"] == split]
        source_ids = [str(record["image_id"]) for record in source]
        source_prompts = {str(record["image_id"]): record["caption"] for record in source}
        ids = [str(row["reference_sample_id"]) for row in split_rows]
        multiplicities = Counter(ids)
        if set(ids) != set(source_ids) or len(multiplicities) != len(source_ids):
            raise ValueError(f"{split} final plan coverage mismatch")
        if any(count not in {12, 13} for count in multiplicities.values()) or sum(count == 13 for count in multiplicities.values()) != remainder or sum(count == 12 for count in multiplicities.values()) != len(source_ids) - remainder:
            raise ValueError(f"{split} final plan multiplicity mismatch")
        expected = source * 12 + source[:remainder]
        if ids != [str(record["image_id"]) for record in expected]:
            raise ValueError(f"{split} final plan cycle ordering mismatch")
        if any(row["prompt"] != source_prompts[str(row["reference_sample_id"])] for row in split_rows):
            raise ValueError(f"{split} final plan rewrites canonical prompt")
        expected_occurrences: Counter[str] = Counter()
        for row in split_rows:
            reference_id = str(row["reference_sample_id"])
            if row["reference_occurrence_index"] != expected_occurrences[reference_id] or row["reference_repeat_count"] != multiplicities[reference_id]:
                raise ValueError(f"{split} final plan occurrence metadata mismatch")
            expected_occurrences[reference_id] += 1
        result[split] = {"records": len(source_ids), "rows": len(split_rows), "remainder": remainder, "multiplicity_13": sum(count == 13 for count in multiplicities.values()), "multiplicity_12": sum(count == 12 for count in multiplicities.values())}
    return result


def materialize_main_quality_generation_plan() -> dict[str, Any]:
    """Deliberate I22 materialization action. Future execution only loads this artifact."""
    if MAIN_PLAN.exists() or MAIN_PLAN_META.exists():
        raise FileExistsError("refusing to overwrite an existing frozen final generation plan")
    rows, sources = build_main_quality_generation_plan()
    MAIN_PLAN.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    plan_sha256 = sha256_file(MAIN_PLAN)
    audit = audit_main_quality_generation_plan(_read_jsonl(MAIN_PLAN), sources)
    metadata = {
        "schema": "dr05_main_quality_generation_plan", "version": 1, "scientific_role": "shared_main_quality_generation_plan",
        "D30_FROZEN": True,
        "source_metadata": {
            "ID": {"path": _relative(ID_TEST), "sha256": ID_TEST_METADATA_SHA256, "record_count": 466},
            "OOD": {"path": _relative(FINAL_OOD), "sha256": FINAL_OOD_METADATA_SHA256, "record_count": 342},
        },
        "allocation": {"ID": MAIN_ID_COUNT, "OOD": MAIN_OOD_COUNT, "total": 10_000},
        "expansion": {"ID": {"complete_cycles": 12, "remainder": 175}, "OOD": {"complete_cycles": 12, "remainder": 129}},
        "ordering": {"record_order": "lexicographic_image_id", "global_order": "all_ID_then_all_OOD"},
        "generation": {"namespace": FINAL_NAMESPACE, "seed_base": FINAL_SEED_BASE, "batch_size": BATCH_SIZE, "batch_count": 625, "first_batch_seed": 502026, "last_batch_seed": 502650},
        "boundary": {"first_ood_planned_sample_index": 5767, "boundary_batch_index": 360, "boundary_sample_index_in_batch": 7, "boundary_batch_seed": 502386},
        "generation_plan_path": _relative(MAIN_PLAN), "generation_plan_sha256": plan_sha256, "audit": audit,
    }
    MAIN_PLAN_META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_frozen_main_quality_generation_plan()
    return metadata


def verify_frozen_main_quality_generation_plan() -> dict[str, Any]:
    """Identity gate for later final execution. It validates and never regenerates a plan."""
    if not MAIN_PLAN.is_file() or not MAIN_PLAN_META.is_file():
        raise FileNotFoundError("frozen final generation plan and metadata are required")
    metadata = json.loads(MAIN_PLAN_META.read_text(encoding="utf-8"))
    if metadata.get("D30_FROZEN") is not True or metadata.get("generation_plan_path") != _relative(MAIN_PLAN) or metadata.get("generation_plan_sha256") != sha256_file(MAIN_PLAN):
        raise ValueError("frozen final generation plan identity mismatch")
    expected_sources = {"ID": (ID_TEST, ID_TEST_METADATA_SHA256, 466, "ID_TEST"), "OOD": (FINAL_OOD, FINAL_OOD_METADATA_SHA256, 342, "FINAL_OOD")}
    sources = {name: _load_final_metadata(path, expected_sha256=sha, expected_count=count, split=role) for name, (path, sha, count, role) in expected_sources.items()}
    expected_source_metadata = {
        "ID": {"path": _relative(ID_TEST), "sha256": ID_TEST_METADATA_SHA256, "record_count": 466},
        "OOD": {"path": _relative(FINAL_OOD), "sha256": FINAL_OOD_METADATA_SHA256, "record_count": 342},
    }
    expected_generation = {"namespace": FINAL_NAMESPACE, "seed_base": FINAL_SEED_BASE, "batch_size": BATCH_SIZE, "batch_count": 625, "first_batch_seed": 502026, "last_batch_seed": 502650}
    if metadata.get("source_metadata") != expected_source_metadata or metadata.get("allocation") != {"ID": MAIN_ID_COUNT, "OOD": MAIN_OOD_COUNT, "total": 10_000} or metadata.get("generation") != expected_generation:
        raise ValueError("frozen final generation plan source, allocation, or seed policy mismatch")
    audit_main_quality_generation_plan(_read_jsonl(MAIN_PLAN), sources)
    return metadata


def final_status_only(root: Path = FINAL_OUTPUT) -> dict[str, Any]:
    """Read the lightweight final-run state without loading models or source data."""
    state = root / "phase_status" / "execution_state.json"
    progress = root / "phase_status" / "progress.json"
    return {"state": json.loads(state.read_text(encoding="utf-8")) if state.is_file() else None,
            "progress": json.loads(progress.read_text(encoding="utf-8")) if progress.is_file() else None}


def verify_final_metric_environment() -> dict[str, str]:
    environment = verify_recovery_environment()
    checked = subprocess.run([sys.executable, "-m", "pip", "check"], check=False, capture_output=True, text=True)
    if checked.returncode != 0:
        raise RuntimeError(f"pip check failed before final metrics: {checked.stdout}{checked.stderr}")
    square_root, _ = scipy.linalg.sqrtm(np.eye(2), disp=False)
    if not np.allclose(square_root, np.eye(2)):
        raise RuntimeError("SciPy sqrtm compatibility gate failed")
    return environment


def _final_reference_records() -> dict[str, list[dict[str, Any]]]:
    """Map canonical plan reference IDs to metadata; no image archive is opened."""
    id_records = _load_final_metadata(ID_TEST, expected_sha256=ID_TEST_METADATA_SHA256, expected_count=466, split="ID_TEST")
    ood_records = _load_final_metadata(FINAL_OOD, expected_sha256=FINAL_OOD_METADATA_SHA256, expected_count=342, split="FINAL_OOD")
    return {"ID": [{**record, "reference_sample_id": record["image_id"], "source_image_reference": record["source_image_path"]} for record in id_records],
            "OOD": [{**record, "reference_sample_id": record["image_id"], "source_image_reference": record["source_image_path"]} for record in ood_records]}


def build_final_executor(*, output: Path = FINAL_OUTPUT) -> FinalQualityExecutor:
    """Construct, but do not start, the D32 executor using only frozen inputs."""
    verify_frozen_main_quality_generation_plan()
    plan = load_jsonl(MAIN_PLAN)
    metadata = json.loads(MAIN_PLAN_META.read_text(encoding="utf-8"))
    references = _final_reference_records()
    records_by_id = {record["image_id"]: record for records in references.values() for record in records}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    schedule = LinearDDPMSchedule().to(device).eval()
    conditional_artifact = torch.load(CONDITIONAL, map_location="cpu", weights_only=False)
    text, conditional = _conditional_models_from_artifact(conditional_artifact)
    text, conditional = text.to(device).eval(), conditional.to(device).eval()
    unconditional_artifact = torch.load(UNCONDITIONAL, map_location="cpu", weights_only=False)
    unconditional = _unconditional_model_from_artifact(unconditional_artifact).to(device).eval()
    if sum(parameter.numel() for parameter in text.parameters()) + sum(parameter.numel() for parameter in conditional.parameters()) != CONDITIONAL_PARAMETERS or sum(parameter.numel() for parameter in unconditional.parameters()) != UNCONDITIONAL_PARAMETERS:
        raise ValueError("canonical final model parameter count mismatch")

    def generate(model: str, rows: list[dict[str, Any]], seed: int) -> list[np.ndarray]:
        if model == "conditional":
            record_rows = [records_by_id[str(row["reference_sample_id"])] for row in rows]
            input_ids, masks = _tokenize_records(record_rows)
            samples = conditional_sample(schedule, text, conditional, input_ids.to(device), masks.to(device), generator=torch.Generator(device=device).manual_seed(seed), clip_denoised=True)
        elif model == "unconditional":
            samples = _sample_unconditional(schedule, unconditional, batch_size=len(rows), seed=seed, device=device)
        else:
            raise ValueError("unknown final model")
        return [generated_to_rgb_uint8(sample) for sample in samples]

    def metrics(model: str, split: str, real: Path, generated: Path) -> dict[str, float]:
        return compute_clean_fid_metrics(real, generated, device=device)["results"]

    return FinalQualityExecutor(root=output, plan=plan,
        identity=ExecutionIdentity(_relative(MAIN_PLAN), sha256_file(MAIN_PLAN), _relative(MAIN_PLAN_META), sha256_file(MAIN_PLAN_META)),
        references=references, reference_loader=_decode_real_pixels, generator=generate, metric=metrics,
        checkpoint_sha={"conditional": CONDITIONAL_SHA256, "unconditional": UNCONDITIONAL_SHA256},
        checkpoint_path={"conditional": _relative(CONDITIONAL), "unconditional": _relative(UNCONDITIONAL)})


def execute_final_quality(*, output: Path = FINAL_OUTPUT) -> dict[str, Any]:
    """D32 phase driver. This is reached only by explicit confirmed final CLI use."""
    verify_final_metric_environment()
    executor = build_final_executor(output=output)
    executor.canonicalize_references()
    executor.generate_model("conditional")
    executor.generate_model("unconditional")
    executor.audit()
    verify_final_metric_environment()
    executor.metrics()
    return executor.summary({"protocol_sha256": PROTOCOL_SHA256, "bindings_sha256": BINDINGS_SHA256,
        "generation_plan": {"path": _relative(MAIN_PLAN), "sha256": sha256_file(MAIN_PLAN), "metadata_path": _relative(MAIN_PLAN_META), "metadata_sha256": sha256_file(MAIN_PLAN_META)},
        "environment": _environment_metadata(), "non_authorized_evaluations": {"evaluator_runs": 0, "lpips_runs": 0, "minimal_edit_runs": 0, "ext_generated_images": 0}})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(pixels: np.ndarray) -> str:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.shape != (32, 32, 3):
        raise ValueError("canonical pixels must have shape [32, 32, 3]")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def final_generation_plan(count: int) -> list[dict[str, int]]:
    if count not in FINAL_ALLOCATIONS:
        raise ValueError(f"unsupported frozen final count: {count}")
    if count % BATCH_SIZE:
        raise ValueError("final count must be divisible by the physical batch size")
    return [
        {"batch_index": index, "batch_seed": FINAL_SEED_BASE + index}
        for index in range(count // BATCH_SIZE)
    ]


def dry_run_generation_plan() -> list[dict[str, int]]:
    return [
        {"batch_index": index, "batch_seed": DRY_RUN_SEED_BASE + index}
        for index in range(DRY_RUN_COUNT // BATCH_SIZE)
    ]


def metric_config(n_real: int, n_generated: int) -> dict[str, Any]:
    if n_real <= 0 or n_generated <= 0:
        raise ValueError("KID/FID pools must be non-empty")
    return {
        "library": "clean-fid",
        "library_version": CLEAN_FID_VERSION,
        "mode": "clean",
        "feature_network": "standard Clean-FID Inception-v3 evaluation network",
        "kid": {
            "num_subsets": 100,
            "subset_size": min(1000, n_real, n_generated),
            "sampling": "without_replacement_within_subset",
            "subset_seed": KID_SUBSET_SEED,
        },
        "fid": {"role": "secondary"},
        "n_real": n_real,
        "n_generated": n_generated,
    }


def validate_mode(mode: str | None, *, confirm_sealed_final_eval: bool = False) -> str:
    if mode is None:
        raise ValueError("--mode is required; FINAL is never the default")
    if mode not in {"dry-run", "final"}:
        raise ValueError("mode must be dry-run or final")
    if mode == "final" and not confirm_sealed_final_eval:
        raise PermissionError("final mode requires --confirm-sealed-final-eval")
    return mode


def assert_dry_run_records(records: list[dict[str, Any]]) -> None:
    if len(records) != DRY_RUN_COUNT:
        raise ValueError(f"dry-run requires exactly {DRY_RUN_COUNT} records")
    if any(record.get("split_role") != "ID_VAL" for record in records):
        raise ValueError("dry-run rejects non-ID_VAL records and sealed content")
    if any("::" not in str(record.get("source_image_path", "")) for record in records):
        raise ValueError("dry-run records require DR01 archive image references")


def select_dry_run_records(path: Path = ID_VAL) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = sorted(records, key=lambda item: str(item["image_id"]))[:DRY_RUN_COUNT]
    assert_dry_run_records(selected)
    if len({item["image_id"] for item in selected}) != len(selected):
        raise ValueError("stable sample identifiers must be unique")
    return selected


def generated_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.shape != (3, 32, 32):
        raise ValueError("generated tensor must have shape [3, 32, 32]")
    pixels = ((tensor.detach().cpu().to(torch.float32).clamp(-1, 1) + 1.0) / 2.0 * 255.0)
    return pixels.round().to(torch.uint8).permute(1, 2, 0).numpy()


def canonicalize_real_image(image: Image.Image) -> np.ndarray:
    # Pillow's non-nearest resize filters apply antialiasing when downsampling.
    resized = image.convert("RGB").resize((32, 32), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.uint8)
    if pixels.shape != (32, 32, 3):
        raise ValueError("real image did not canonicalize to RGB 32x32")
    return np.ascontiguousarray(pixels)


def save_canonical_png(pixels: np.ndarray, path: Path) -> dict[str, str]:
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.shape != (32, 32, 3):
        raise ValueError("canonical PNG must be RGB 32x32 uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    with Image.open(path) as reloaded:
        round_trip = np.asarray(reloaded.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(pixels, round_trip):
        raise RuntimeError("lossless PNG round-trip pixel mismatch")
    return {"png_sha256": sha256_file(path), "pixel_sha256": pixel_sha256(round_trip)}


def _decode_real_pixels(record: dict[str, Any]) -> np.ndarray:
    reference, member = str(record["source_image_path"]).split("::", maxsplit=1)
    archive_path = ROOT / reference
    with tarfile.open(archive_path, "r:gz") as archive:
        stream = archive.extractfile(member)
        if stream is None:
            raise FileNotFoundError(f"archive member not found: {member}")
        payload = stream.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return canonicalize_real_image(image)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda item: str(item.get("sample_id", item.get("reference_sample_id"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    return sha256_file(path)


def canonicalize_real_pool(records: list[dict[str, Any]], root: Path) -> tuple[Path, str]:
    assert_dry_run_records(records)
    rows: list[dict[str, Any]] = []
    for record in records:
        pixels = _decode_real_pixels(record)
        relative = Path("real_id_val") / f"{record['image_id']}.png"
        hashes = save_canonical_png(pixels, root / relative)
        rows.append({
            "reference_sample_id": record["image_id"], "source_split": "ID_VAL",
            "source_image_reference": record["source_image_path"], "canonical_shape": [32, 32, 3],
            "mode": "RGB", "dtype": "uint8", "canonical_png_relative_path": relative.as_posix(),
            **hashes, "status": "complete",
        })
    if any(set(row) != REAL_MANIFEST_FIELDS for row in rows):
        raise RuntimeError("real-reference manifest schema mismatch")
    manifest = root / "manifests" / "real_id_val_manifest.jsonl"
    return manifest, _write_jsonl(manifest, rows)


def _tokenize_records(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    vocabulary = json.loads(VOCAB.read_text(encoding="utf-8"))["token_to_id"]
    tokenizer = WhitespaceTokenizer(vocabulary)
    encoded = [tokenizer.encode(str(record["caption"])) for record in records]
    for record, item in zip(records, encoded):
        if item.token_ids != record["token_ids"] or item.attention_mask != record["attention_mask"]:
            raise ValueError("caption tokenization differs from frozen DR01 metadata")
    return torch.tensor([item.token_ids for item in encoded], dtype=torch.long), torch.tensor([item.attention_mask for item in encoded], dtype=torch.long)


def _sample_unconditional(schedule: LinearDDPMSchedule, model: torch.nn.Module, *, batch_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn((batch_size, 3, 32, 32), device=device, dtype=torch.float32, generator=generator)
    with torch.inference_mode():
        for value in range(schedule.betas.shape[0] - 1, -1, -1):
            timestep = torch.full((batch_size,), value, dtype=torch.long, device=device)
            noise = torch.randn(x_t.shape, device=device, dtype=torch.float32, generator=generator)
            x_t = schedule.p_sample_from_noise(x_t, timestep, model(x_t, timestep), noise=noise, clip_denoised=True)["sample"]
    return x_t


def _generation_rows(model_name: str, records: list[dict[str, Any]], images: torch.Tensor, *, batch_index: int, batch_seed: int, checkpoint: Path, checkpoint_sha: str, root: Path, conditioning_consumed: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (record, image) in enumerate(zip(records, images)):
        sample_id = f"{model_name}_{record['image_id']}"
        relative = Path(model_name) / f"{sample_id}.png"
        hashes = save_canonical_png(generated_to_rgb_uint8(image), root / relative)
        rows.append({
            "sample_id": sample_id, "model": model_name, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checkpoint_sha, "evaluation_role": "NONSCIENTIFIC_DRYRUN", "split": "ID_VAL",
            "reference_sample_id": record["image_id"], "prompt": record["caption"],
            "generation_namespace": DRY_RUN_NAMESPACE, "batch_index": batch_index, "batch_seed": batch_seed,
            "sample_index_in_batch": index, "image_shape": [32, 32, 3], "image_mode": "RGB", "dtype": "uint8",
            "png_relative_path": relative.as_posix(), **hashes, "status": "complete",
            "conditioning_consumed": conditioning_consumed,
        })
    if any(set(row) != GENERATED_MANIFEST_FIELDS for row in rows):
        raise RuntimeError("generated manifest schema mismatch")
    return rows


def generate_dry_run_pool(records: list[dict[str, Any]], root: Path, *, device: torch.device) -> tuple[Path, str, Path, str]:
    assert_dry_run_records(records)
    schedule = LinearDDPMSchedule().to(device).eval()
    conditional_artifact = torch.load(CONDITIONAL, map_location="cpu", weights_only=False)
    text, conditional = _conditional_models_from_artifact(conditional_artifact)
    text, conditional = text.to(device).eval(), conditional.to(device).eval()
    if sum(parameter.numel() for parameter in text.parameters()) + sum(parameter.numel() for parameter in conditional.parameters()) != CONDITIONAL_PARAMETERS:
        raise ValueError("conditional canonical parameter count mismatch")
    ids, masks = _tokenize_records(records)
    conditional_rows: list[dict[str, Any]] = []
    for plan in dry_run_generation_plan():
        start = plan["batch_index"] * BATCH_SIZE
        end = start + BATCH_SIZE
        samples = conditional_sample(schedule, text, conditional, ids[start:end].to(device), masks[start:end].to(device), generator=torch.Generator(device=device).manual_seed(plan["batch_seed"]), clip_denoised=True)
        conditional_rows.extend(_generation_rows("conditional", records[start:end], samples, batch_index=plan["batch_index"], batch_seed=plan["batch_seed"], checkpoint=CONDITIONAL, checkpoint_sha=CONDITIONAL_SHA256, root=root, conditioning_consumed=True))
    del text, conditional, conditional_artifact
    if device.type == "cuda": torch.cuda.empty_cache()
    unconditional_artifact = torch.load(UNCONDITIONAL, map_location="cpu", weights_only=False)
    unconditional = _unconditional_model_from_artifact(unconditional_artifact).to(device).eval()
    if sum(parameter.numel() for parameter in unconditional.parameters()) != UNCONDITIONAL_PARAMETERS:
        raise ValueError("unconditional canonical parameter count mismatch")
    unconditional_rows: list[dict[str, Any]] = []
    for plan in dry_run_generation_plan():
        start = plan["batch_index"] * BATCH_SIZE
        end = start + BATCH_SIZE
        samples = _sample_unconditional(schedule, unconditional, batch_size=BATCH_SIZE, seed=plan["batch_seed"], device=device)
        unconditional_rows.extend(_generation_rows("unconditional", records[start:end], samples, batch_index=plan["batch_index"], batch_seed=plan["batch_seed"], checkpoint=UNCONDITIONAL, checkpoint_sha=UNCONDITIONAL_SHA256, root=root, conditioning_consumed=False))
    conditional_manifest = root / "manifests" / "conditional_manifest.jsonl"
    unconditional_manifest = root / "manifests" / "unconditional_manifest.jsonl"
    return conditional_manifest, _write_jsonl(conditional_manifest, conditional_rows), unconditional_manifest, _write_jsonl(unconditional_manifest, unconditional_rows)


@contextmanager
def deterministic_kid_rng(seed: int) -> Iterator[None]:
    """Narrow wrapper for clean-fid 0.1.35's global numpy KID sampling RNG."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def compute_clean_fid_metrics(real_dir: Path, generated_dir: Path, *, device: torch.device) -> dict[str, Any]:
    if importlib.metadata.version("clean-fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")
    from cleanfid import fid
    config = metric_config(len(list(real_dir.glob("*.png"))), len(list(generated_dir.glob("*.png"))))
    model = fid.build_feature_extractor(mode="clean", device=device, use_dataparallel=False)
    real_features = fid.get_folder_features(str(real_dir), model=model, num_workers=0, batch_size=BATCH_SIZE, device=device, mode="clean", verbose=False)
    generated_features = fid.get_folder_features(str(generated_dir), model=model, num_workers=0, batch_size=BATCH_SIZE, device=device, mode="clean", verbose=False)
    with deterministic_kid_rng(KID_SUBSET_SEED):
        kid_value = fid.kernel_distance(real_features, generated_features, num_subsets=100, max_subset_size=config["kid"]["subset_size"])
    fid_value = fid.frechet_distance(np.mean(real_features, axis=0), np.cov(real_features, rowvar=False), np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False))
    values = {"kid": float(kid_value), "fid": float(fid_value)}
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError("Clean-FID returned a non-finite dry-run metric")
    return {"configuration": config, "results": values}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _existing_pool_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Return a byte-level inventory without decoding or rewriting retained files."""
    inventory: dict[str, dict[str, Any]] = {}
    for relative in (Path("real_id_val"), Path("conditional"), Path("unconditional"), Path("manifests")):
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"missing retained I19 directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                key = path.resolve().relative_to(ROOT.resolve()).as_posix()
                inventory[key] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return inventory


def _resolved_retained_png(root: Path, relative: str, expected_directory: str) -> Path:
    path = (root / relative).resolve()
    directory = (root / expected_directory).resolve()
    if path.parent != directory or path.suffix.lower() != ".png":
        raise ValueError(f"retained PNG path escapes expected pool: {relative}")
    return path


def audit_retained_dry_run_pools(root: Path) -> dict[str, Any]:
    """Validate I19-derived pools only; this function never accesses project source images."""
    manifest_paths = {
        "real": root / "manifests" / "real_id_val_manifest.jsonl",
        "conditional": root / "manifests" / "conditional_manifest.jsonl",
        "unconditional": root / "manifests" / "unconditional_manifest.jsonl",
    }
    if any(path.name.endswith(".tmp") for path in root.rglob("*")):
        raise ValueError("retained dry-run output contains a temporary file")
    rows = {name: _read_jsonl(path) for name, path in manifest_paths.items()}
    manifest_hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    if manifest_hashes != RETAINED_I19_MANIFEST_SHA256:
        raise ValueError("retained I19 manifest SHA256 mismatch")
    if any(len(pool_rows) != DRY_RUN_COUNT for pool_rows in rows.values()):
        raise ValueError("retained manifests must each contain exactly 32 rows")
    pool_directories = {"real": "real_id_val", "conditional": "conditional", "unconditional": "unconditional"}
    png_paths: dict[str, set[str]] = {}
    for name, directory in pool_directories.items():
        paths = {path.relative_to(root).as_posix() for path in (root / directory).glob("*.png")}
        if len(paths) != DRY_RUN_COUNT or any(path.stat().st_size == 0 for path in (root / directory).glob("*.png")):
            raise ValueError(f"retained {name} pool must contain exactly 32 non-empty PNGs")
        png_paths[name] = paths
    if any(set(row) != REAL_MANIFEST_FIELDS for row in rows["real"]):
        raise ValueError("retained real manifest schema mismatch")
    for name in ("conditional", "unconditional"):
        if any(set(row) != GENERATED_MANIFEST_FIELDS for row in rows[name]):
            raise ValueError(f"retained {name} manifest schema mismatch")
    real_ids = [str(row["reference_sample_id"]) for row in rows["real"]]
    if len(set(real_ids)) != DRY_RUN_COUNT or real_ids != sorted(real_ids):
        raise ValueError("retained real reference IDs must be unique and sorted")
    if real_ids[0] != "cs10002930376453395152" or real_ids[-1] != "cs10096894922495846065":
        raise ValueError("retained real reference IDs differ from I19 selection")
    all_output_paths: set[str] = set()
    checked_pngs = 0
    for name, pool_rows in rows.items():
        directory = pool_directories[name]
        field = "canonical_png_relative_path" if name == "real" else "png_relative_path"
        row_paths = [str(row[field]) for row in pool_rows]
        if len(set(row_paths)) != DRY_RUN_COUNT or set(row_paths) != png_paths[name]:
            raise ValueError(f"retained {name} PNG paths do not match its manifest")
        if all_output_paths.intersection(row_paths):
            raise ValueError("retained pools have colliding output paths")
        all_output_paths.update(row_paths)
        for row in pool_rows:
            path = _resolved_retained_png(root, str(row[field]), directory)
            if sha256_file(path) != row["png_sha256"]:
                raise ValueError(f"retained PNG SHA256 mismatch: {path}")
            with Image.open(path) as image:
                pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
            if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row["pixel_sha256"]:
                raise ValueError(f"retained PNG pixel integrity mismatch: {path}")
            checked_pngs += 1
    expected_plan = dry_run_generation_plan()
    for name, expected_conditioning, checkpoint, checkpoint_sha in (
        ("conditional", True, CONDITIONAL, CONDITIONAL_SHA256),
        ("unconditional", False, UNCONDITIONAL, UNCONDITIONAL_SHA256),
    ):
        pool_rows = rows[name]
        if len({row["sample_id"] for row in pool_rows}) != DRY_RUN_COUNT:
            raise ValueError(f"retained {name} sample IDs are not unique")
        if any(row["conditioning_consumed"] is not expected_conditioning for row in pool_rows):
            raise ValueError(f"retained {name} conditioning flag mismatch")
        if any(row["checkpoint_path"] != checkpoint.relative_to(ROOT).as_posix() or row["checkpoint_sha256"] != checkpoint_sha for row in pool_rows):
            raise ValueError(f"retained {name} checkpoint mismatch")
        if any(row["generation_namespace"] != DRY_RUN_NAMESPACE or row["status"] != "complete" or row["image_shape"] != [32, 32, 3] or row["image_mode"] != "RGB" or row["dtype"] != "uint8" for row in pool_rows):
            raise ValueError(f"retained {name} generation metadata mismatch")
        observed_plan = sorted({(row["batch_index"], row["batch_seed"]) for row in pool_rows})
        if observed_plan != [(item["batch_index"], item["batch_seed"]) for item in expected_plan]:
            raise ValueError(f"retained {name} batch plan mismatch")
        if any(row["reference_sample_id"] not in set(real_ids) for row in pool_rows):
            raise ValueError(f"retained {name} reference mapping mismatch")
    return {
        "manifest_paths": {name: path.resolve().relative_to(ROOT.resolve()).as_posix() for name, path in manifest_paths.items()},
        "manifest_hashes": manifest_hashes,
        "pool_counts": {name: len(paths) for name, paths in png_paths.items()},
        "selected_reference_ids": real_ids,
        "png_and_pixel_sha_checked": checked_pngs,
        "cross_pool_audit": "PASS",
    }


def _environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "clean_fid": importlib.metadata.version("clean-fid"),
        "lpips": importlib.metadata.version("lpips"),
    }


def verify_recovery_environment() -> dict[str, str]:
    environment = _environment_metadata()
    expected = {"python": "3.12.4", "pytorch": "2.5.1+cu121", "numpy": "2.4.6", "scipy": "1.15.3", "clean_fid": CLEAN_FID_VERSION, "lpips": "0.1.4"}
    if environment != expected:
        raise RuntimeError(f"unexpected recovery environment: {environment}")
    return environment


def _verify_frozen_inputs() -> None:
    required = {
        PROTOCOL: PROTOCOL_SHA256, BINDINGS: BINDINGS_SHA256, CONDITIONAL: CONDITIONAL_SHA256,
        UNCONDITIONAL: UNCONDITIONAL_SHA256, EVALUATOR: EVALUATOR_SHA256,
        RUNTIME_BENCHMARK: RUNTIME_BENCHMARK_SHA256,
    }
    for path, expected in required.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"unexpected frozen artifact SHA256: {path} ({actual})")
    if importlib.metadata.version("clean-fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")


def run_dry_run_recovery(output: Path = OUTPUT) -> dict[str, Any]:
    """Run phases 2--4 against immutable I19 pools; phase 1 is deliberately absent."""
    _verify_frozen_inputs()
    environment = verify_recovery_environment()
    if not output.is_dir():
        raise FileNotFoundError(f"missing retained I19 dry-run root: {output}")
    summary_path = output / "dryrun_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite retained dry-run evidence: {summary_path}")
    before_inventory = _existing_pool_inventory(output)
    audit = audit_retained_dry_run_pools(output)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    conditional_metrics = compute_clean_fid_metrics(output / "real_id_val", output / "conditional", device=device)
    unconditional_metrics = compute_clean_fid_metrics(output / "real_id_val", output / "unconditional", device=device)
    after_inventory = _existing_pool_inventory(output)
    if before_inventory != after_inventory:
        raise RuntimeError("retained I19 pools or manifests changed during recovery")
    summary = {
        "schema": "dr05_final_quality_dryrun_recovery", "version": 1,
        "NONSCIENTIFIC_DRYRUN": True, "NOT_FOR_REPORT_RESULTS": True,
        "recovery": {
            "recovered_from_i19": True, "original_i19_verdict": "FAIL", "recovery_task": RECOVERY_TASK,
            "phase1_reused": True, "generation_executed_in_recovery": False,
            "original_source_images_opened_in_recovery": 0,
            "original_i19_source_decode_count_certified": False,
        },
        "protocol_sha256": PROTOCOL_SHA256, "bindings_sha256": BINDINGS_SHA256,
        "environment": environment, "pool_inventory": before_inventory,
        "integrity": {**audit, "manifest_schema": "PASS", "png_sha": "PASS", "pixel_sha": "PASS", "counts": "PASS"},
        "metrics": {"conditional_vs_real_id_val": conditional_metrics, "unconditional_vs_real_id_val": unconditional_metrics},
        "safety": {"sealed_data_not_accessed": True, "generator_executed": False, "evaluator_executed": False, "lpips_executed": False},
        "phase_completion": {"phase1": "inherited_verified", "phase2": "complete", "phase3": "complete", "phase4": "complete"},
        "wall_time_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_dry_run(output: Path = OUTPUT) -> dict[str, Any]:
    _verify_frozen_inputs()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dry-run output: {output}")
    records = select_dry_run_records()
    output.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    real_manifest, real_manifest_sha = canonicalize_real_pool(records, output)
    conditional_manifest, conditional_manifest_sha, unconditional_manifest, unconditional_manifest_sha = generate_dry_run_pool(records, output, device=device)
    conditional_metrics = compute_clean_fid_metrics(output / "real_id_val", output / "conditional", device=device)
    unconditional_metrics = compute_clean_fid_metrics(output / "real_id_val", output / "unconditional", device=device)
    summary = {
        "schema": "dr05_final_quality_dryrun", "version": 1, "NONSCIENTIFIC_DRYRUN": True, "NOT_FOR_REPORT_RESULTS": True,
        "protocol_sha256": PROTOCOL_SHA256, "bindings_sha256": BINDINGS_SHA256,
        "checkpoints": {"conditional": CONDITIONAL_SHA256, "unconditional": UNCONDITIONAL_SHA256},
        "clean_fid_version": CLEAN_FID_VERSION, "source_selection": {"split": "ID_VAL", "policy": "lexicographically first 32 image_id values", "selected_sample_ids": [item["image_id"] for item in records]},
        "generation": {"namespace": DRY_RUN_NAMESPACE, "seed_base": DRY_RUN_SEED_BASE, "batch_size": BATCH_SIZE, "batch_seeds": [item["batch_seed"] for item in dry_run_generation_plan()], "timesteps": 1000, "sampler": "ancestral_ddpm_fixed_small", "clip_denoised": True},
        "final_plan_design": {"layout": FINAL_LAYOUT, "allocations": FINAL_ALLOCATIONS, "namespace": FINAL_NAMESPACE, "seed_base": FINAL_SEED_BASE, "batch_size": BATCH_SIZE},
        "manifests": {"real": {"path": real_manifest.relative_to(ROOT).as_posix(), "sha256": real_manifest_sha}, "conditional": {"path": conditional_manifest.relative_to(ROOT).as_posix(), "sha256": conditional_manifest_sha}, "unconditional": {"path": unconditional_manifest.relative_to(ROOT).as_posix(), "sha256": unconditional_manifest_sha}},
        "metrics": {"conditional_vs_real_id_val": conditional_metrics, "unconditional_vs_real_id_val": unconditional_metrics},
        "integrity": {"png_round_trip_and_pixel_sha": "PASS"},
        "sealed_data_compliance": {"id_val_metadata_accessed": True, "id_val_images_opened": DRY_RUN_COUNT, "comp_val_images_opened": 0, "id_test_images_opened": 0, "final_ood_images_opened": 0, "final_generated_images": 0, "evaluator_runs": 0, "lpips_runs": 0},
        "final_mode_guard": {"default": "reject", "requires_confirm_sealed_final_eval": True, "dry_run_rejects_non_id_val": True},
        "wall_time_seconds": time.perf_counter() - started,
    }
    summary_path = output / "dryrun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "final"))
    parser.add_argument("--confirm-sealed-final-eval", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--final-output", type=Path, default=FINAL_OUTPUT)
    parser.add_argument("--resume-from-existing-pools", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.status_only:
        if args.mode is not None or args.confirm_sealed_final_eval:
            raise ValueError("--status-only is read-only and cannot be combined with execution flags")
        print(json.dumps(final_status_only(args.final_output), indent=2, sort_keys=True))
        return
    mode = validate_mode(args.mode, confirm_sealed_final_eval=args.confirm_sealed_final_eval)
    if mode == "final":
        print(json.dumps(execute_final_quality(output=args.final_output), indent=2, sort_keys=True))
        return
    runner = run_dry_run_recovery if args.resume_from_existing_pools else run_dry_run
    print(json.dumps(runner(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
