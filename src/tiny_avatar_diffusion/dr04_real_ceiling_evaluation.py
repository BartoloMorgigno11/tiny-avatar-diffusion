"""D22 real-ceiling evaluation for the frozen DR04 evaluator."""
from __future__ import annotations

import csv, hashlib, json, platform, subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tiny_avatar_diffusion.evaluator import (CACHE_ROOT, CLASS_COUNTS, ROOT, SPLIT_ROOT, AttributeEvaluator, EvaluatorDataset, class_weight_artifact, parameter_count)

RUN_ID = "dr04_evaluator_main_100k_seed2026_r1"
RUN_ROOT = ROOT / "results" / "runs" / RUN_ID
OUTPUT_ROOT = ROOT / "results" / "evaluation" / "dr04_real_ceiling_v1"
FREEZE = SPLIT_ROOT / "evaluator_checkpoint_freeze.json"
ALLOWED_CEILINGS = ("REAL_ID_CEILING", "REAL_GENERATOR_OOD_CEILING")
ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
OOD_PAIRS = {"OOD1": ("face_shape", "glasses"), "OOD2": ("face_shape", "hair_color"), "OOD3": ("glasses", "face_color"), "OOD4": ("hair_color", "face_color")}

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def _guard_ceiling(split: str) -> None:
    if split not in ALLOWED_CEILINGS:
        raise ValueError(f"Only real ceiling splits are permitted: {split}")

def confusion_matrix(target: list[int], prediction: list[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for actual, predicted in zip(target, prediction, strict=True): matrix[actual][predicted] += 1
    return matrix

def attribute_metrics(target: list[int], prediction: list[int], classes: int, train_frequency: list[float]) -> dict[str, Any]:
    matrix = confusion_matrix(target, prediction, classes); support = [sum(row) for row in matrix]; total = len(target)
    accuracy = sum(matrix[index][index] for index in range(classes)) / total
    f1, recalls = [], []
    for index in range(classes):
        true_positive = matrix[index][index]; false_positive = sum(matrix[row][index] for row in range(classes) if row != index); false_negative = sum(matrix[index][column] for column in range(classes) if column != index)
        f1.append(0.0 if 2 * true_positive + false_positive + false_negative == 0 else 2 * true_positive / (2 * true_positive + false_positive + false_negative))
        recalls.append(0.0 if support[index] == 0 else true_positive / support[index])
    target_frequency = [count / total for count in support]
    return {"accuracy": accuracy, "macro_f1": sum(f1) / classes, "balanced_accuracy": sum(recalls) / classes, "support": support, "confusion_matrix": matrix, "baselines": {"uniform_random_expected_accuracy": 1 / classes, "majority_accuracy": max(support) / total, "frequency_aware_random_expected_accuracy": sum(p * q for p, q in zip(target_frequency, train_frequency, strict=True))}}

def joint_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    return {"mean_attribute_accuracy": sum(sum(record[f"{attribute}_correct"] for record in records) / len(records) for attribute in ATTRIBUTES) / len(ATTRIBUTES), "exact_attribute_match": sum(all(record[f"{attribute}_correct"] for attribute in ATTRIBUTES) for record in records) / len(records)}

def ood_pair_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source, (first, second) in OOD_PAIRS.items():
        group = [record for record in records if record["source_ood"] == source]
        if len(group) != 100: raise ValueError(f"Expected 100 {source} OOD ceiling records")
        first_ok = [record[f"{first}_correct"] for record in group]; second_ok = [record[f"{second}_correct"] for record in group]
        result.append({"source_ood": source, "first_attribute": first, "second_attribute": second, "n": len(group), "first_attribute_success": sum(first_ok) / len(group), "second_attribute_success": sum(second_ok) / len(group), "both_correct": sum(a and b for a, b in zip(first_ok, second_ok, strict=True)) / len(group), "neither_correct": sum(not a and not b for a, b in zip(first_ok, second_ok, strict=True)) / len(group)})
    return result

def macro_psr(pair_diagnostics: list[dict[str, Any]]) -> float:
    return sum(row["both_correct"] for row in pair_diagnostics) / len(pair_diagnostics)

def matched_id_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source, (first, second) in OOD_PAIRS.items():
        for direction in ("ALL", "A_FIXED", "B_FIXED"):
            group = [record for record in records if record["source_ood"] == source and (direction == "ALL" or record["control_direction"] == direction)]
            expected = 100 if direction == "ALL" else 50
            if len(group) != expected: raise ValueError(f"Expected {expected} {source} {direction} controls")
            result.append({"source_ood": source, "control_direction": direction, "n": len(group), "first_attribute": first, "first_attribute_accuracy": sum(record[f"{first}_correct"] for record in group) / len(group), "second_attribute": second, "second_attribute_accuracy": sum(record[f"{second}_correct"] for record in group) / len(group)})
    return result

def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def _frequency() -> dict[str, list[float]]:
    artifact = class_weight_artifact()
    return {attribute: [count / artifact["total_train_examples"] for count in item["train_counts"]] for attribute, item in artifact["attributes"].items()}

@torch.no_grad()
def evaluate_ceiling(split: str, model: AttributeEvaluator, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _guard_ceiling(split); dataset = EvaluatorDataset(split, include_image_id=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    rows_by_id = {row["image_id"]: row for row in dataset.rows}; records: list[dict[str, Any]] = []
    model.eval()
    for image, labels, image_ids in loader:
        logits = model(image.to(device, dtype=torch.float32)); probabilities = {attribute: torch.softmax(value, dim=1).cpu() for attribute, value in logits.items()}
        for index, image_id in enumerate(image_ids):
            source = rows_by_id[image_id]; record: dict[str, Any] = {"example_id": image_id, "ceiling_domain": split, "source_ood": source["source_ood"], "control_direction": source["control_direction"]}
            for attribute in ATTRIBUTES:
                actual = int(labels[attribute][index]); predicted = int(probabilities[attribute][index].argmax())
                record[f"{attribute}_true"] = actual; record[f"{attribute}_predicted"] = predicted; record[f"{attribute}_correct"] = actual == predicted; record[f"{attribute}_confidence"] = float(probabilities[attribute][index][predicted])
            records.append(record)
    if len(records) != 400: raise ValueError(f"Expected 400 {split} records")
    frequency = _frequency(); metrics = {attribute: attribute_metrics([record[f"{attribute}_true"] for record in records], [record[f"{attribute}_predicted"] for record in records], CLASS_COUNTS[attribute], frequency[attribute]) for attribute in ATTRIBUTES}
    return records, {"n": len(records), "attributes": metrics, "joint": joint_metrics(records)}

def _provenance() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8")); checkpoint = RUN_ROOT / "best.pt"
    if _sha(FREEZE) != "85b0cef08bb5a298442d7d863c4f76b114ecf023335d312f319c6feef168f13f" or _sha(checkpoint) != freeze["best_checkpoint_sha256"]: raise RuntimeError("Frozen evaluator artifact hash mismatch")
    return {"timestamp_utc": datetime.now(UTC).isoformat(), "implementation_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "evaluator_freeze_manifest_sha256": _sha(FREEZE), "evaluator_best_checkpoint_sha256": _sha(checkpoint), "split_manifest_sha256": freeze["split_manifest_sha256"], "class_weights_sha256": freeze["class_weights_sha256"], "preprocessing_identity": "rgb_bilinear_antialias_32_float32_minus1_to1_v1", "precision": "fp32", "ceiling_sizes": {split: 400 for split in ALLOWED_CEILINGS}, "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "parameter_count": parameter_count(), "sealed_generator_splits_accessed": False, "generated_ddpm_images_accessed": False}

def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.exists(): raise RuntimeError(f"Refusing to overwrite existing evaluation results: {output_root}")
    provenance = _provenance(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = AttributeEvaluator().to(device)
    payload = torch.load(RUN_ROOT / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(payload["model_state_dict"])
    output_root.mkdir(parents=True); (output_root / "confusion_matrices").mkdir()
    all_records, summaries = {}, {}
    for split in ALLOWED_CEILINGS:
        records, summary = evaluate_ceiling(split, model, device); all_records[split] = records; summaries[split] = summary
    per_attribute = []
    for split, summary in summaries.items():
        for attribute, metric in summary["attributes"].items(): per_attribute.append({"ceiling_domain": split, "attribute": attribute, "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"], **metric["baselines"]})
    _write_csv(output_root / "per_attribute_metrics.csv", per_attribute, list(per_attribute[0]))
    for split, summary in summaries.items():
        for attribute, metric in summary["attributes"].items():
            matrix_rows = [{"true_class_id": actual, **{f"predicted_{predicted}": count for predicted, count in enumerate(row)}} for actual, row in enumerate(metric["confusion_matrix"])]
            _write_csv(output_root / "confusion_matrices" / f"{split.lower()}_{attribute}.csv", matrix_rows, list(matrix_rows[0]))
    misclassified = [record for records in all_records.values() for record in records if not all(record[f"{attribute}_correct"] for attribute in ATTRIBUTES)]
    error_fields = ["example_id", "ceiling_domain", "source_ood", "control_direction", *[f"{attribute}_{suffix}" for attribute in ATTRIBUTES for suffix in ("true", "predicted", "correct", "confidence")]]
    _write_csv(output_root / "misclassifications.csv", misclassified, error_fields)
    matched = matched_id_metrics(all_records["REAL_ID_CEILING"]); _write_csv(output_root / "matched_id_diagnostics.csv", matched, list(matched[0]))
    pairs = ood_pair_metrics(all_records["REAL_GENERATOR_OOD_CEILING"]); _write_csv(output_root / "ood_pair_diagnostics.csv", pairs, list(pairs[0]))
    summaries["ood_pair_diagnostics"] = pairs; summaries["macro_psr"] = macro_psr(pairs); summaries["misclassification_count"] = {split: sum(not all(row[f"{attribute}_correct"] for attribute in ATTRIBUTES) for row in records) for split, records in all_records.items()}
    _json(output_root / "metrics_summary.json", summaries)
    artifacts = {str(path.relative_to(output_root)).replace("\\", "/"): _sha(path) for path in sorted(output_root.rglob("*")) if path.is_file()}
    provenance["artifact_sha256"] = artifacts; _json(output_root / "provenance.json", provenance)
    return summaries

if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
