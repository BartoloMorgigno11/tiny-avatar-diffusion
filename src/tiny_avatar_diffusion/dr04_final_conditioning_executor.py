"""Durable, hash-bound execution primitives for the frozen DR04 final panel.

The module deliberately receives generation and evaluator functions as callbacks.
Consequently its state machine and all metric semantics can be tested with tiny
synthetic data without unsealing or producing scientific images.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from tiny_avatar_diffusion.dr05_final_executor import atomic_json, atomic_jsonl, load_jsonl, pixel_sha256, save_png, sha256_file
from tiny_avatar_diffusion.dr04_real_ceiling_evaluation import attribute_metrics
from tiny_avatar_diffusion.evaluator import CLASS_COUNTS

ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
SEEDS = (2026, 2027, 2028, 2029, 2030)
OOD_FAMILIES = ("OOD1", "OOD2", "OOD3", "OOD4")
FROZEN_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "data/processed/dr04_final_eval_v1/final_evaluation_protocol.json"
FROZEN_PARAMETER_COUNTS = {
    "conditional_generator": {"total": 4_723_907, "unet_backbone": 4_595_779, "cross_attention": 25_152, "text_encoder": 102_976},
    "evaluator": 293_064,
}
TIMING_FIELDS = ("generation", "integrity", "evaluator_inference", "metrics", "total")


def canonical_sha256(path: Path) -> str:
    return sha256_file(path)


def frozen_bootstrap_config() -> dict[str, Any]:
    """Load and validate the bootstrap settings from the frozen DR04 protocol."""
    protocol = json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))
    config = protocol.get("bootstrap")
    expected = {
        "confidence_level": 0.95,
        "method": "percentile",
        "repetitions": 10_000,
        "seed": 2026,
        "cluster_unit": "prompt",
        "all_seeds_clustered": True,
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("invalid frozen DR04 bootstrap configuration")
    return {key: config[key] for key in expected}


def validate_parameter_counts(counts: dict[str, Any]) -> dict[str, Any]:
    """Reject any finalization count that differs from the frozen architecture."""
    expected = FROZEN_PARAMETER_COUNTS
    if not isinstance(counts, dict) or set(counts) != set(expected) or counts.get("conditional_generator") != expected["conditional_generator"] or counts.get("evaluator") != expected["evaluator"]:
        raise ValueError("frozen parameter-count mismatch")
    return {"conditional_generator": dict(expected["conditional_generator"]), "evaluator": expected["evaluator"]}


def collect_parameter_counts(unet_backbone: Any, cross_attention: Any, text_encoder: Any, evaluator: Any) -> dict[str, Any]:
    """Count the separately loaded frozen modules before a real final summary."""
    def count(model: Any) -> int:
        return sum(int(parameter.numel()) for parameter in model.parameters())
    unet, cross, text, evaluator_total = count(unet_backbone), count(cross_attention), count(text_encoder), count(evaluator)
    return validate_parameter_counts({"conditional_generator": {"total": unet + cross + text, "unet_backbone": unet, "cross_attention": cross, "text_encoder": text}, "evaluator": evaluator_total})


def secondary_attribute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen per-attribute evaluator metrics with canonical numeric class order."""
    if not predictions:
        raise ValueError("secondary metrics require predictions")
    result: dict[str, Any] = {}
    for attribute in ATTRIBUTES:
        classes = CLASS_COUNTS[attribute]
        targets = [row.get("targets", {}).get(attribute) for row in predictions]
        values = [row.get("predictions", {}).get(attribute) for row in predictions]
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < classes for value in [*targets, *values]):
            raise ValueError(f"invalid {attribute} evaluator label")
        metric = attribute_metrics(targets, values, classes, [1 / classes] * classes)
        result[attribute] = {"label_order": list(range(classes)), "confusion_matrix": metric["confusion_matrix"], "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"]}
    return result


def validate_timings(timings: dict[str, Any]) -> dict[str, float]:
    if set(timings) != set(TIMING_FIELDS):
        raise ValueError("incomplete final operational timing report")
    result = {name: float(timings[name]) for name in TIMING_FIELDS}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("invalid final operational timing report")
    return result


def validate_panel(panel: dict[str, Any], controls: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = panel.get("prompt_records", [])
    if len(rows) != 252 or len({r.get("prompt_id") for r in rows}) != 252:
        raise ValueError("frozen panel must contain 252 unique prompts")
    ids = [r for r in rows if r.get("domain") == "ID"]
    ood = [r for r in rows if r.get("domain") == "OOD"]
    if len(ids) != 152 or len(ood) != 100:
        raise ValueError("frozen panel ID/OOD counts mismatch")
    if any(sum(r.get("pair_id") == family for r in ood) != 25 for family in ("OOD1", "OOD2", "OOD3", "OOD4")):
        raise ValueError("frozen OOD family geometry mismatch")
    if any(sum(r.get("source_pair_id") == family and r.get("control_type") == role for r in ids) != 19 for family in ("OOD1", "OOD2", "OOD3", "OOD4") for role in ("A_FIXED", "B_FIXED")):
        raise ValueError("frozen matched-control geometry mismatch")
    relations = defaultdict(list)
    for row in ids: relations[row.get("matched_ood_prompt_id")].append(row)
    counts = [len(relations[row["prompt_id"]]) for row in ood]
    if counts.count(2) != 76 or counts.count(0) != 24 or any(count not in (0, 2) for count in counts):
        raise ValueError("frozen matched-control coverage mismatch")
    if controls is not None and controls.get("prompt_records") != ids:
        raise ValueError("matched controls artifact does not bind panel controls")
    return rows


def expand_plan(rows: list[dict[str, Any]], checkpoint_path: str, checkpoint_sha256: str) -> list[dict[str, Any]]:
    """Stable ID-then-OOD, prompt-id and seed ordered frozen expansion."""
    expanded = []
    ordered = sorted(rows, key=lambda r: (0 if r["domain"] == "ID" else 1, r["prompt_id"]))
    for prompt in ordered:
        for seed in SEEDS:
            index = len(expanded)
            expanded.append({"planned_sample_index": index, "sample_id": f"dr04_main_{index:04d}", "prompt_id": prompt["prompt_id"], "prompt": prompt["caption"], "panel_split": prompt["domain"], "ood_family": prompt.get("pair_id", prompt.get("source_pair_id")), "control_role": prompt.get("control_type"), "matched_ood_prompt_id": prompt.get("matched_ood_prompt_id"), "seed": seed, "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha256, "targets": {key: int(prompt[key]) for key in ATTRIBUTES}})
    if len(expanded) != 1260 or sum(r["panel_split"] == "ID" for r in expanded) != 760 or len({(r["prompt_id"], r["seed"]) for r in expanded}) != 1260:
        raise ValueError("invalid final plan expansion")
    return expanded


def _json_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def prompt_cluster_bootstrap(values: dict[str, list[float]], *, repetitions: int | None = None, seed: int | None = None) -> list[float]:
    """Percentile-bootstrap draws using prompts, never individual seed images."""
    config = frozen_bootstrap_config()
    repetitions = config["repetitions"] if repetitions is None else repetitions
    seed = config["seed"] if seed is None else seed
    if repetitions != config["repetitions"] or seed != config["seed"]:
        raise ValueError("DR04 bootstrap must use the frozen protocol repetitions and seed")
    clusters = [np.asarray(v, dtype=float) for _, v in sorted(values.items())]
    if not clusters or any(v.shape != (5,) for v in clusters):
        raise ValueError("each bootstrap unit must be exactly one five-seed prompt cluster")
    rng = np.random.default_rng(seed)
    return [float(np.mean([clusters[i].mean() for i in rng.integers(0, len(clusters), len(clusters))])) for _ in range(repetitions)]


def ood_prediction_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the canonical OOD-only 25-prompt, five-seed PSR populations."""
    ood_rows = [row for row in rows if row.get("panel_split") == "OOD"]
    if not ood_rows:
        return {}
    groups = {family: [row for row in ood_rows if row.get("ood_family") == family] for family in OOD_FAMILIES}
    if any(row.get("ood_family") not in OOD_FAMILIES for row in ood_rows):
        raise ValueError("OOD prediction has missing or unknown family")
    if len(ood_rows) != sum(len(group) for group in groups.values()):
        raise ValueError("OOD prediction belongs to multiple families")
    for family, group in groups.items():
        if len(group) != 125:
            raise ValueError("OOD family must have 25 x 5 predictions")
        prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            prompt_groups[row.get("prompt_id")].append(row)
        if len(prompt_groups) != 25 or any(len(items) != 5 or {item.get("seed") for item in items} != set(SEEDS) for items in prompt_groups.values()):
            raise ValueError("OOD family must have 25 unique five-seed prompt clusters")
        if len({(row.get("prompt_id"), row.get("seed")) for row in group}) != len(group):
            raise ValueError("OOD family has duplicate prompt-seed prediction")
    return groups


def bootstrap_summary(prompt_values: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen prompt-cluster MAA uncertainty, persisted by reporting scope."""
    config = frozen_bootstrap_config()
    scopes: dict[str, list[dict[str, Any]]] = {"ID": [], "OOD": [], **{f: [] for f in ("OOD1", "OOD2", "OOD3", "OOD4")}}
    for item in prompt_values:
        scopes["ID" if item["split"] == "ID" else "OOD"].append(item)
        if item["split"] == "OOD" and item.get("ood_family") in scopes: scopes[item["ood_family"]].append(item)
    result: dict[str, Any] = {"resamples": config["repetitions"], "seed": config["seed"], "unit": config["cluster_unit"], "all_five_seeds_clustered": config["all_seeds_clustered"], "scopes": {}}
    for name, items in scopes.items():
        if items:
            clusters = {x["prompt_id"]: [float(np.mean(list(x["attributes"].values())))] * 5 for x in items}
            draws = prompt_cluster_bootstrap(clusters)
            result["scopes"][name] = {"estimate": float(np.mean([np.mean(v) for v in clusters.values()])), "percentile_95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
    return result


def prompt_values_from_canonical_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the frozen prompt clusters solely from the completed canonical predictions."""
    if len(predictions) != 1260 or sum(row.get("panel_split") == "ID" for row in predictions) != 760 or sum(row.get("panel_split") == "OOD" for row in predictions) != 500:
        raise ValueError("canonical prediction population mismatch")
    if len({row.get("sample_id") for row in predictions}) != 1260:
        raise ValueError("duplicate canonical prediction sample")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[row.get("prompt_id")].append(row)
    if len(groups) != 252 or any(len(group) != 5 or {row.get("seed") for row in group} != set(SEEDS) for group in groups.values()):
        raise ValueError("canonical prompt-seed clustering mismatch")
    ood_prediction_groups(predictions)
    return [{"prompt_id": key, "split": group[0]["panel_split"], "ood_family": group[0]["ood_family"], "attributes": {attribute: float(np.mean([row["correct"][attribute] for row in group])) for attribute in ATTRIBUTES}, "eam": float(np.mean([all(row["correct"].values()) for row in group]))} for key, group in groups.items()]


def id_ood_gap_bootstrap(prompt_values: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the frozen absolute MAA gap with independent prompt resampling."""
    config = frozen_bootstrap_config()
    scores = {
        split: [float(np.mean(list(item["attributes"].values()))) for item in prompt_values if item["split"] == split]
        for split in ("ID", "OOD")
    }
    if not scores["ID"] or not scores["OOD"]:
        raise ValueError("ID and OOD prompt populations are required for the MAA gap")
    rng = np.random.default_rng(config["seed"])
    id_scores, ood_scores = np.asarray(scores["ID"], dtype=float), np.asarray(scores["OOD"], dtype=float)
    draws = [float(id_scores[rng.integers(0, len(id_scores), len(id_scores))].mean() - ood_scores[rng.integers(0, len(ood_scores), len(ood_scores))].mean()) for _ in range(config["repetitions"])]
    return {"formula": "delta_maa = maa_id - maa_ood", "estimate": float(id_scores.mean() - ood_scores.mean()), "percentile_95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "bootstrap": {"confidence_level": config["confidence_level"], "method": config["method"], "repetitions": config["repetitions"], "seed": config["seed"], "cluster_unit": config["cluster_unit"], "all_five_seeds_clustered": config["all_seeds_clustered"], "id_ood_resampling": "independent"}}


def build_release_addendum(root: Path, *, vocabulary_path: str, vocabulary_sha256: str) -> dict[str, Any]:
    """Build derived reporting provenance without changing finalized DR04 evidence."""
    prediction_path, summary_path = root / "manifests" / "predictions.jsonl", root / "final_conditioning_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = summary.get("run_identity", {})
    predictions = load_jsonl(prediction_path)
    return {"schema": "dr04_release_provenance_addendum", "version": 1, "derived_reporting_only": True, "no_samples_regenerated": True, "source_artifacts": {"prediction_manifest_path": _portable_path(prediction_path), "prediction_manifest_sha256": sha256_file(prediction_path), "final_summary_path": _portable_path(summary_path), "final_summary_sha256": sha256_file(summary_path), "plan_sha256": identity["plan_sha256"], "generator_checkpoint_sha256": identity["conditional_checkpoint_sha256"], "evaluator_checkpoint_sha256": identity["evaluator_checkpoint_sha256"], "vocabulary_path": vocabulary_path, "vocabulary_sha256": vocabulary_sha256}, "id_ood_gap": id_ood_gap_bootstrap(prompt_values_from_canonical_predictions(predictions))}


def materialize_release_addendum(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Create an immutable release addendum without overwriting prior evidence."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError("release addendum already exists with a different payload")
        return existing
    atomic_json(path, payload)
    return payload


def matched_control_diagnostics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Existing DR04 secondary A_FIXED/B_FIXED attribute-accuracy diagnostic."""
    pairs = {"OOD1": ("face_shape", "glasses"), "OOD2": ("face_shape", "hair_color"), "OOD3": ("glasses", "face_color"), "OOD4": ("hair_color", "face_color")}
    output = []
    for family, attrs in pairs.items():
        for role in ("ALL", "A_FIXED", "B_FIXED"):
            group = [r for r in predictions if r.get("ood_family") == family and r.get("control_role") in (("A_FIXED", "B_FIXED") if role == "ALL" else (role,))]
            expected = 190 if role == "ALL" else 95
            if len(group) != expected: raise ValueError(f"expected {expected} matched-control seed rows for {family} {role}")
            output.append({"ood_family": family, "control_role": role, "n": len(group), "first_attribute": attrs[0], "first_attribute_accuracy": float(np.mean([r["correct"][attrs[0]] for r in group])), "second_attribute": attrs[1], "second_attribute_accuracy": float(np.mean([r["correct"][attrs[1]] for r in group]))})
    return {"applicability": "secondary diagnostic; primary MAA/MacroPSR unaffected", "coverage": {"both": 76, "neither": 24, "one_only": 0}, "rows": output}


class FinalConditioningExecutor:
    """Five-phase executor.  It has no implicit model loading or final-run side effects."""
    def __init__(self, root: Path, plan: list[dict[str, Any]], identities: dict[str, str], *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.root, self.plan, self.identities = root, plan, identities
        self.clock = clock
        self.state_path = root / "phase_status" / "execution_state.json"
        self.manifest_path = root / "manifests" / "generated_images.jsonl"
        self.predictions_path = root / "manifests" / "predictions.jsonl"
        if [r.get("planned_sample_index") for r in plan] != list(range(len(plan))) or len({r.get("sample_id") for r in plan}) != len(plan):
            raise ValueError("plan must be contiguous and unique")

    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("plan_sha256") != _json_sha(self.plan) or state.get("identities") != self.identities: raise ValueError("run identity mismatch")
            return state
        if (self.root / "final_conditioning_summary.json").exists(): raise FileExistsError("summary exists without state")
        state = {"schema": "dr04_final_conditioning_state", "version": 2, "plan_sha256": _json_sha(self.plan), "identities": self.identities, "phase": 0, "slots": {}, "retries": {}, "integrity": "pending", "predictions": "pending", "metrics": "pending", "summary": "pending", "started_perf_counter": self.clock(), "timings_seconds": {}, "parameter_counts": None}
        atomic_json(self.state_path, state); return state

    def _write(self, state: dict[str, Any]) -> None: atomic_json(self.state_path, state)
    def rows(self) -> list[dict[str, Any]]: return load_jsonl(self.manifest_path) if self.manifest_path.exists() else []

    def record_parameter_counts(self, counts: dict[str, Any]) -> None:
        state = self.state()
        state["parameter_counts"] = validate_parameter_counts(counts)
        self._write(state)

    def _finish_timed_phase(self, state: dict[str, Any], name: str, started: float) -> None:
        elapsed = self.clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"invalid {name} operational timing")
        state["timings_seconds"][name] = elapsed

    def generate(self, generator: Callable[[dict[str, Any]], np.ndarray]) -> None:
        started = self.clock(); state, existing = self.state(), {r["sample_id"]: r for r in self.rows()}
        for slot in self.plan:
            old = existing.get(slot["sample_id"])
            if old and old.get("status") == "completed": continue
            if old:
                (self.root / old["png_path"]).unlink(missing_ok=True); existing.pop(slot["sample_id"], None)
            state["slots"][slot["sample_id"]] = "in_progress"; self._write(state)
            try:
                pixels = generator(slot)
                relative = Path("images") / f"{slot['sample_id']}.png"; hashes = save_png(pixels, self.root / relative)
                row = {**slot, "conditioning_consumed": True, "shape": [32, 32, 3], "mode": "RGB", "dtype": "uint8", "png_path": relative.as_posix(), **hashes, "status": "completed"}
                existing[slot["sample_id"]] = row; atomic_jsonl(self.manifest_path, list(existing.values()))
                state["slots"][slot["sample_id"]] = "completed"; self._write(state)
            except Exception:
                state["slots"][slot["sample_id"]] = "failed_retriable"; state["retries"][slot["sample_id"]] = state["retries"].get(slot["sample_id"], 0) + 1; self._write(state); raise
        state["phase"] = 1; self._finish_timed_phase(state, "generation", started); self._write(state)

    def audit(self) -> None:
        started = self.clock(); state, rows = self.state(), self.rows(); planned = {r["sample_id"]: r for r in self.plan}
        if len(rows) != len(self.plan) or len({r.get("sample_id") for r in rows}) != len(rows) or len({r.get("png_path") for r in rows}) != len(rows): raise ValueError("manifest population/uniqueness mismatch")
        for row in rows:
            slot = planned.get(row.get("sample_id"))
            if slot is None or any(row.get(k) != v for k, v in slot.items()) or row.get("status") != "completed": raise ValueError("plan row mismatch")
            path = self.root / row["png_path"]
            if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != row["png_sha256"]: raise ValueError("PNG integrity mismatch")
            with Image.open(path) as im:
                if im.mode != "RGB" or im.size != (32, 32): raise ValueError("image shape/mode mismatch")
                pixels = np.asarray(im, dtype=np.uint8)
            if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row["pixel_sha256"]: raise ValueError("pixel integrity mismatch")
        state["integrity"] = "complete"; state["generation_manifest_sha256"] = sha256_file(self.manifest_path); state["phase"] = 2; self._finish_timed_phase(state, "integrity", started); self._write(state)

    def predict(self, evaluator: Callable[[np.ndarray], dict[str, int]]) -> None:
        started = self.clock(); state = self.state()
        if state["integrity"] != "complete": raise RuntimeError("integrity must pass before evaluator inference")
        output = []
        for row in self.rows():
            with Image.open(self.root / row["png_path"]) as image: prediction = evaluator(np.asarray(image.convert("RGB"), dtype=np.uint8))
            if set(prediction) != set(ATTRIBUTES): raise ValueError("evaluator output schema mismatch")
            output.append({"sample_id": row["sample_id"], "prompt_id": row["prompt_id"], "seed": row["seed"], "panel_split": row["panel_split"], "ood_family": row["ood_family"], "control_role": row["control_role"], "matched_ood_prompt_id": row["matched_ood_prompt_id"], "targets": row["targets"], "predictions": {k: int(prediction[k]) for k in ATTRIBUTES}, "correct": {k: int(prediction[k]) == row["targets"][k] for k in ATTRIBUTES}, "evaluator_checkpoint_path": self.identities["evaluator_checkpoint_path"], "evaluator_checkpoint_sha256": self.identities["evaluator_checkpoint_sha256"]})
        if len(output) != len(self.plan) or len({r["sample_id"] for r in output}) != len(output): raise ValueError("prediction population mismatch")
        atomic_jsonl(self.predictions_path, output); state["predictions"] = "complete"; state["prediction_manifest_sha256"] = sha256_file(self.predictions_path); state["phase"] = 3; self._finish_timed_phase(state, "evaluator_inference", started); self._write(state)

    def metrics(self) -> dict[str, Any]:
        started = self.clock(); state = self.state()
        if state["predictions"] != "complete": raise RuntimeError("predictions must complete before metrics")
        rows = load_jsonl(self.predictions_path); groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: groups[row["prompt_id"]].append(row)
        if any(len(group) != 5 for group in groups.values()): raise ValueError("five-seed prompt clusters required")
        prompt = [{"prompt_id": key, "split": group[0]["panel_split"], "ood_family": group[0]["ood_family"], "attributes": {a: float(np.mean([r["correct"][a] for r in group])) for a in ATTRIBUTES}, "eam": float(np.mean([all(r["correct"].values()) for r in group]))} for key, group in groups.items()]
        def maa(items: list[dict[str, Any]]) -> float | None:
            values = [v for item in items for v in item["attributes"].values()]
            return float(np.mean(values)) if values else None
        result = {"ID_MAA": maa([x for x in prompt if x["split"] == "ID"]), "OOD_MAA": maa([x for x in prompt if x["split"] == "OOD"]), "prompt_level": prompt}
        psr = {}
        # Tiny MAA fixtures may deliberately omit compositional families.
        psr_groups = {} if {row.get("ood_family") for row in rows if row.get("panel_split") == "OOD"} == {None} else ood_prediction_groups(rows)
        for family, items in psr_groups.items():
            # Frozen pairs are the two attributes encoded by each OOD family; infer them from panel targets is intentionally avoided.
            pair = {"OOD1": ("face_shape", "glasses"), "OOD2": ("face_shape", "hair_color"), "OOD3": ("glasses", "face_color"), "OOD4": ("hair_color", "face_color")}[family]
            outcomes = [tuple(bool(r["correct"][a]) for a in pair) for r in items]
            psr[family] = {"PSR": float(np.mean([a and b for a,b in outcomes])), **{f"p{x}{y}": int(sum(a == bool(x) and b == bool(y) for a,b in outcomes)) for x in (0,1) for y in (0,1)}}
        if psr and len(psr) != 4: raise ValueError("PSR requires all four frozen OOD families")
        result["PSR"] = psr; result["MacroPSR"] = float(np.mean([psr[x]["PSR"] for x in psr])) if psr else None
        result["secondary_metrics"] = secondary_attribute_metrics(rows)
        if len(self.plan) == 1260:
            result["bootstrap"] = bootstrap_summary(prompt)
            result["matched_control_diagnostics"] = matched_control_diagnostics(rows)
        state["metrics"] = "complete"; state["phase"] = 4; self._finish_timed_phase(state, "metrics", started); self._write(state); atomic_json(self.root / "metrics.json", result); return result

    def finalize(self, environment: dict[str, Any]) -> dict[str, Any]:
        state = self.state()
        if state["metrics"] != "complete" or state["integrity"] != "complete" or state["predictions"] != "complete": raise RuntimeError("all phases through metrics are required before summary")
        if sha256_file(self.manifest_path) != state.get("generation_manifest_sha256") or sha256_file(self.predictions_path) != state.get("prediction_manifest_sha256"): raise ValueError("evidence manifest identity changed")
        target = self.root / "final_conditioning_summary.json"
        if target.exists(): raise FileExistsError("final summary is immutable")
        metrics = json.loads((self.root / "metrics.json").read_text(encoding="utf-8"))
        counts = validate_parameter_counts(state.get("parameter_counts"))
        secondary = metrics.get("secondary_metrics", {})
        if set(secondary) != set(ATTRIBUTES) or any(not {"label_order", "confusion_matrix", "macro_f1", "balanced_accuracy"} <= set(secondary[attribute]) for attribute in ATTRIBUTES): raise ValueError("secondary metrics are incomplete")
        state["timings_seconds"]["total"] = self.clock() - state["started_perf_counter"]
        timings = validate_timings(state["timings_seconds"])
        payload = {"schema": "dr04_final_conditioning_summary", "version": 3, "FINAL_SCIENTIFIC_EVALUATION": True, "run_identity": {"plan_sha256": _json_sha(self.plan), **self.identities}, "environment": environment, "parameter_counts": counts, "timings_seconds": {"kind": "operational_wall_time_perf_counter", **timings}, "generation_plan": {"prompt_count": len({r["prompt_id"] for r in self.plan}), "slot_count": len(self.plan), "id_slots": sum(r["panel_split"] == "ID" for r in self.plan), "ood_slots": sum(r["panel_split"] == "OOD" for r in self.plan), "seeds": list(SEEDS), "sha256": _json_sha(self.plan)}, "generation": {"completed": len(self.rows()), "errors": state["slots"], "retries": state["retries"]}, "integrity": {"status": state["integrity"], "checked_pngs": len(self.rows())}, "evaluator_inference": {"status": state["predictions"], "expected_predictions": len(self.plan), "observed_predictions": len(load_jsonl(self.predictions_path))}, "primary_conditioning": {"ID_MAA": metrics["ID_MAA"], "OOD_MAA": metrics["OOD_MAA"], "per_prompt": metrics["prompt_level"]}, "primary_compositional": {"PSR": metrics["PSR"], "MacroPSR": metrics["MacroPSR"]}, "secondary_metrics": metrics["secondary_metrics"], "secondary": {"matched_control_diagnostics": metrics.get("matched_control_diagnostics", "not_applicable_synthetic")}, "bootstrap": metrics.get("bootstrap", "not_applicable_synthetic"), "manifest_paths_hashes": {"generation": {"path": str(self.manifest_path), "sha256": state["generation_manifest_sha256"]}, "predictions": {"path": str(self.predictions_path), "sha256": state["prediction_manifest_sha256"]}}, "forbidden_work_audit": {"unconditional_generations": 0, "KID": 0, "FID": 0, "LPIPS": 0, "L1": 0, "minimal_edit": 0, "EXT": 0}, "phase_completion": 5}
        atomic_json(target, payload); state["summary"] = "complete"; state["phase"] = 5; self._write(state); return payload
