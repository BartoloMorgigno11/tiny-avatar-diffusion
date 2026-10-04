"""Hash-bound, resumable infrastructure for the DR05 final-quality run.

This module deliberately knows nothing about project image archives: callers inject
reference decoding, sampling, and metric functions.  That makes its state machine
testable without opening sealed data and prevents a final plan from being rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


PLAN_FIELDS = ("planned_sample_index", "split", "reference_sample_id", "prompt", "generation_namespace", "batch_index", "batch_seed", "sample_index_in_batch")
MODELS = ("conditional", "unconditional")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(pixels: np.ndarray) -> str:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.shape != (32, 32, 3):
        raise ValueError("DR05 canonical pixels must be RGB uint8 32x32")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Durably replace JSON using a unique sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(rows, key=lambda item: int(item.get("planned_sample_index", -1))):
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_png(pixels: np.ndarray, path: Path) -> dict[str, str]:
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.shape != (32, 32, 3):
        raise ValueError("canonical output must be RGB uint8 32x32")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
        with Image.open(temporary) as image:
            loaded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(pixels, loaded):
            raise RuntimeError("PNG round-trip mismatch")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"png_sha256": sha256_file(path), "pixel_sha256": pixel_sha256(pixels)}


def directory_identity(directory: Path) -> str:
    entries = [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(directory.glob("*.png"))]
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentity:
    plan_path: str
    plan_sha256: str
    plan_meta_path: str
    plan_meta_sha256: str


class FinalQualityExecutor:
    """A callback-driven final executor; real project access is opt-in at call time."""
    def __init__(self, *, root: Path, plan: list[dict[str, Any]], identity: ExecutionIdentity,
                 references: dict[str, list[dict[str, Any]]], reference_loader: Callable[[dict[str, Any]], np.ndarray],
                 generator: Callable[[str, list[dict[str, Any]], int], list[np.ndarray]],
                 metric: Callable[[str, str, Path, Path], dict[str, float]], checkpoint_sha: dict[str, str],
                 checkpoint_path: dict[str, str], expected_counts: dict[str, int] | None = None) -> None:
        self.root, self.plan, self.identity = root, plan, identity
        self.references, self.reference_loader, self.generator, self.metric = references, reference_loader, generator, metric
        self.checkpoint_sha, self.checkpoint_path = checkpoint_sha, checkpoint_path
        self.expected_counts = expected_counts or {"real_id": 466, "real_ood": 342, "conditional_id": 5767, "conditional_ood": 4233, "unconditional_id": 5767, "unconditional_ood": 4233}
        self.state_path = root / "phase_status" / "execution_state.json"
        self.progress_path = root / "phase_status" / "progress.json"
        self._validate_plan()

    def _validate_plan(self) -> None:
        if not self.plan or [row.get("planned_sample_index") for row in self.plan] != list(range(len(self.plan))):
            raise ValueError("plan must be contiguous and loaded from the frozen artifact")
        if any(any(key not in row for key in PLAN_FIELDS) for row in self.plan):
            raise ValueError("plan schema mismatch")
        if any(row["batch_index"] != row["planned_sample_index"] // 16 or row["batch_seed"] != 502026 + row["batch_index"] or row["sample_index_in_batch"] != row["planned_sample_index"] % 16 for row in self.plan):
            raise ValueError("plan batch schedule mismatch")
        if [row["split"] for row in self.plan] != sorted((row["split"] for row in self.plan), key=lambda value: 0 if value == "ID" else 1):
            raise ValueError("plan must place ID before OOD")

    def _initial_state(self) -> dict[str, Any]:
        return {"schema": "dr05_final_execution_state", "version": 1, "identity": self.identity.__dict__, "phase": "phase1",
                "references": {"ID": [], "OOD": []}, "models": {model: {"completed_batches": [], "retries": {}, "errors": {}} for model in MODELS},
                "phase2": {"status": "pending"}, "phase3": {"metrics": {}}, "phase4": {"status": "pending"}, "started": time.time()}

    def state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            if (self.root / "final_quality_summary.json").exists():
                raise FileExistsError("final summary exists without resumable state; refusing ambiguous duplicate run")
            state = self._initial_state(); atomic_json(self.state_path, state); return state
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("identity") != self.identity.__dict__:
            raise ValueError("existing run has different frozen plan identity")
        if (self.root / "final_quality_summary.json").exists() and state.get("phase4", {}).get("status") == "complete":
            raise FileExistsError("complete final scientific run is immutable")
        return state

    def _write_state(self, state: dict[str, Any], *, phase: str, model: str | None = None) -> None:
        state["phase"] = phase; state["last_update"] = time.time(); atomic_json(self.state_path, state)
        completed = len(state["models"][model]["completed_batches"]) if model else 0
        atomic_json(self.progress_path, {"phase": phase, "model": model, "completed_batches": completed, "total_batches": len({r["batch_index"] for r in self.plan}), "last_update": state["last_update"]})

    def _manifest(self, model: str, split: str) -> Path:
        return self.root / "manifests" / f"{model}_{split.lower()}.jsonl"

    def _rows(self, model: str, split: str) -> list[dict[str, Any]]:
        path = self._manifest(model, split)
        return load_jsonl(path) if path.exists() else []

    def canonicalize_references(self) -> None:
        state = self.state()
        for split, records in self.references.items():
            done = set(state["references"][split]); rows = self._rows("real", split)
            for record in records:
                identifier = str(record["reference_sample_id"])
                if identifier in done:
                    continue
                pixels = self.reference_loader(record)
                relative = Path("real") / split.lower() / f"{identifier}.png"
                hashes = save_png(pixels, self.root / relative)
                rows.append({"reference_sample_id": identifier, "source_split": split, "source_image_reference": record.get("source_image_reference", identifier), "canonical_shape": [32, 32, 3], "mode": "RGB", "dtype": "uint8", "canonical_png_relative_path": relative.as_posix(), **hashes, "status": "complete"})
                atomic_jsonl(self._manifest("real", split), rows)
                state["references"][split].append(identifier); self._write_state(state, phase="phase1", model=None)

    def _remove_incomplete_batch(self, model: str, batch: int) -> None:
        for split in ("ID", "OOD"):
            retained = [row for row in self._rows(model, split) if row["batch_index"] != batch]
            for row in self._rows(model, split):
                if row["batch_index"] == batch:
                    (self.root / row["png_relative_path"]).unlink(missing_ok=True)
            atomic_jsonl(self._manifest(model, split), retained)

    def generate_model(self, model: str) -> None:
        if model not in MODELS: raise ValueError("unknown model")
        state = self.state(); completed = set(state["models"][model]["completed_batches"])
        for batch in sorted({row["batch_index"] for row in self.plan}):
            if batch in completed: continue
            batch_rows = [row for row in self.plan if row["batch_index"] == batch]
            self._remove_incomplete_batch(model, batch)
            try:
                images = self.generator(model, batch_rows, batch_rows[0]["batch_seed"])
                if len(images) != len(batch_rows): raise ValueError("generator did not return exact batch size")
                writes = {split: self._rows(model, split) for split in ("ID", "OOD")}
                for item, pixels in zip(batch_rows, images):
                    split = item["split"]; relative = Path(model) / split.lower() / f"{model}_{item['planned_sample_index']:05d}.png"
                    hashes = save_png(pixels, self.root / relative)
                    writes[split].append({"sample_id": f"{model}_{item['planned_sample_index']:05d}", "planned_sample_index": item["planned_sample_index"], "model": model, "checkpoint_path": self.checkpoint_path[model], "checkpoint_sha256": self.checkpoint_sha[model], "evaluation_role": "main_quality", "split": split, "reference_sample_id": item["reference_sample_id"], "prompt": item["prompt"], "conditioning_consumed": model == "conditional", "generation_namespace": item["generation_namespace"], "batch_index": batch, "batch_seed": item["batch_seed"], "sample_index_in_batch": item["sample_index_in_batch"], "image_shape": [32,32,3], "image_mode": "RGB", "dtype": "uint8", "png_relative_path": relative.as_posix(), **hashes, "status": "complete"})
                for split in ("ID", "OOD"): atomic_jsonl(self._manifest(model, split), writes[split])
                state["models"][model]["completed_batches"].append(batch); state["models"][model]["completed_batches"].sort(); self._write_state(state, phase="phase1", model=model)
            except Exception as error:
                state["models"][model]["retries"][str(batch)] = state["models"][model]["retries"].get(str(batch), 0) + 1
                state["models"][model]["errors"][str(batch)] = str(error); self._write_state(state, phase="phase1", model=model); raise

    def audit(self) -> dict[str, str]:
        state = self.state(); expected_plan = {row["planned_sample_index"]: row for row in self.plan}
        inventories: dict[str, str] = {}
        for model, split, expected in (("real", "ID", self.expected_counts["real_id"]), ("real", "OOD", self.expected_counts["real_ood"]), *((m, s, self.expected_counts[f"{m}_{s.lower()}"]) for m in MODELS for s in ("ID", "OOD"))):
            rows = self._rows(model, split)
            if len(rows) != expected: raise ValueError(f"{model} {split} count mismatch")
            items = []
            for row in rows:
                relative = row.get("canonical_png_relative_path", row.get("png_relative_path")); path = self.root / relative
                if not path.is_file() or path.stat().st_size == 0 or row.get("status") != "complete" or sha256_file(path) != row["png_sha256"]: raise ValueError("PNG integrity mismatch")
                with Image.open(path) as image: pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if pixels.shape != (32,32,3) or pixel_sha256(pixels) != row["pixel_sha256"]: raise ValueError("pixel integrity mismatch")
                if model != "real":
                    plan = expected_plan.get(row["planned_sample_index"])
                    if not plan or any(row[key] != plan[key] for key in PLAN_FIELDS) or row["conditioning_consumed"] != (model == "conditional"): raise ValueError("plan-to-manifest mismatch")
                items.append({"path": relative, "png_sha256": row["png_sha256"], "pixel_sha256": row["pixel_sha256"]})
            inventories[f"{model}_{split.lower()}"] = hashlib.sha256(json.dumps(sorted(items, key=lambda i: i["path"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        state["phase2"] = {"status": "complete", "inventories": inventories}; self._write_state(state, phase="phase2"); return inventories

    def _union(self, model: str) -> Path:
        union = self.root / "metrics" / "unions" / model; union.mkdir(parents=True, exist_ok=True)
        for split in ("ID", "OOD"):
            for row in self._rows(model, split):
                relative = row.get("canonical_png_relative_path", row.get("png_relative_path")); source = self.root / relative; target = union / f"{split.lower()}_{Path(relative).name}"
                if not target.exists(): shutil.copy2(source, target)
        expected = self.expected_counts["real_id"] + self.expected_counts["real_ood"] if model == "real" else self.expected_counts[f"{model}_id"] + self.expected_counts[f"{model}_ood"]
        if len(list(union.glob("*.png"))) != expected: raise ValueError("global union membership mismatch")
        return union

    def metrics(self) -> dict[str, Any]:
        state = self.state()
        if state["phase2"].get("status") != "complete": raise RuntimeError("phase2 integrity must pass before metrics")
        results = state["phase3"]["metrics"]
        real = {"ID": self.root / "real" / "id", "OOD": self.root / "real" / "ood", "GLOBAL": self._union("real")}
        for model in MODELS:
            generated = {"ID": self.root / model / "id", "OOD": self.root / model / "ood", "GLOBAL": self._union(model)}
            for split in ("ID", "OOD", "GLOBAL"):
                key = f"{model}:{split}"; identity = {"real": state["phase2"]["inventories"][f"real_{split.lower()}"] if split != "GLOBAL" else directory_identity(real[split]), "generated": state["phase2"]["inventories"][f"{model}_{split.lower()}"] if split != "GLOBAL" else directory_identity(generated[split])}
                if key in results:
                    if results[key]["pool_identity"] != identity: raise ValueError("cached metric pool identity changed")
                    continue
                values = self.metric(model, split, real[split], generated[split])
                if not all(np.isfinite(value) for value in values.values()): raise ValueError("non-finite metric")
                results[key] = {"pool_identity": identity, "result": values}; self._write_state(state, phase="phase3")
        state["phase3"]["status"] = "complete"; self._write_state(state, phase="phase3"); return results

    def summary(self, provenance: dict[str, Any]) -> dict[str, Any]:
        state = self.state()
        if state["phase2"].get("status") != "complete" or state["phase3"].get("status") != "complete": raise RuntimeError("phases 2 and 3 must complete before summary")
        state["phase4"] = {"status": "complete"}
        payload = {"schema": "dr05_final_quality_main", "version": 1, "FINAL_SCIENTIFIC_EVALUATION": True, "identity": self.identity.__dict__, "state": state, **provenance}
        atomic_json(self.root / "final_quality_summary.json", payload); self._write_state(state, phase="phase4"); return payload
