"""Fail-closed R13 CA16 prospective-quality orchestration.

R13 deliberately consumes only the frozen ID_VAL/COMP_VAL quality population.
It has no evaluator, selection, or held-out-data dependency.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_evaluation import (
    CLEAN_FID_VERSION,
    load_ema_checkpoint,
    sha256_file,
)
from tiny_avatar_diffusion.ca16_experiment import ROOT
from tiny_avatar_diffusion.ca16_r11_orchestration import tensor_to_pixels
from tiny_avatar_diffusion.ca16_r12_orchestration import (
    resolve_run_spec,
    resolve_selected_checkpoints,
)
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.dr05_final_executor import pixel_sha256
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

PROTOCOL_ID = "ca16_r13_quality_guardrail_batched_v1"
PROTOCOL_PATH = ROOT / "data/processed/ca16_r13_quality_v1/r13_execution_protocol.json"
PROTOCOL_SHA256 = "fe292f1a67b4f755bfae0512601b206f3abd7d4ec4a2cae4bc820098631f240b"
QUALITY_MANIFEST = ROOT / "data/processed/ca16_eval_v1/quality_guardrail_manifest.json"
LOCK_NAME = ".r13.lock"
_REPLACE_DELAYS = (0.05, 0.15, 0.45)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL object required: {path}")
    return rows


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required R13 input is absent: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"R13 frozen input hash mismatch for {_relative(path)}")
    return actual


def _is_retryable_replace_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) and getattr(error, "winerror", None) == 5


def _replace_with_retry(
    temporary: Path,
    target: Path,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    for attempt, delay in enumerate(_REPLACE_DELAYS, start=1):
        try:
            replace(temporary, target)
            return
        except OSError as error:
            if not _is_retryable_replace_error(error) or attempt == len(
                _REPLACE_DELAYS
            ):
                raise
            time.sleep(delay)


def atomic_json(
    path: Path,
    payload: Any,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path, replace=replace)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(
                rows, key=lambda value: int(value["planned_sample_index"])
            ):
                stream.write(
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path, replace=replace)
    finally:
        temporary.unlink(missing_ok=True)


def _save_png(path: Path, pixels: np.ndarray) -> dict[str, str]:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.shape != (32, 32, 3):
        raise ValueError("R13 output must be RGB uint8 32x32")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
        with Image.open(temporary) as image:
            loaded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(array, loaded):
            raise RuntimeError("R13 PNG round-trip mismatch")
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"png_sha256": sha256_file(path), "pixel_sha256": pixel_sha256(array)}


def load_protocol() -> dict[str, Any]:
    if not PROTOCOL_SHA256:
        raise RuntimeError("R13 protocol hash has not been bound at freeze review")
    _require_hash(PROTOCOL_PATH, PROTOCOL_SHA256)
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("schema") != "tiny_avatar_diffusion.ca16_r13_quality_protocol_v1":
        raise ValueError("unexpected R13 protocol schema")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected R13 protocol identifier")
    return protocol


def _git_state() -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return (
        run("git", "rev-parse", "HEAD"),
        run("git", "branch", "--show-current"),
        not bool(run("git", "status", "--porcelain", "--untracked-files=normal")),
    )


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    quality = protocol.get("quality_manifest", {})
    if quality.get("geometry") != {"ID_VAL": 466, "COMP_VAL": 338, "GLOBAL": 804}:
        raise ValueError("R13 quality geometry changed")
    if quality.get("generation_seed") != "2026+canonical_manifest_index":
        raise ValueError("R13 canonical generation-seed policy changed")
    sampling = protocol.get("sampling", {})
    if (
        sampling.get("weight_variant") != "EMA"
        or sampling.get("precision") != "FP32"
        or sampling.get("sampler") != "ancestral DDPM"
        or sampling.get("timesteps") != 1000
        or sampling.get("sampling_batch_size") != 8
    ):
        raise ValueError("R13 sampling contract changed")
    metric = protocol.get("quality_metrics", {})
    if metric.get("primary") != ["KID_ID", "KID_COMP", "KID_GLOBAL"]:
        raise ValueError("R13 primary quality metrics changed")
    implementation = metric.get("implementation", {})
    if (
        implementation.get("version") != CLEAN_FID_VERSION
        or implementation.get("mode") != "clean"
        or implementation.get("feature_extractor") != "standard Inception-v3"
    ):
        raise ValueError("R13 Clean-FID contract changed")


def validate_preflight(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str | None = None
) -> dict[str, Any]:
    protocol = load_protocol()
    _validate_protocol(protocol)
    if importlib.metadata.version("clean-fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")
    head, branch, clean = _git_state()
    if head != expected_eval_sha or branch != "exp/ca16-ablation" or not clean:
        raise ValueError("R13 requires the clean expected exp/ca16-ablation identity")
    if expected_tag is not None:
        tagged = subprocess.run(
            ("git", "rev-parse", f"{expected_tag}^{{commit}}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tagged != head:
            raise ValueError("R13 expected tag does not resolve to execution identity")
    quality = protocol["quality_manifest"]
    bindings = protocol["r11_bindings"]
    hashes = {
        _relative(PROTOCOL_PATH): PROTOCOL_SHA256,
        _relative(QUALITY_MANIFEST): _require_hash(QUALITY_MANIFEST, quality["sha256"]),
    }
    for role, binding in protocol["token_sources"].items():
        path = ROOT / str(binding["path"])
        hashes[_relative(path)] = _require_hash(path, str(binding["sha256"]))
        if role not in {"ID_VAL", "COMP_VAL"}:
            raise ValueError("R13 has an unauthorized token source")
    for name in ("complete_path", "inventory_path"):
        path = ROOT / str(bindings[name])
        hashes[_relative(path)] = _require_hash(
            path, str(bindings[name.replace("_path", "_sha256")])
        )
    selected = resolve_selected_checkpoints()
    hashes.update(
        {str(row["selection_path"]): str(row["selection_sha256"]) for row in selected}
    )
    return {
        "status": "preflight_pass",
        "protocol_status": protocol.get("status"),
        "output_root": str(output_root),
        "output_exists": output_root.exists(),
        "hashes": hashes,
        "selected_checkpoints": list(selected),
        "geometry": {"models": 6, "slots_per_model": 804, "total_slots": 4824},
    }


def _token_index(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for role, binding in protocol["token_sources"].items():
        for source in _read_jsonl(ROOT / str(binding["path"])):
            key = "|".join(
                str(source.get(field))
                for field in ("dataset_variant", "split_role", "image_id")
            )
            if source.get("split_role") != role or key in indexed:
                raise ValueError(
                    "R13 token source has an invalid or duplicate stable key"
                )
            indexed[key] = source
    return indexed


def build_execution_plan(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = _read_json(QUALITY_MANIFEST)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise TypeError("R13 quality manifest entries are absent")
    token_index = _token_index(protocol)
    plan: list[dict[str, Any]] = []
    for index, record in enumerate(entries):
        if not isinstance(record, dict):
            raise TypeError("R13 quality manifest record is invalid")
        role = record.get("split_role")
        key = record.get("stable_record_key")
        if role not in {"ID_VAL", "COMP_VAL"} or not isinstance(key, str):
            raise ValueError("R13 contains a forbidden quality role")
        if (
            record.get("canonical_manifest_index") != index
            or record.get("generation_seed") != 2026 + index
        ):
            raise ValueError("R13 quality manifest has noncanonical index or seed")
        source = token_index.get(key)
        if source is None:
            raise ValueError("R13 quality record has no canonical token source")
        for field in ("dataset_variant", "split_role", "image_id", "caption"):
            if source.get(field) != record.get(field):
                raise ValueError(
                    f"R13 token source {field} differs from quality record"
                )
        if not isinstance(source.get("token_ids"), list) or not isinstance(
            source.get("attention_mask"), list
        ):
            raise TypeError("R13 token source lacks canonical conditioning")
        plan.append(
            {
                **record,
                "planned_sample_index": index,
                "sample_id": f"quality_{index:04d}",
                "token_ids": source["token_ids"],
                "attention_mask": source["attention_mask"],
            }
        )
    counts = {
        role: sum(row["split_role"] == role for row in plan)
        for role in ("ID_VAL", "COMP_VAL")
    }
    if len(plan) != 804 or counts != {"ID_VAL": 466, "COMP_VAL": 338}:
        raise ValueError("R13 quality-plan geometry is invalid")
    if len({row["stable_record_key"] for row in plan}) != len(plan):
        raise ValueError("R13 quality plan has duplicate stable keys")
    if [(row["split_role"], row["stable_record_key"]) for row in plan] != sorted(
        (row["split_role"], row["stable_record_key"]) for row in plan
    ):
        raise ValueError("R13 quality plan order is not canonical")
    return plan


class BatchedQualityGenerator:
    """Fixed-B8 sampler preserving original lanes across partial resume."""

    def __init__(
        self,
        plan: Sequence[Mapping[str, Any]],
        encoder: TextConditioningEncoder,
        denoiser: ConditionalTinyUNet,
        device: torch.device,
    ) -> None:
        self.plan = tuple(plan)
        self.encoder, self.denoiser, self.device = encoder, denoiser, device
        self.schedule = LinearDDPMSchedule(num_timesteps=1000).to(device)
        self.cache: dict[str, np.ndarray] = {}
        self.indices = {
            str(row["sample_id"]): index for index, row in enumerate(self.plan)
        }

    def _lanes(self, start: int) -> tuple[Mapping[str, Any], ...]:
        real = self.plan[start : start + 8]
        if len(real) == 8:
            return real
        if len(real) == 4:
            return real + real
        raise ValueError("R13 permits only the frozen four-lane padded tail")

    def _generate_group(self, start: int) -> None:
        real = self.plan[start : start + 8]
        lanes = self._lanes(start)
        ids = [row["token_ids"] for row in lanes]
        masks = [row["attention_mask"] for row in lanes]
        if len({len(row) for row in ids}) != 1 or any(
            len(a) != len(b) for a, b in zip(ids, masks, strict=True)
        ):
            raise ValueError("R13 frozen token rows are not B8-compatible")
        generators = [
            torch.Generator(device=self.device).manual_seed(int(row["generation_seed"]))
            for row in lanes
        ]

        def noise() -> torch.Tensor:
            return torch.cat(
                [
                    torch.randn(
                        (1, 3, 32, 32),
                        device=self.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    for generator in generators
                ],
                dim=0,
            )

        with torch.no_grad():
            samples = conditional_sample(
                self.schedule,
                self.encoder,
                self.denoiser,
                torch.tensor(ids, device=self.device),
                torch.tensor(masks, device=self.device),
                x_T=noise(),
                reverse_noise_provider=lambda _t, _x: noise(),
                clip_denoised=True,
            )
        if len(samples) != 8:
            raise ValueError("R13 sampler returned a non-B8 result")
        for row, sample in zip(real, samples[: len(real)], strict=True):
            self.cache[str(row["sample_id"])] = (
                ((sample.detach().cpu().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )

    def __call__(self, row: Mapping[str, Any]) -> np.ndarray:
        sample_id = str(row["sample_id"])
        if sample_id not in self.cache:
            index = self.indices.get(sample_id)
            if index is None or self.plan[index] != row:
                raise ValueError("R13 sample is outside the canonical plan")
            self._generate_group(index - index % 8)
        return self.cache.pop(sample_id)


class R13Executor:
    def __init__(
        self,
        root: Path,
        plan: Sequence[Mapping[str, Any]],
        identities: Mapping[str, Any],
    ) -> None:
        self.root, self.plan, self.identities = root, tuple(plan), dict(identities)
        self.state_path, self.manifest_path = (
            root / "r13_state.json",
            root / "generation_manifest.jsonl",
        )

    def _plan_sha(self) -> str:
        public = [
            {
                key: value
                for key, value in row.items()
                if key not in {"token_ids", "attention_mask"}
            }
            for row in self.plan
        ]
        return hashlib.sha256(
            json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = _read_json(self.state_path)
            if (
                state.get("schema") != "tiny_avatar_diffusion.ca16_r13_quality_state_v1"
                or state.get("identities") != self.identities
                or state.get("plan_sha256") != self._plan_sha()
            ):
                raise ValueError(
                    "R13 existing state has a different execution identity"
                )
            return state
        if (self.root / "r13_summary.json").exists():
            raise FileExistsError("R13 summary exists without resumable state")
        state = {
            "schema": "tiny_avatar_diffusion.ca16_r13_quality_state_v1",
            "plan_sha256": self._plan_sha(),
            "identities": self.identities,
            "phase": 0,
            "generation": "pending",
            "integrity": "pending",
            "metrics": "pending",
            "summary": "pending",
        }
        atomic_json(self.state_path, state)
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        atomic_json(self.state_path, state)

    def _expected(self, row: Mapping[str, Any]) -> dict[str, Any]:
        relative = (
            Path("images")
            / str(row["split_role"]).lower()
            / f"{int(row['planned_sample_index']):04d}.png"
        )
        return {
            "schema": "tiny_avatar_diffusion.ca16_r13_quality_sample_v1",
            "sample_id": row["sample_id"],
            "planned_sample_index": row["planned_sample_index"],
            "stable_record_key": row["stable_record_key"],
            "split_role": row["split_role"],
            "generation_seed": row["generation_seed"],
            "checkpoint_path": self.identities["checkpoint_path"],
            "checkpoint_sha256": self.identities["checkpoint_sha256"],
            "png_relative_path": relative.as_posix(),
            "status": "complete",
        }

    def _validate_row(
        self, row: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("R13 generation manifest identity mismatch")
        path = self.root / str(row["png_relative_path"])
        if not path.is_file() or sha256_file(path) != row.get("png_sha256"):
            raise ValueError("R13 generated PNG identity mismatch")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row.get(
            "pixel_sha256"
        ):
            raise ValueError("R13 generated pixels are invalid")

    def generate(self, generator: Callable[[Mapping[str, Any]], np.ndarray]) -> None:
        state = self.state()
        rows = _read_jsonl(self.manifest_path)
        by_id = {str(row.get("sample_id")): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("R13 generation manifest has duplicate sample IDs")
        for item in self.plan:
            expected = self._expected(item)
            existing = by_id.get(str(item["sample_id"]))
            path = self.root / str(expected["png_relative_path"])
            if existing is not None:
                self._validate_row(existing, expected)
                continue
            if path.exists():
                raise RuntimeError("R13 orphan PNG exists without provenance")
            rows.append({**expected, **_save_png(path, generator(item))})
            by_id[str(item["sample_id"])] = rows[-1]
            atomic_jsonl(self.manifest_path, rows)
        if len(rows) != len(self.plan):
            raise ValueError("R13 generation did not complete the canonical plan")
        state["generation"] = "complete"
        state["phase"] = 1
        state["generation_manifest_sha256"] = sha256_file(self.manifest_path)
        self._write(state)

    def audit(self) -> dict[str, str]:
        state = self.state()
        rows = _read_jsonl(self.manifest_path)
        if len(rows) != len(self.plan):
            raise ValueError("R13 integrity requires all generated records")
        for item in self.plan:
            row = next(
                (
                    candidate
                    for candidate in rows
                    if candidate.get("sample_id") == item["sample_id"]
                ),
                None,
            )
            if row is None:
                raise ValueError("R13 generation record is absent")
            self._validate_row(row, self._expected(item))
        inventory = hashlib.sha256(
            json.dumps(
                sorted(
                    [
                        {
                            key: row[key]
                            for key in (
                                "png_relative_path",
                                "png_sha256",
                                "pixel_sha256",
                            )
                        }
                        for row in rows
                    ],
                    key=lambda value: value["png_relative_path"],
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        state["integrity"] = "complete"
        state["phase"] = 2
        state["inventory_sha256"] = inventory
        self._write(state)
        return {"generated_inventory_sha256": inventory}

    def metric_rows(self, role: str) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.manifest_path)
        selected = [row for row in rows if row.get("split_role") == role]
        expected = {"ID_VAL": 466, "COMP_VAL": 338}[role]
        if len(selected) != expected:
            raise ValueError("R13 metric role membership mismatch")
        return selected

    def finalize(
        self, metrics: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> dict[str, Any]:
        state = self.state()
        if state.get("integrity") != "complete" or state.get("metrics") != "complete":
            raise RuntimeError("R13 summary requires completed integrity and metrics")
        target = self.root / "r13_summary.json"
        if target.exists():
            raise FileExistsError("R13 summary is immutable")
        payload = {
            "schema": "tiny_avatar_diffusion.ca16_r13_quality_summary_v1",
            "status": "COMPLETE",
            "run_identity": {"plan_sha256": self._plan_sha(), **self.identities},
            "generation": {
                "completed": len(self.plan),
                "manifest_sha256": sha256_file(self.manifest_path),
            },
            "integrity": {"inventory_sha256": state["inventory_sha256"]},
            "metrics": dict(metrics),
            "environment": dict(environment),
        }
        atomic_json(target, payload)
        state["summary"] = "complete"
        state["phase"] = 4
        self._write(state)
        return payload


def validate_completed_run(executor: R13Executor) -> dict[str, Any]:
    """Read-only integrity check before accepting an immutable completed R13 run."""
    summary_path = executor.root / "r13_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("R13 completion summary is absent")
    state = executor.state()
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != "tiny_avatar_diffusion.ca16_r13_quality_summary_v1"
        or summary.get("status") != "COMPLETE"
        or summary.get("run_identity")
        != {"plan_sha256": executor._plan_sha(), **executor.identities}
        or state.get("phase") != 4
        or state.get("summary") != "complete"
        or state.get("integrity") != "complete"
        or state.get("metrics") != "complete"
        or summary.get("generation", {}).get("completed") != len(executor.plan)
        or summary.get("generation", {}).get("manifest_sha256")
        != sha256_file(executor.manifest_path)
    ):
        raise ValueError("R13 completion identity or status is inconsistent")
    rows = _read_jsonl(executor.manifest_path)
    if len(rows) != len(executor.plan):
        raise ValueError("R13 completed generation evidence is incomplete")
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("R13 completed generation evidence has duplicate IDs")
    for item in executor.plan:
        row = by_id.get(str(item["sample_id"]))
        if row is None:
            raise ValueError("R13 completed generation record is absent")
        executor._validate_row(row, executor._expected(item))
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "KID_ID",
        "KID_COMP",
        "KID_GLOBAL",
    }:
        raise ValueError("R13 completed metric record is invalid")
    if _read_json(executor.root / "quality_metrics.json") != metrics:
        raise ValueError("R13 completed metric artifact differs from summary")
    return summary


def _load_generator(
    checkpoint: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> BatchedQualityGenerator:
    spec = resolve_run_spec(checkpoint)
    encoder, denoiser, artifact = load_ema_checkpoint(
        ROOT / str(checkpoint["checkpoint_path"]), spec, 20_000, device=device
    )
    if artifact.get("experiment", {}).get("run_id") != spec.run_id:
        raise ValueError("R13 checkpoint identity mismatch")
    return BatchedQualityGenerator(plan, encoder, denoiser, device)


def _metric(
    real: Path, generated: Path, *, device: torch.device, expected_count: int
) -> dict[str, Any]:
    if (
        len(list(real.glob("*.png"))) != expected_count
        or len(list(generated.glob("*.png"))) != expected_count
    ):
        raise ValueError("R13 KID pool count mismatch")
    if importlib.metadata.version("clean-fid") != CLEAN_FID_VERSION:
        raise RuntimeError(f"clean-fid must equal {CLEAN_FID_VERSION}")
    from cleanfid import fid

    model = fid.build_feature_extractor(
        mode="clean", device=device, use_dataparallel=False
    )
    real_features = fid.get_folder_features(
        str(real),
        model=model,
        num_workers=0,
        batch_size=8,
        device=device,
        mode="clean",
        verbose=False,
    )
    generated_features = fid.get_folder_features(
        str(generated),
        model=model,
        num_workers=0,
        batch_size=8,
        device=device,
        mode="clean",
        verbose=False,
    )
    caller_state = np.random.get_state()
    try:
        np.random.seed(2026)
        value = fid.kernel_distance(
            real_features,
            generated_features,
            num_subsets=100,
            max_subset_size=expected_count,
        )
    finally:
        np.random.set_state(caller_state)
    return {
        "status": "COMPUTED",
        "value": float(value),
        "clean_fid_version": CLEAN_FID_VERSION,
        "mode": "clean",
        "feature_extractor": "standard Inception-v3",
        "num_subsets": 100,
        "subset_size": expected_count,
        "subset_seed": 2026,
        "fid": {"status": "NOT_COMPUTED_OPTIONAL"},
    }


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError("R13 metric pool contains a conflicting existing file")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(source) != sha256_file(temporary):
            raise RuntimeError("R13 metric-pool copy checksum mismatch")
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _pool(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    directory = root / "metrics" / "pools" / name
    for row in rows:
        source = root / str(row["png_relative_path"])
        target = directory / f"{int(row['planned_sample_index']):04d}.png"
        _link_or_copy(source, target)
    if len(list(directory.glob("*.png"))) != len(rows):
        raise ValueError("R13 metric pool has an unexpected membership")
    return directory


def materialize_real_references(
    output_root: Path, plan: Sequence[Mapping[str, Any]]
) -> dict[str, Path]:
    """Materialize only the hash-bound ID_VAL/COMP_VAL real populations."""
    from tiny_avatar_diffusion.debug_tiny_overfit import preload_images

    manifest_path = output_root / "real_reference_manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("R13 real-reference manifest has duplicate IDs")
    for item in plan:
        sample_id = str(item["sample_id"])
        relative = (
            Path("real_references")
            / str(item["split_role"]).lower()
            / f"{int(item['planned_sample_index']):04d}.png"
        )
        expected = {
            "schema": "tiny_avatar_diffusion.ca16_r13_real_reference_v1",
            "sample_id": sample_id,
            "planned_sample_index": item["planned_sample_index"],
            "stable_record_key": item["stable_record_key"],
            "split_role": item["split_role"],
            "png_relative_path": relative.as_posix(),
            "status": "complete",
        }
        existing = by_id.get(sample_id)
        path = output_root / relative
        if existing is not None:
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ValueError("R13 real-reference identity mismatch")
            if not path.is_file() or sha256_file(path) != existing.get("png_sha256"):
                raise ValueError("R13 real-reference PNG mismatch")
            continue
        if path.exists():
            raise RuntimeError("R13 orphan real-reference PNG exists")
        pixels = tensor_to_pixels(
            preload_images([{"source_image_path": item["source_image_path"]}])[0]
        )
        rows.append({**expected, **_save_png(path, pixels)})
        by_id[sample_id] = rows[-1]
        atomic_jsonl(manifest_path, rows)
    if len(rows) != 804:
        raise ValueError("R13 real-reference geometry is incomplete")
    output: dict[str, Path] = {}
    for role, expected_count in (("ID_VAL", 466), ("COMP_VAL", 338)):
        selected = [row for row in rows if row["split_role"] == role]
        if len(selected) != expected_count:
            raise ValueError("R13 real-reference role count mismatch")
        output[role] = _pool(output_root, f"real_{role.lower()}", selected)
    output["GLOBAL"] = _pool(output_root, "real_global", rows)
    return output


def _run_metrics(
    executor: R13Executor, real: Mapping[str, Path], device: torch.device
) -> dict[str, Any]:
    state = executor.state()
    if state.get("metrics") == "complete":
        return _read_json(executor.root / "quality_metrics.json")
    rows = {role: executor.metric_rows(role) for role in ("ID_VAL", "COMP_VAL")}
    rows["GLOBAL"] = rows["ID_VAL"] + rows["COMP_VAL"]
    metrics = {
        "KID_ID": _metric(
            real["ID_VAL"],
            _pool(executor.root, "generated_id_val", rows["ID_VAL"]),
            device=device,
            expected_count=466,
        ),
        "KID_COMP": _metric(
            real["COMP_VAL"],
            _pool(executor.root, "generated_comp_val", rows["COMP_VAL"]),
            device=device,
            expected_count=338,
        ),
        "KID_GLOBAL": _metric(
            real["GLOBAL"],
            _pool(executor.root, "generated_global", rows["GLOBAL"]),
            device=device,
            expected_count=804,
        ),
    }
    atomic_json(executor.root / "quality_metrics.json", metrics)
    state["metrics"] = "complete"
    state["phase"] = 3
    executor._write(state)
    return metrics


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("R13 output lock exists; inspect before recovery") from error
    try:
        os.write(
            descriptor,
            json.dumps({"pid": os.getpid(), "protocol_id": PROTOCOL_ID}).encode(),
        )
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def execute_r13(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str | None = None
) -> dict[str, Any]:
    if expected_tag is None:
        raise ValueError("scientific R13 execution requires an expected annotated tag")
    preflight = validate_preflight(
        output_root=output_root,
        expected_eval_sha=expected_eval_sha,
        expected_tag=expected_tag,
    )
    if preflight["protocol_status"] != "FROZEN":
        raise RuntimeError("R13 execution requires a frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("scientific R13 execution requires CUDA")
    plan = build_execution_plan(load_protocol())
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    with exclusive_run_lock(output_root):
        real = materialize_real_references(output_root, plan)
        for checkpoint in preflight["selected_checkpoints"]:
            root = output_root / str(checkpoint["run_id"])
            identities = {
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "quality_manifest_sha256": preflight["hashes"][
                    _relative(QUALITY_MANIFEST)
                ],
                **checkpoint,
                "sampling_batch_size": 8,
            }
            executor = R13Executor(root, plan, identities)
            if (root / "r13_summary.json").is_file():
                results.append(validate_completed_run(executor))
                continue
            if executor.state().get("generation") != "complete":
                executor.generate(_load_generator(checkpoint, plan, device))
            inventory = executor.audit()
            metrics = _run_metrics(executor, real, device)
            summary = executor.finalize(
                metrics,
                {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": str(device),
                },
            )
            results.append(
                {"run_id": checkpoint["run_id"], **inventory, "summary": summary}
            )
    return {
        "schema": "tiny_avatar_diffusion.ca16_r13_quality_execution_result_v1",
        "runs": results,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
        },
    }
