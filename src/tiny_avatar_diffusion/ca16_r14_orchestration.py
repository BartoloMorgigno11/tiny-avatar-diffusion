"""R14 prospective diversity and efficiency orchestration.

The module is intentionally inert until its protocol is frozen.  Preflight
only hashes and validates inputs; it cannot sample, invoke LPIPS, or benchmark
CUDA.  Runtime functions are kept here so the frozen implementation and the
execution contract are one small, auditable unit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tiny_avatar_diffusion.ca16_r12_orchestration import (
    _load_generator,
    resolve_selected_checkpoints,
)
from tiny_avatar_diffusion.ca16_r13_quality import (
    _save_png,
    atomic_json,
    atomic_jsonl,
)
from tiny_avatar_diffusion.dr05_diversity import (
    lpips_alexnet_metric,
    measure_pairs,
)
from tiny_avatar_diffusion.dr05_final_executor import pixel_sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "data/processed/ca16_r14_diversity_efficiency_v1/r14_execution_protocol.json"
)
PROTOCOL_ID = "ca16_r14_diversity_efficiency_b8_v1"
# Bound only during the authorized freeze transition.
PROTOCOL_SHA256 = "6c075702c53d2290d384c2c51a8647fc1be191b43010f3783665b076bb4ad4a6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _git_state() -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip()

    return (
        run("git", "rev-parse", "HEAD"),
        run("git", "branch", "--show-current"),
        not bool(run("git", "status", "--porcelain")),
    )


def load_protocol(*, require_bound_hash: bool = False) -> dict[str, Any]:
    actual = sha256_file(PROTOCOL_PATH)
    if require_bound_hash and (not PROTOCOL_SHA256 or actual != PROTOCOL_SHA256):
        raise ValueError("R14 protocol hash is not frozen")
    protocol = _read_json(PROTOCOL_PATH)
    if (
        protocol.get("schema")
        != "tiny_avatar_diffusion.ca16_r14_diversity_efficiency_protocol_v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("unexpected R14 protocol identity")
    if protocol.get("status") not in {"PRE_FREEZE", "FROZEN"}:
        raise ValueError("invalid R14 protocol status")
    return protocol


def _require_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def _token_index(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for role, binding in protocol["token_sources"].items():
        path = ROOT / str(binding["path"])
        _require_hash(path, str(binding["sha256"]))
        for row in _read_jsonl(path):
            key = "|".join(
                str(row.get(field))
                for field in ("dataset_variant", "split_role", "image_id")
            )
            if row.get("split_role") != role or key in indexed:
                raise ValueError("R14 token source identity is invalid")
            indexed[key] = row
    return indexed


def build_execution_plan(
    protocol: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    protocol = protocol or load_protocol()
    binding = protocol["diversity_manifest"]
    manifest_path = ROOT / str(binding["path"])
    _require_hash(manifest_path, str(binding["sha256"]))
    manifest = _read_json(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise TypeError("R14 diversity manifest entries are invalid")
    tokens = _token_index(protocol)
    seeds = tuple(protocol["diversity_manifest"]["generation_seeds"])
    plan: list[dict[str, Any]] = []
    for entry_index, item in enumerate(entries):
        role, key = item.get("split_role"), item.get("stable_record_key")
        if role not in {"ID_VAL", "COMP_VAL"} or not isinstance(key, str):
            raise ValueError("R14 contains a forbidden split")
        token = tokens.get(key)
        if token is None or any(
            token.get(field) != item.get(field)
            for field in ("dataset_variant", "split_role", "image_id", "caption")
        ):
            raise ValueError("R14 canonical token join mismatch")
        if not isinstance(token.get("token_ids"), list) or not isinstance(
            token.get("attention_mask"), list
        ):
            raise TypeError("R14 canonical token source is incomplete")
        for seed_index, seed in enumerate(seeds):
            sample_index = len(plan)
            plan.append(
                {
                    **item,
                    "planned_sample_index": sample_index,
                    "sample_id": f"r14_{entry_index:03d}_{seed}",
                    "generation_seed": seed,
                    "token_ids": token["token_ids"],
                    "attention_mask": token["attention_mask"],
                    "panel_split": "ID" if role == "ID_VAL" else "COMP",
                }
            )
    validate_execution_plan(plan)
    return plan


def validate_execution_plan(plan: Sequence[Mapping[str, Any]]) -> None:
    if len(plan) != 1000 or len({row["sample_id"] for row in plan}) != 1000:
        raise ValueError("R14 requires exactly 1,000 samples per checkpoint")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan:
        groups[str(row["stable_record_key"])].append(row)
    if len(groups) != 200 or any(
        tuple(row["generation_seed"] for row in rows) != (2026, 2027, 2028, 2029, 2030)
        for rows in groups.values()
    ):
        raise ValueError("R14 prompt/seed geometry is invalid")
    counts = {
        role: sum(row["split_role"] == role for row in plan) // 5
        for role in ("ID_VAL", "COMP_VAL")
    }
    families = {
        family: sum(row.get("comp_family") == family for row in plan) // 5
        for family in ("CV1", "CV2", "CV3", "CV4")
    }
    if counts != {"ID_VAL": 100, "COMP_VAL": 100} or families != {
        family: 25 for family in families
    }:
        raise ValueError("R14 panel geometry is invalid")


def resolve_inputs(
    protocol: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    protocol = protocol or load_protocol()
    bindings = protocol["r11_bindings"]
    _require_hash(
        ROOT / str(bindings["complete_path"]), str(bindings["complete_sha256"])
    )
    _require_hash(
        ROOT / str(bindings["inventory_path"]), str(bindings["inventory_sha256"])
    )
    checkpoints = resolve_selected_checkpoints()
    if len(checkpoints) != 6 or any(
        row["milestone_step"] != bindings["selected_milestone"] for row in checkpoints
    ):
        raise ValueError("R14 must use exactly six R11 20k winners")
    return build_execution_plan(protocol), checkpoints


def validate_preflight(
    *,
    output_root: Path,
    expected_eval_sha: str,
    expected_tag: str | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol()
    head, branch, clean = _git_state()
    if (head, branch, clean) != (expected_eval_sha, "exp/ca16-ablation", True):
        raise ValueError("R14 requires the expected clean Git identity")
    if expected_tag:
        tagged = subprocess.run(
            ("git", "rev-parse", f"{expected_tag}^{{commit}}"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if tagged != head:
            raise ValueError("R14 tag does not resolve to expected identity")
    if output_root.exists() and not allow_existing:
        raise FileExistsError("R14 output root must be unused")
    plan, checkpoints = resolve_inputs(protocol)
    resumed_conditions = 0
    if output_root.exists() and allow_existing:
        order = efficiency_execution_order(checkpoints)
        resumed_conditions = len(
            EfficiencyExecutor(
                output_root, {"protocol_sha256": PROTOCOL_SHA256}, order
            ).completed()
        )
    if importlib.metadata.version("lpips") != protocol["lpips"]["package_version"]:
        raise RuntimeError("R14 requires the frozen LPIPS version")
    return {
        "status": "preflight_pass",
        "protocol_status": protocol["status"],
        "output_root": str(output_root),
        "samples_per_checkpoint": len(plan),
        "png_total": len(plan) * len(checkpoints),
        "lpips_pairs_per_checkpoint": 2000,
        "lpips_pairs_total": 12000,
        "resume_completed_conditions": resumed_conditions,
        "selected_checkpoints": list(checkpoints),
    }


def enumerate_r14_pairs(
    plan: Sequence[Mapping[str, Any]],
    manifest_by_sample: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan:
        groups[str(row["stable_record_key"])].append(row)
    pairs: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["generation_seed"]))
        for left, right in itertools.combinations(ordered, 2):
            left_artifact, right_artifact = (
                manifest_by_sample[str(left["sample_id"])],
                manifest_by_sample[str(right["sample_id"])],
            )
            pairs.append(
                {
                    "pair_id": f"{left['sample_id']}:{right['sample_id']}",
                    "prompt_id": key,
                    "panel_split": left["panel_split"],
                    "comp_family": left.get("comp_family"),
                    "seed_a": left["generation_seed"],
                    "seed_b": right["generation_seed"],
                    "png_a": left_artifact["png_relative_path"],
                    "png_b": right_artifact["png_relative_path"],
                }
            )
    if len(pairs) != 2000 or len({row["pair_id"] for row in pairs}) != 2000:
        raise ValueError("R14 pair geometry is invalid")
    return pairs


def aggregate_r14_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not math.isfinite(float(row["lpips"])):
            raise ValueError("non-finite R14 LPIPS")
        grouped[str(row["prompt_id"])].append(row)
    prompt_rows = []
    for prompt_id, values in sorted(grouped.items()):
        if len(values) != 10:
            raise ValueError("R14 requires ten pairs per prompt")
        first = values[0]
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "panel_split": first["panel_split"],
                "comp_family": first.get("comp_family"),
                "pair_count": 10,
                "mean_lpips": float(np.mean([row["lpips"] for row in values])),
            }
        )
    id_values = [row["mean_lpips"] for row in prompt_rows if row["panel_split"] == "ID"]
    families = {
        family: [
            row["mean_lpips"] for row in prompt_rows if row.get("comp_family") == family
        ]
        for family in ("CV1", "CV2", "CV3", "CV4")
    }
    if len(id_values) != 100 or any(len(values) != 25 for values in families.values()):
        raise ValueError("R14 aggregation population is invalid")
    aggregates = {"ID_VAL_prompt_macro_lpips": float(np.mean(id_values))}
    aggregates.update(
        {
            f"{family}_prompt_macro_lpips": float(np.mean(values))
            for family, values in families.items()
        }
    )
    aggregates["COMP_VAL_equal_family_macro_lpips"] = float(
        np.mean([aggregates[f"{family}_prompt_macro_lpips"] for family in families])
    )
    return prompt_rows, aggregates


def paired_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {
        (row["architecture_arm"], int(row["training_seed"])): row["aggregates"]
        for row in results
    }
    expected = {
        (arm, seed) for arm in ("ca8", "ca8_ca16") for seed in (2026, 2027, 2028)
    }
    if set(values) != expected:
        raise ValueError("R14 paired summary requires the six selected checkpoints")
    metrics = ("ID_VAL_prompt_macro_lpips", "COMP_VAL_equal_family_macro_lpips")
    paired = {
        str(seed): {
            metric: float(
                values[("ca8_ca16", seed)][metric] - values[("ca8", seed)][metric]
            )
            for metric in metrics
        }
        for seed in (2026, 2027, 2028)
    }
    means = {
        metric: float(
            np.mean([paired[str(seed)][metric] for seed in (2026, 2027, 2028)])
        )
        for metric in metrics
    }
    return {
        "comparison": "CA8+CA16 minus CA8",
        "paired_differences": paired,
        "mean_paired_differences": means,
        "inference": "descriptive across three training seeds; LPIPS pairs are not independent training replicates",
    }


def efficiency_execution_order(
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A fixed balanced order, recorded rather than chosen during benchmarking."""
    by_seed = {
        int(row["training_seed"]): row
        for row in checkpoints
        if row["architecture_arm"] == "ca8"
    }
    treatment = {
        int(row["training_seed"]): row
        for row in checkpoints
        if row["architecture_arm"] == "ca8_ca16"
    }
    if set(by_seed) != {2026, 2027, 2028} or set(treatment) != set(by_seed):
        raise ValueError("R14 efficiency order requires paired selected checkpoints")
    order: list[dict[str, Any]] = []
    for seed, arms in (
        (2026, ("ca8", "ca8_ca16")),
        (2027, ("ca8_ca16", "ca8")),
        (2028, ("ca8", "ca8_ca16")),
    ):
        indexed = {"ca8": by_seed[seed], "ca8_ca16": treatment[seed]}
        for batch_size in (1, 8):
            for arm in arms:
                order.append(
                    {
                        "ordinal": len(order),
                        "training_seed": seed,
                        "architecture_arm": arm,
                        "batch_size": batch_size,
                        "checkpoint_sha256": indexed[arm]["checkpoint_sha256"],
                    }
                )
    if len(order) != 12:
        raise ValueError("R14 efficiency order is incomplete")
    return order


def validate_parameter_counts(
    checkpoint: Mapping[str, Any],
    counts: Mapping[str, int],
    protocol: Mapping[str, Any],
) -> None:
    expected = protocol["efficiency"]["parameter_counts"]
    total = int(counts.get("total", -1))
    arm = checkpoint["architecture_arm"]
    required = expected["ca8_total"] if arm == "ca8" else expected["ca8_ca16_total"]
    if total != required or int(counts.get("text_encoder", -1)) != 102_976:
        raise ValueError("R14 loaded parameter count differs from frozen expectation")


def summarize_efficiency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure validation/aggregation for future measured records; no GPU actions."""
    if len(records) != 12:
        raise ValueError("R14 efficiency requires six checkpoints at B1 and B8")
    for row in records:
        if (
            row.get("batch_size") not in (1, 8)
            or len(row.get("raw_measured_seconds_per_batch", [])) != 10
        ):
            raise ValueError("R14 efficiency timing geometry is invalid")
        if not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in row["raw_measured_seconds_per_batch"]
        ):
            raise ValueError("R14 non-finite timing result")
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if int(row.get(name, -1)) < 0:
                raise ValueError("R14 invalid CUDA memory result")
    return {
        "comparison": "CA8+CA16 minus CA8 by training seed; descriptive only",
        "records": len(records),
    }


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EfficiencyExecutor:
    """Small fail-closed state machine for the twelve independently published conditions."""

    def __init__(
        self,
        root: Path,
        identities: Mapping[str, Any],
        order: Sequence[Mapping[str, Any]],
    ) -> None:
        self.root, self.identities, self.order = root, dict(identities), tuple(order)
        self.state_path = root / "r14_efficiency_state.json"

    def state(self) -> dict[str, Any]:
        plan_sha = _json_digest({"order": list(self.order)})
        expected = {
            "schema": "tiny_avatar_diffusion.ca16_r14_efficiency_state_v1",
            "identities": self.identities,
            "plan_sha256": plan_sha,
        }
        if self.state_path.exists():
            state = _read_json(self.state_path)
            if any(state.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "R14 efficiency state has incompatible protocol or plan identity"
                )
            return state
        self.root.mkdir(parents=True, exist_ok=True)
        state = {**expected, "completed_ordinals": [], "summary": "pending"}
        atomic_json(self.state_path, state)
        return state

    def path(self, ordinal: int) -> Path:
        return self.root / f"efficiency_{ordinal:02d}.json"

    def validate_record(self, condition: Mapping[str, Any]) -> dict[str, Any]:
        record = _read_json(self.path(int(condition["ordinal"])))
        expected = {
            key: condition[key]
            for key in (
                "ordinal",
                "training_seed",
                "architecture_arm",
                "batch_size",
                "checkpoint_sha256",
            )
        }
        if record.get(
            "schema"
        ) != "tiny_avatar_diffusion.ca16_r14_efficiency_condition_v1" or any(
            record.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("R14 efficiency record identity mismatch")
        digest = record.pop("record_sha256", None)
        if digest != _json_digest(record):
            raise ValueError("R14 efficiency record integrity mismatch")
        record["record_sha256"] = digest
        summarize_efficiency(
            [record] * 12
        )  # validates timing/memory fields without a GPU call
        return record

    def completed(self) -> dict[int, dict[str, Any]]:
        state = self.state()
        completed = {}
        for condition in self.order:
            path = self.path(int(condition["ordinal"]))
            if path.exists():
                completed[int(condition["ordinal"])] = self.validate_record(condition)
        if sorted(completed) != sorted(state["completed_ordinals"]):
            raise ValueError(
                "R14 efficiency state and published condition records disagree"
            )
        return completed

    def publish(self, condition: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        if self.path(int(condition["ordinal"])).exists():
            self.validate_record(condition)
            return
        payload = {
            "schema": "tiny_avatar_diffusion.ca16_r14_efficiency_condition_v1",
            **condition,
            **record,
        }
        payload["record_sha256"] = _json_digest(payload)
        atomic_json(self.path(int(condition["ordinal"])), payload)
        state = self.state()
        state["completed_ordinals"] = sorted(
            set(state["completed_ordinals"]) | {int(condition["ordinal"])}
        )
        atomic_json(self.state_path, state)


def _sample_rows(
    encoder: Any, denoiser: Any, rows: Sequence[Mapping[str, Any]], device: Any
) -> Any:
    """One FP32 ancestral-DDPM invocation with canonical per-record RNG streams."""
    import torch

    from tiny_avatar_diffusion.conditional_sampling import conditional_sample
    from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule

    ids = [row["token_ids"] for row in rows]
    masks = [row["attention_mask"] for row in rows]
    if len({len(item) for item in ids}) != 1 or any(
        len(left) != len(right) for left, right in zip(ids, masks, strict=True)
    ):
        raise ValueError("R14 token lanes are not batch compatible")
    generators = [
        torch.Generator(device=device).manual_seed(int(row["generation_seed"]))
        for row in rows
    ]

    def noise() -> Any:
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

    with torch.inference_mode():
        return conditional_sample(
            LinearDDPMSchedule(num_timesteps=1000).to(device),
            encoder,
            denoiser,
            torch.tensor(ids, device=device),
            torch.tensor(masks, device=device),
            x_T=noise(),
            reverse_noise_provider=lambda _t, _x: noise(),
            clip_denoised=True,
        )


def _benchmark_rows(
    plan: Sequence[Mapping[str, Any]], batch_size: int, iteration: int
) -> Sequence[Mapping[str, Any]]:
    if batch_size == 1:
        return (plan[iteration % len(plan)],)
    start = (iteration * batch_size) % len(plan)
    return plan[start : start + batch_size]


def _run_efficiency_worker(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str, ordinal: int
) -> dict[str, Any]:
    """One-condition CUDA worker.  It is never called by the parent directly."""
    import platform
    import time

    import torch

    from tiny_avatar_diffusion.ca16_evaluation import load_ema_checkpoint
    from tiny_avatar_diffusion.ca16_experiment import CA16RunSpec
    from tiny_avatar_diffusion.ca16_r12_orchestration import resolve_run_spec

    protocol = load_protocol(require_bound_hash=True)
    if protocol["status"] != "FROZEN" or not torch.cuda.is_available():
        raise RuntimeError("R14 efficiency execution requires frozen protocol and CUDA")
    # Worker validates identity without calling the output-root-unused preflight.
    head, branch, clean = _git_state()
    if (head, branch, clean) != (expected_eval_sha, "exp/ca16-ablation", True):
        raise ValueError("R14 worker requires the expected clean Git identity")
    tagged = subprocess.run(
        ("git", "rev-parse", f"{expected_tag}^{{commit}}"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tagged != head:
        raise ValueError("R14 worker tag identity mismatch")
    plan, checkpoints = resolve_inputs(protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    order = efficiency_execution_order(checkpoints)
    try:
        assigned = next(item for item in order if item["ordinal"] == ordinal)
    except StopIteration as error:
        raise ValueError("unknown R14 efficiency worker ordinal") from error
    executor = EfficiencyExecutor(
        output_root, {"protocol_sha256": PROTOCOL_SHA256}, order
    )
    if ordinal in executor.completed():
        return executor.validate_record(assigned)
    by_sha = {row["checkpoint_sha256"]: row for row in checkpoints}
    for condition in [assigned]:
        checkpoint = by_sha[condition["checkpoint_sha256"]]
        spec: CA16RunSpec = resolve_run_spec(checkpoint)
        encoder, denoiser, _artifact = load_ema_checkpoint(
            ROOT / checkpoint["checkpoint_path"],
            spec,
            20_000,
            device=torch.device("cuda"),
        )
        counts = {
            "text_encoder": sum(
                parameter.numel() for parameter in encoder.parameters()
            ),
            "denoiser": sum(parameter.numel() for parameter in denoiser.parameters()),
        }
        counts["total"] = counts["text_encoder"] + counts["denoiser"]
        validate_parameter_counts(checkpoint, counts, protocol)
        batch_size = int(condition["batch_size"])
        for iteration in range(int(protocol["efficiency"]["warmups"])):
            output = _sample_rows(
                encoder,
                denoiser,
                _benchmark_rows(plan, batch_size, iteration),
                torch.device("cuda"),
            )
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        observed: list[float] = []
        for iteration in range(int(protocol["efficiency"]["measured_iterations"])):
            rows = _benchmark_rows(plan, batch_size, iteration + 3)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = _sample_rows(encoder, denoiser, rows, torch.device("cuda"))
            torch.cuda.synchronize()
            observed.append(time.perf_counter() - started)
            if tuple(output.shape) != (batch_size, 3, 32, 32) or not bool(
                torch.isfinite(output).all()
            ):
                raise RuntimeError("R14 benchmark sampler returned invalid output")
            del output
        record = {
            **checkpoint,
            **condition,
            "parameter_counts": counts,
            "warmups": 3,
            "raw_measured_seconds_per_batch": observed,
            "median_seconds_per_batch": float(np.median(observed)),
            "images_per_second": float(batch_size / np.median(observed)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "environment": {
                "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
            },
        }
        executor.publish(condition, record)
        del encoder, denoiser
        torch.cuda.empty_cache()
    return executor.validate_record(assigned)


class R14Executor:
    """Immutable PNG publication and resume validation for one checkpoint."""

    def __init__(
        self,
        root: Path,
        plan: Sequence[Mapping[str, Any]],
        identities: Mapping[str, Any],
    ) -> None:
        self.root, self.plan, self.identities = root, tuple(plan), dict(identities)
        self.manifest_path = root / "generation_manifest.jsonl"
        self.state_path = root / "r14_state.json"

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

    def _state(self) -> dict[str, Any]:
        expected = {
            "schema": "tiny_avatar_diffusion.ca16_r14_state_v1",
            "identities": self.identities,
            "plan_sha256": self._plan_sha(),
        }
        if self.state_path.exists():
            state = _read_json(self.state_path)
            if any(state.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "R14 resume state has incompatible protocol, plan, or checkpoint identity"
                )
            return state
        if self.manifest_path.exists():
            raise ValueError("R14 manifest exists without bound resume state")
        state = {**expected, "generation": "pending"}
        atomic_json(self.state_path, state)
        return state

    def _expected(self, row: Mapping[str, Any]) -> dict[str, Any]:
        relative = (
            Path("images")
            / str(row["split_role"]).lower()
            / f"{int(row['planned_sample_index']):04d}.png"
        )
        return {
            "schema": "tiny_avatar_diffusion.ca16_r14_sample_v1",
            "sample_id": row["sample_id"],
            "planned_sample_index": row["planned_sample_index"],
            "stable_record_key": row["stable_record_key"],
            "split_role": row["split_role"],
            "generation_seed": row["generation_seed"],
            "checkpoint_path": self.identities["checkpoint_path"],
            "checkpoint_sha256": self.identities["checkpoint_sha256"],
            "architecture_arm": self.identities["architecture_arm"],
            "training_seed": self.identities["training_seed"],
            "png_relative_path": relative.as_posix(),
            "status": "complete",
        }

    def generate(self, generator: Callable[[Mapping[str, Any]], np.ndarray]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state = self._state()
        rows = _read_jsonl(self.manifest_path) if self.manifest_path.exists() else []
        known = {str(row.get("sample_id")): row for row in rows}
        if len(known) != len(rows):
            raise ValueError("R14 duplicate manifest records")
        if len(rows) > len(self.plan):
            raise ValueError("R14 generation manifest exceeds canonical plan")
        for index, row in enumerate(rows):
            expected = self._expected(self.plan[index])
            path = self.root / expected["png_relative_path"]
            if (
                any(row.get(key) != value for key, value in expected.items())
                or not path.is_file()
                or sha256_file(path) != row.get("png_sha256")
            ):
                raise ValueError("R14 existing PNG identity mismatch")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row.get(
                "pixel_sha256"
            ):
                raise ValueError("R14 existing PNG pixel identity mismatch")
        for item in self.plan:
            expected = self._expected(item)
            existing = known.get(str(item["sample_id"]))
            path = self.root / expected["png_relative_path"]
            if existing is not None:
                continue
            if path.exists():
                raise ValueError("R14 orphan PNG")
            row = {**expected, **_save_png(path, generator(item))}
            rows.append(row)
            known[str(item["sample_id"])] = row
            atomic_jsonl(self.manifest_path, rows)
        atomic_json(
            self.state_path,
            {
                **state,
                "generation": "complete",
                "generation_manifest_sha256": sha256_file(self.manifest_path),
            },
        )


def execute_diversity(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str
) -> dict[str, Any]:
    """Authorized future execution path; never called by preflight."""
    import torch

    protocol = load_protocol(require_bound_hash=True)
    if protocol["status"] != "FROZEN" or not torch.cuda.is_available():
        raise RuntimeError("R14 diversity execution requires frozen protocol and CUDA")
    preflight = validate_preflight(
        output_root=output_root,
        expected_eval_sha=expected_eval_sha,
        expected_tag=expected_tag,
    )
    plan, checkpoints = resolve_inputs(protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for checkpoint in checkpoints:
        run_root = output_root / checkpoint["run_id"]
        generator, _counts = _load_generator(checkpoint, plan, torch.device("cuda"))
        executor = R14Executor(
            run_root, plan, {"protocol_sha256": PROTOCOL_SHA256, **checkpoint}
        )
        executor.generate(generator)
        manifest = {
            str(row["sample_id"]): row for row in _read_jsonl(executor.manifest_path)
        }
        pairs = measure_pairs(
            enumerate_r14_pairs(plan, manifest), run_root, lpips_alexnet_metric("cuda")
        )
        prompt_rows, aggregates = aggregate_r14_pairs(pairs)
        atomic_jsonl(run_root / "lpips_pairs.jsonl", pairs)
        atomic_jsonl(run_root / "lpips_prompts.jsonl", prompt_rows)
        summary = {
            "schema": "tiny_avatar_diffusion.ca16_r14_diversity_v1",
            "status": "COMPLETE",
            **checkpoint,
            "aggregates": aggregates,
            "pair_count": len(pairs),
        }
        atomic_json(run_root / "r14_diversity_summary.json", summary)
        summaries.append(summary)
    result = {
        "schema": "tiny_avatar_diffusion.ca16_r14_complete_v1",
        "status": "COMPLETE",
        "preflight": preflight,
        "runs": summaries,
        "paired": paired_summary(summaries),
    }
    atomic_json(output_root / "R14_COMPLETE.json", result)
    return result


def execute_efficiency(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str
) -> dict[str, Any]:
    """CPU-only parent: dispatch one fresh interpreter per missing condition."""
    import sys

    protocol = load_protocol(require_bound_hash=True)
    if protocol["status"] != "FROZEN":
        raise RuntimeError("R14 efficiency execution requires a frozen protocol")
    head, branch, clean = _git_state()
    if (head, branch, clean) != (expected_eval_sha, "exp/ca16-ablation", True):
        raise ValueError("R14 parent requires the expected clean Git identity")
    tagged = subprocess.run(
        ("git", "rev-parse", f"{expected_tag}^{{commit}}"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tagged != head:
        raise ValueError("R14 parent tag identity mismatch")
    _plan, checkpoints = resolve_inputs(protocol)
    order = efficiency_execution_order(checkpoints)
    executor = EfficiencyExecutor(
        output_root, {"protocol_sha256": PROTOCOL_SHA256}, order
    )
    completed = executor.completed()
    summary_path = output_root / "r14_efficiency_summary.json"
    if summary_path.exists():
        if len(completed) != 12:
            raise ValueError("R14 efficiency summary exists before all conditions")
        return _read_json(summary_path)
    worker = ROOT / "scripts" / "run_ca16_r14_efficiency.py"
    for condition in order:
        if int(condition["ordinal"]) in completed:
            continue
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--worker-ordinal",
                str(condition["ordinal"]),
                "--output-root",
                str(output_root),
                "--expected-eval-sha",
                expected_eval_sha,
                "--expected-eval-tag",
                expected_tag,
            ],
            cwd=ROOT,
            check=True,
        )
        completed = executor.completed()
        if int(condition["ordinal"]) not in completed:
            raise RuntimeError("R14 worker exited without a valid published condition")
    records = [executor.validate_record(condition) for condition in order]
    result = {
        "schema": "tiny_avatar_diffusion.ca16_r14_efficiency_v1",
        "status": "COMPLETE",
        "records": records,
        "summary": summarize_efficiency(records),
    }
    atomic_json(summary_path, result)
    return result
