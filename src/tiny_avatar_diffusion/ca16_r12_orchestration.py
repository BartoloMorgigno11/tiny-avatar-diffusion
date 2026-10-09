"""Fail-closed R12 held-out evaluation for the R11-selected CA16 checkpoints.

Preflight deliberately hashes sealed inputs but never parses their prompt or
label records.  Parsing the DR04 panel is confined to the explicitly
confirmed execution path after the R12 protocol has been frozen.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_evaluation import (
    EVALUATOR_CHECKPOINT_RELATIVE,
    EVALUATOR_CHECKPOINT_SHA256,
    load_ema_checkpoint,
    load_frozen_evaluator,
    sha256_file,
)
from tiny_avatar_diffusion.ca16_experiment import ROOT, CA16RunSpec, ca16_run_matrix
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.dr04_final_conditioning_executor import (
    FinalConditioningExecutor,
    _json_sha,
    expand_plan,
    validate_panel,
)
from tiny_avatar_diffusion.dr05_final_executor import (
    atomic_json,
    load_jsonl,
    pixel_sha256,
)
from tiny_avatar_diffusion.evaluator import preprocess_cached
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

PROTOCOL_ID = "ca16_r12_selected_checkpoint_evaluation_batched_v1"
PROTOCOL_PATH = ROOT / "data/processed/ca16_r12_eval_v1/r12_execution_protocol.json"
PROTOCOL_SHA256 = "5c84c27da4303f685a3d3104cde128b4a05d4dde2ba50950b115014159183882"
R11_ROOT = ROOT / "results/r11_checkpoint_selection_reproducible_tokenfix_batched_v2"
R11_COMPLETE = R11_ROOT / "R11_COMPLETE.json"
INVENTORY_PATH = ROOT / "data/processed/ca16_eval_v1/r11_checkpoint_inventory.json"
PANEL_PATH = ROOT / "data/processed/dr04_final_eval_v1/final_prompt_panel_manifest.json"
CONTROLS_PATH = (
    ROOT / "data/processed/dr04_final_eval_v1/matched_controls_manifest.json"
)
DR04_PROTOCOL_PATH = (
    ROOT / "data/processed/dr04_final_eval_v1/final_evaluation_protocol.json"
)
VOCAB_PATH = ROOT / "data/processed/dr01_v1/text/10k/vocab.json"
EVALUATOR_PATH = ROOT / EVALUATOR_CHECKPOINT_RELATIVE
LOCK_NAME = ".r12.lock"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required R12 input is absent: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"R12 frozen input hash mismatch for {_relative(path)}")
    return actual


def load_protocol() -> dict[str, Any]:
    _require_hash(PROTOCOL_PATH, PROTOCOL_SHA256)
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("schema") != "tiny_avatar_diffusion.ca16_r12_execution_protocol_v1":
        raise ValueError("unexpected R12 protocol schema")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected R12 protocol identifier")
    return protocol


def _selection_path(spec: CA16RunSpec) -> Path:
    return R11_ROOT / "selections" / f"{spec.run_id}.json"


def resolve_selected_checkpoints() -> tuple[dict[str, Any], ...]:
    """Resolve winners from R11 records; this never ranks candidates again."""
    complete = _read_json(R11_COMPLETE)
    if (
        complete.get("schema") != "tiny_avatar_diffusion.ca16_r11_complete_v1"
        or complete.get("status") != "COMPLETE"
    ):
        raise ValueError("R11 completion marker is not complete")
    inventory = _read_json(INVENTORY_PATH)
    entries = inventory.get("entries")
    if inventory.get(
        "schema"
    ) != "tiny_avatar_diffusion.ca16_r11_checkpoint_inventory_v1" or not isinstance(
        entries, list
    ):
        raise ValueError("unexpected R11 checkpoint inventory")
    indexed = {
        entry.get("sha256"): entry for entry in entries if isinstance(entry, dict)
    }
    resolved: list[dict[str, Any]] = []
    for spec in ca16_run_matrix():
        path = _selection_path(spec)
        selection = _read_json(path)
        candidates = selection.get("candidates")
        winners = (
            [
                row
                for row in candidates
                if isinstance(row, dict) and row.get("selected") is True
            ]
            if isinstance(candidates, list)
            else []
        )
        if (
            selection.get("schema") != "tiny_avatar_diffusion.ca16_r11_run_selection_v1"
            or selection.get("run_id") != spec.run_id
            or selection.get("architecture_arm") != spec.architecture_arm.value
            or selection.get("training_seed") != spec.training_seed
            or len(winners) != 1
        ):
            raise ValueError(f"invalid R11 selection record for {spec.run_id}")
        winner = winners[0]
        if (
            winner.get("milestone_step") != 20_000
            or selection.get("selected_milestone") != 20_000
        ):
            raise ValueError(f"R12 requires the recorded 20k winner for {spec.run_id}")
        entry = indexed.get(winner.get("checkpoint_sha256"))
        if (
            not isinstance(entry, dict)
            or entry.get("run_id") != spec.run_id
            or entry.get("milestone") != 20_000
        ):
            raise ValueError(
                f"selected checkpoint is not inventory-bound for {spec.run_id}"
            )
        checkpoint = ROOT / str(entry.get("checkpoint_path"))
        if _require_hash(checkpoint, str(entry.get("sha256"))) != winner.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"selected checkpoint hash mismatch for {spec.run_id}")
        selection_sha256 = sha256_file(path)
        if complete.get("selection_sha256", {}).get(spec.run_id) != selection_sha256:
            raise ValueError(f"R11 completion binding mismatch for {spec.run_id}")
        resolved.append(
            {
                "run_id": spec.run_id,
                "architecture_arm": spec.architecture_arm.value,
                "training_seed": spec.training_seed,
                "milestone_step": 20_000,
                "checkpoint_path": str(entry["checkpoint_path"]),
                "checkpoint_sha256": str(entry["sha256"]),
                "selection_path": _relative(path),
                "selection_sha256": selection_sha256,
            }
        )
    if len(resolved) != 6 or len({row["checkpoint_sha256"] for row in resolved}) != 6:
        raise ValueError("R12 requires six distinct R11-selected checkpoints")
    return tuple(resolved)


def resolve_run_spec(checkpoint: Mapping[str, Any]) -> CA16RunSpec:
    """Resolve a winner through the frozen matrix instead of reconstructing it."""
    matches = [
        spec for spec in ca16_run_matrix() if spec.run_id == checkpoint.get("run_id")
    ]
    if len(matches) != 1:
        raise ValueError("R12 checkpoint has no unique frozen CA16 run specification")
    spec = matches[0]
    if (
        checkpoint.get("architecture_arm") != spec.architecture_arm.value
        or checkpoint.get("training_seed") != spec.training_seed
        or checkpoint.get("milestone_step") != 20_000
    ):
        raise ValueError(
            "R12 checkpoint does not match its frozen CA16 run specification"
        )
    return spec


def _git_state() -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return (
        run("git", "rev-parse", "HEAD"),
        run("git", "branch", "--show-current"),
        not bool(run("git", "status", "--porcelain")),
    )


def validate_preflight(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str | None = None
) -> dict[str, Any]:
    """Read only non-semantic metadata and cryptographic bindings of sealed inputs."""
    protocol = load_protocol()
    head, branch, clean = _git_state()
    if head != expected_eval_sha or branch != "exp/ca16-ablation" or not clean:
        raise ValueError(
            "R12 must start from the clean expected exp/ca16-ablation identity"
        )
    if expected_tag is not None:
        tag_commit = subprocess.run(
            ("git", "rev-parse", f"{expected_tag}^{{}}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tag_commit != head:
            raise ValueError(
                "R12 expected tag does not resolve to the execution identity"
            )
    reuse = protocol["dr04_reuse"]
    bindings = protocol["bindings"]
    hashes = {
        _relative(PROTOCOL_PATH): PROTOCOL_SHA256,
        _relative(PANEL_PATH): _require_hash(
            PANEL_PATH, reuse["panel_manifest_sha256"]
        ),
        _relative(CONTROLS_PATH): _require_hash(
            CONTROLS_PATH, reuse["matched_controls_sha256"]
        ),
        _relative(DR04_PROTOCOL_PATH): _require_hash(
            DR04_PROTOCOL_PATH, reuse["protocol_sha256"]
        ),
        _relative(R11_COMPLETE): _require_hash(
            R11_COMPLETE, bindings["r11_complete_sha256"]
        ),
        _relative(INVENTORY_PATH): _require_hash(
            INVENTORY_PATH, bindings["r11_inventory_sha256"]
        ),
        _relative(VOCAB_PATH): _require_hash(VOCAB_PATH, bindings["vocabulary_sha256"]),
        _relative(ROOT / bindings["evaluator_freeze_path"]): _require_hash(
            ROOT / bindings["evaluator_freeze_path"],
            bindings["evaluator_freeze_sha256"],
        ),
        _relative(EVALUATOR_PATH): _require_hash(
            EVALUATOR_PATH, bindings["evaluator_checkpoint_sha256"]
        ),
    }
    selected = resolve_selected_checkpoints()
    hashes.update(
        {str(row["selection_path"]): str(row["selection_sha256"]) for row in selected}
    )
    geometry = reuse["geometry"]
    if geometry != {
        "prompt_count": 252,
        "id_prompt_count": 152,
        "final_ood_prompt_count": 100,
        "seeds": [2026, 2027, 2028, 2029, 2030],
        "slots_per_model": 1260,
    }:
        raise ValueError("R12 protocol has an incompatible DR04 geometry")
    sampling = protocol["sampling"]
    if (
        sampling.get("weight_variant") != "EMA"
        or sampling.get("precision") != "FP32"
        or sampling.get("sampler") != "ancestral DDPM"
        or sampling.get("timesteps") != 1000
        or sampling.get("sampling_batch_size") != 8
        or sampling.get("batch_partitioning")
        != "canonical contiguous plan groups; incomplete resumed groups are regenerated in memory with their original lanes and only missing outputs are published"
    ):
        raise ValueError("R12 protocol sampling contract changed")
    if sampling.get("final_batch_padding") != (
        "the final four canonical records are repeated in canonical order as four "
        "non-published padding lanes, so every sampler invocation has exactly eight lanes"
    ):
        raise ValueError("R12 protocol final-batch padding contract changed")
    return {
        "status": "preflight_pass",
        "protocol_status": protocol.get("status"),
        "output_root": str(output_root),
        "output_exists": output_root.exists(),
        "hashes": hashes,
        "selected_checkpoints": list(selected),
        "geometry": {
            **geometry,
            "models": 6,
            "total_slots": 6 * int(geometry["slots_per_model"]),
        },
        "sealed_content_opened": False,
    }


def build_execution_plan(*, checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Open sealed records only from the confirmed post-freeze execution path."""
    rows = validate_panel(_read_json(PANEL_PATH), _read_json(CONTROLS_PATH))
    plan = expand_plan(
        rows, str(checkpoint["checkpoint_path"]), str(checkpoint["checkpoint_sha256"])
    )
    run_id = str(checkpoint["run_id"])
    for row in plan:
        row["sample_id"] = f"{run_id}__{row['sample_id']}"
        row["r12_run_id"] = run_id
        row["architecture_arm"] = checkpoint["architecture_arm"]
        row["training_seed"] = checkpoint["training_seed"]
        row["milestone_step"] = checkpoint["milestone_step"]
    validate_execution_plan(plan)
    return plan


def validate_execution_plan(plan: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the exact DR04 panel expansion after the sealed gate opens."""
    expected_seeds = {2026, 2027, 2028, 2029, 2030}
    if len(plan) != 1260 or len({row.get("sample_id") for row in plan}) != 1260:
        raise ValueError("R12 plan must contain 1,260 unique outputs")
    if [row.get("planned_sample_index") for row in plan] != list(range(1260)):
        raise ValueError("R12 plan indices are not canonical")
    for split, expected_prompts, expected_slots_per_seed in (
        ("ID", 152, 152),
        ("OOD", 100, 100),
    ):
        rows = [row for row in plan if row.get("panel_split") == split]
        if len(rows) != expected_prompts * 5:
            raise ValueError(f"R12 {split} output count mismatch")
        if len({row.get("prompt_id") for row in rows}) != expected_prompts:
            raise ValueError(f"R12 {split} prompt count mismatch")
        if {row.get("seed") for row in rows} != expected_seeds:
            raise ValueError(f"R12 {split} seed schedule mismatch")
        if any(
            sum(row.get("seed") == seed for row in rows) != expected_slots_per_seed
            for seed in expected_seeds
        ):
            raise ValueError(f"R12 {split} per-seed count mismatch")
        if len({(row.get("prompt_id"), row.get("seed")) for row in rows}) != len(rows):
            raise ValueError(f"R12 {split} prompt/seed identities are not unique")
    if any(row.get("panel_split") not in {"ID", "OOD"} for row in plan):
        raise ValueError("R12 plan contains an unauthorized panel role")


class R12Executor(FinalConditioningExecutor):
    """DR04 artifact/metric lifecycle with CA16 checkpoint-specific identities."""

    def state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = _read_json(self.state_path)
            if (
                state.get("schema") != "tiny_avatar_diffusion.ca16_r12_state_v1"
                or state.get("plan_sha256") != _json_sha(self.plan)
                or state.get("identities") != self.identities
            ):
                raise ValueError("R12 run identity mismatch")
            return state
        if (self.root / "r12_summary.json").exists():
            raise FileExistsError("R12 summary exists without state")
        state = {
            "schema": "tiny_avatar_diffusion.ca16_r12_state_v1",
            "version": 1,
            "plan_sha256": _json_sha(self.plan),
            "identities": self.identities,
            "phase": 0,
            "slots": {},
            "retries": {},
            "integrity": "pending",
            "predictions": "pending",
            "metrics": "pending",
            "summary": "pending",
            "started_perf_counter": self.clock(),
            "timings_seconds": {},
            "parameter_counts": None,
        }
        atomic_json(self.state_path, state)
        return state

    def record_parameter_counts(self, counts: dict[str, Any]) -> None:
        if set(counts) != {"text_encoder", "denoiser", "total", "evaluator"} or any(
            type(value) is not int or value <= 0 for value in counts.values()
        ):
            raise ValueError("invalid R12 parameter count record")
        state = self.state()
        state["parameter_counts"] = counts
        self._write(state)

    def finalize(self, environment: dict[str, Any]) -> dict[str, Any]:
        state = self.state()
        if (
            state.get("metrics") != "complete"
            or state.get("integrity") != "complete"
            or state.get("predictions") != "complete"
        ):
            raise RuntimeError(
                "all R12 phases through metrics are required before summary"
            )
        if sha256_file(self.manifest_path) != state.get(
            "generation_manifest_sha256"
        ) or sha256_file(self.predictions_path) != state.get(
            "prediction_manifest_sha256"
        ):
            raise ValueError("R12 evidence manifest identity changed")
        target = self.root / "r12_summary.json"
        if target.exists():
            raise FileExistsError("R12 summary is immutable")
        metrics = _read_json(self.root / "metrics.json")
        state["timings_seconds"]["total"] = (
            time.perf_counter() - state["started_perf_counter"]
        )
        payload = {
            "schema": "tiny_avatar_diffusion.ca16_r12_summary_v1",
            "status": "COMPLETE",
            "run_identity": {"plan_sha256": _json_sha(self.plan), **self.identities},
            "environment": environment,
            "parameter_counts": state["parameter_counts"],
            "generation": {
                "completed": len(self.rows()),
                "errors": state["slots"],
                "retries": state["retries"],
            },
            "metrics": metrics,
            "evidence": {
                "generation_manifest_sha256": state["generation_manifest_sha256"],
                "prediction_manifest_sha256": state["prediction_manifest_sha256"],
            },
        }
        atomic_json(target, payload)
        state["summary"] = "complete"
        state["phase"] = 5
        self._write(state)
        return payload


class BatchedPlanGenerator:
    """B8 sampler retaining original lanes when an interrupted group is resumed."""

    def __init__(
        self,
        *,
        plan: Sequence[Mapping[str, Any]],
        text_encoder: TextConditioningEncoder,
        denoiser: ConditionalTinyUNet,
        tokenizer: WhitespaceTokenizer,
        device: torch.device,
        batch_size: int = 8,
    ) -> None:
        self.plan = tuple(plan)
        self.text_encoder, self.denoiser, self.tokenizer, self.device = (
            text_encoder,
            denoiser,
            tokenizer,
            device,
        )
        self.batch_size, self.schedule, self.cache = (
            batch_size,
            LinearDDPMSchedule(num_timesteps=1000).to(device),
            {},
        )
        self.indices = {
            str(row["sample_id"]): index for index, row in enumerate(self.plan)
        }

    def _lanes(
        self, start: int
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
        real = self.plan[start : start + self.batch_size]
        if not real:
            raise ValueError("R12 batch start is outside the canonical plan")
        if len(real) == self.batch_size:
            return real, real
        if len(real) != 4:
            raise ValueError("R12 permits only the frozen four-record padded tail")
        return real, real + real

    def _generate_group(self, start: int) -> None:
        real, lanes = self._lanes(start)
        token_rows, mask_rows = [], []
        for row in lanes:
            encoded = self.tokenizer.encode(str(row["prompt"]))
            token_rows.append(encoded.token_ids)
            mask_rows.append(encoded.attention_mask)
        generators = [
            torch.Generator(device=self.device).manual_seed(int(row["seed"]))
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
                self.text_encoder,
                self.denoiser,
                torch.tensor(token_rows, device=self.device),
                torch.tensor(mask_rows, device=self.device),
                x_T=noise(),
                reverse_noise_provider=lambda _t, _x: noise(),
                clip_denoised=True,
            )
        if len(samples) != self.batch_size:
            raise ValueError("R12 sampler did not return exactly eight lanes")
        for row, sample in zip(real, samples[: len(real)], strict=True):
            self.cache[str(row["sample_id"])] = (
                ((sample.detach().cpu().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )

    def __call__(self, slot: Mapping[str, Any]) -> np.ndarray:
        sample_id = str(slot["sample_id"])
        if sample_id not in self.cache:
            index = self.indices.get(sample_id)
            if index is None or self.plan[index] != slot:
                raise ValueError(
                    "R12 generation slot is not part of the canonical plan"
                )
            self._generate_group(index - index % self.batch_size)
        return self.cache.pop(sample_id)


def _parameter_count(module: Any) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _load_generator(
    checkpoint: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[BatchedPlanGenerator, dict[str, int]]:
    spec = resolve_run_spec(checkpoint)
    encoder, denoiser, artifact = load_ema_checkpoint(
        ROOT / str(checkpoint["checkpoint_path"]), spec, 20_000, device=device
    )
    experiment = artifact.get("experiment", {})
    if experiment.get("run_id") != spec.run_id:
        raise ValueError("R12 loaded checkpoint identity mismatch")
    vocab = _read_json(VOCAB_PATH)
    tokenizer = WhitespaceTokenizer(vocab["token_to_id"])
    return BatchedPlanGenerator(
        plan=plan,
        text_encoder=encoder,
        denoiser=denoiser,
        tokenizer=tokenizer,
        device=device,
    ), {
        "text_encoder": _parameter_count(encoder),
        "denoiser": _parameter_count(denoiser),
        "total": _parameter_count(encoder) + _parameter_count(denoiser),
        "evaluator": 293_064,
    }


def validate_completed_run(executor: R12Executor) -> dict[str, Any]:
    """Read-only completion validation before a resume accepts prior R12 evidence."""
    summary_path = executor.root / "r12_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("R12 completion summary is absent")
    state = executor.state()
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != "tiny_avatar_diffusion.ca16_r12_summary_v1"
        or summary.get("status") != "COMPLETE"
        or summary.get("run_identity")
        != {"plan_sha256": _json_sha(executor.plan), **executor.identities}
        or state.get("phase") != 5
        or state.get("summary") != "complete"
    ):
        raise ValueError("R12 completion identity or status is inconsistent")
    rows = executor.rows()
    if len(rows) != len(executor.plan) or len(
        {row.get("sample_id") for row in rows}
    ) != len(rows):
        raise ValueError("R12 completed generation evidence is incomplete")
    for row in rows:
        path = executor.root / str(row.get("png_path"))
        if not path.is_file():
            raise FileNotFoundError("R12 completed PNG is missing")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (32, 32, 3) or pixel_sha256(pixels) != row.get(
            "pixel_sha256"
        ):
            raise ValueError("R12 completed PNG integrity mismatch")
    predictions = load_jsonl(executor.predictions_path)
    if len(predictions) != len(executor.plan) or len(
        {row.get("sample_id") for row in predictions}
    ) != len(predictions):
        raise ValueError("R12 completed prediction evidence is incomplete")
    evidence = summary.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("generation_manifest_sha256")
        != sha256_file(executor.manifest_path)
        or evidence.get("prediction_manifest_sha256")
        != sha256_file(executor.predictions_path)
        or evidence
        != {
            "generation_manifest_sha256": state.get("generation_manifest_sha256"),
            "prediction_manifest_sha256": state.get("prediction_manifest_sha256"),
        }
    ):
        raise ValueError("R12 completed evidence hashes are inconsistent")
    generation = summary.get("generation")
    if not isinstance(generation, dict) or generation.get("completed") != len(
        executor.plan
    ):
        raise ValueError("R12 summary has an incomplete generation count")
    return summary


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            "R12 output lock exists; inspect it before any recovery"
        ) from error
    try:
        os.write(
            descriptor,
            json.dumps({"pid": os.getpid(), "protocol_id": PROTOCOL_ID}).encode(
                "utf-8"
            ),
        )
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def execute_r12(
    *, output_root: Path, expected_eval_sha: str, expected_tag: str | None = None
) -> dict[str, Any]:
    if expected_tag is None:
        raise ValueError("scientific R12 execution requires an expected annotated tag")
    preflight = validate_preflight(
        output_root=output_root,
        expected_eval_sha=expected_eval_sha,
        expected_tag=expected_tag,
    )
    if preflight["protocol_status"] != "FROZEN":
        raise RuntimeError("R12 execution requires a frozen protocol review")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("scientific R12 execution requires CUDA")
    results: list[dict[str, Any]] = []
    with exclusive_run_lock(output_root):
        for checkpoint in preflight["selected_checkpoints"]:
            plan = build_execution_plan(checkpoint=checkpoint)
            root = output_root / str(checkpoint["run_id"])
            identities = {
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "r11_selection_sha256": checkpoint["selection_sha256"],
                "checkpoint_path": checkpoint["checkpoint_path"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "evaluator_checkpoint_path": EVALUATOR_CHECKPOINT_RELATIVE,
                "evaluator_checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
                "sampling_batch_size": "8",
            }
            executor = R12Executor(root, plan, identities)
            if (root / "r12_summary.json").is_file():
                results.append(validate_completed_run(executor))
                continue
            generator, counts = _load_generator(checkpoint, plan, device)
            evaluator, evaluator_identity = load_frozen_evaluator(device=device)
            executor.record_parameter_counts(counts)
            if executor.state().get("phase", 0) < 1:
                executor.generate(generator)
            if executor.state().get("phase", 0) < 2:
                executor.audit()
            if executor.state().get("phase", 0) < 3:

                def predict(
                    pixels: np.ndarray,
                    evaluator: Any = evaluator,
                ) -> dict[str, int]:
                    with torch.no_grad():
                        values = evaluator(
                            preprocess_cached(Image.fromarray(pixels, mode="RGB"))
                            .unsqueeze(0)
                            .to(device)
                        )
                    return {
                        name: int(value.argmax(dim=1).item())
                        for name, value in values.items()
                    }

                executor.predict(predict)
            if executor.state().get("phase", 0) < 4:
                executor.metrics()
            results.append(
                executor.finalize(
                    {
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "device": str(device),
                        "evaluator_identity": evaluator_identity,
                    }
                )
            )
    return {
        "schema": "tiny_avatar_diffusion.ca16_r12_execution_result_v1",
        "runs": results,
    }
