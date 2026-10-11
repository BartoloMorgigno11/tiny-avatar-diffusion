"""Fail-closed H7 binding for the frozen R11 conditioning endpoint.

This is deliberately not R11 orchestration: it accepts one H7 40k terminal
checkpoint while resolving every frozen historical R11 input from an explicit,
read-only CA16 worktree.  Only H7 code is imported.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion import ca16_evaluation as r11
from tiny_avatar_diffusion.ca16_experiment import ROOT, SUPPORTED_SEEDS
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.h7_extended_training import (
    EXPECTED_SOURCE_SHA256,
    EXPERIMENT_ID,
    TARGET_STEP,
    sha256_file,
)
from tiny_avatar_diffusion.h7_extended_training import (
    SCHEMA as H7_SCHEMA,
)
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

PROTOCOL_ID = "h7_r11_terminal_evaluation_v1"
OUTPUT_ROOT = ROOT / "results" / EXPERIMENT_ID
CA16_R11_ROOT = Path(
    "results/r11_checkpoint_selection_reproducible_tokenfix_batched_v2"
)
CA16_REQUIRED = {
    "selection_manifest": (
        "data/processed/ca16_eval_v1/checkpoint_selection_manifest.json",
        r11.SELECTION_MANIFEST_SHA256,
    ),
    "robust_manifest": (
        "data/processed/ca16_eval_v1/robust_conditioning_manifest.json",
        r11.ROBUST_MANIFEST_SHA256,
    ),
    "evaluator_freeze": (
        "data/processed/dr04_evaluator_v1/evaluator_checkpoint_freeze.json",
        r11.EVALUATOR_FREEZE_SHA256,
    ),
    "evaluator_checkpoint": (
        r11.EVALUATOR_CHECKPOINT_RELATIVE,
        r11.EVALUATOR_CHECKPOINT_SHA256,
    ),
    "r11_inventory": (
        "data/processed/ca16_eval_v1/r11_checkpoint_inventory.json",
        "cace0eb71751c770c4e4c1318e9683813566fd1a36b7595b4bb8ab84a37e7790",
    ),
    "r11_complete": (
        str(CA16_R11_ROOT / "R11_COMPLETE.json"),
        "3fd6c3b3caf8df527c5d93d2320398ce492dea57a0b76243c420ac02ffde0122",
    ),
    "r11_summary": (
        str(CA16_R11_ROOT / "r11_summary.json"),
        "3349cdbd2f91493374f95d197cbc14e4f262b3a385433bdb54cffdbe9411a901",
    ),
    "id_tokens": (
        "data/processed/dr01_v1/text/10k/id_val.jsonl",
        "34d610b9a9d3b8f20caf3004e93c154aa7646319006a53c2552b7d994850b487",
    ),
    "comp_tokens": (
        "data/processed/dr01_v1/text/10k/comp_val.jsonl",
        "cfb3467420469c3b34c09d0a4b21ec1a336f5446276d8ae2a1fa6c6d9cb1025e",
    ),
}
BASELINE_MAA = {
    2026: 0.670254194514063,
    2027: 0.7919420948203844,
    2028: 0.7501475041771094,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        # Invalid external protocol content is a value-contract failure.
        raise ValueError(f"JSON object required: {path}")  # noqa: TRY004
    return value


def _external_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("external CA16 path escaped root")
    return path


def verify_ca16_assets(root: Path) -> dict[str, Any]:
    """Prove, without importing CA16 code, every R11 input this adapter uses."""
    root = root.resolve()
    if root == ROOT.resolve() or not (root / ".git").exists():
        raise ValueError("--ca16-assets-root must name a separate CA16 Git worktree")
    evidence: dict[str, Any] = {}
    for name, (relative, expected) in CA16_REQUIRED.items():
        path = _external_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"CA16 R11 identity mismatch: {name}")
        evidence[name] = {"path": relative.replace("\\", "/"), "sha256": expected}
    complete = _read(_external_path(root, str(CA16_R11_ROOT / "R11_COMPLETE.json")))
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("schema") != "tiny_avatar_diffusion.ca16_r11_complete_v1"
    ):
        raise ValueError("authoritative R11 completion marker is invalid")
    return evidence


def verify_terminal_checkpoint(path: Path, seed: int) -> dict[str, Any]:
    if seed not in SUPPORTED_SEEDS:
        raise ValueError("H7 evaluation accepts only seeds 2026, 2027, and 2028")
    path = path.resolve()
    expected = (
        ROOT
        / "results"
        / EXPERIMENT_ID
        / f"{EXPERIMENT_ID}_seed{seed}"
        / "terminal_40000.pt"
    ).resolve()
    if path != expected or ROOT.resolve() not in path.parents:
        raise ValueError("terminal checkpoint is not the exact H7-only terminal path")
    if not path.is_file():
        raise FileNotFoundError(f"H7 terminal checkpoint is absent: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    continuation = payload.get("continuation", {}) if isinstance(payload, dict) else {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != H7_SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("run_id") != f"{EXPERIMENT_ID}_seed{seed}"
        or payload.get("progress", {}).get("step") != TARGET_STEP
        or continuation.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256[seed]
        or continuation.get("target_successful_step") != TARGET_STEP
        or set(payload.get("ema_state_dict", {}))
        != {"text_encoder", "conditional_unet"}
    ):
        raise ValueError("checkpoint is not a verified H7 CA8+CA16 40k terminal state")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "seed": seed,
        "step": TARGET_STEP,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256[seed],
    }


def load_h7_terminal_ema(
    path: Path, seed: int, *, device: torch.device | str
) -> tuple[TextConditioningEncoder, ConditionalTinyUNet, dict[str, Any]]:
    """Load only verified H7 terminal EMA weights; raw weights are never used."""
    identity = verify_terminal_checkpoint(path, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ema = payload["ema_state_dict"]
    if set(ema) != {"text_encoder", "conditional_unet"}:
        raise ValueError("H7 terminal EMA state is incomplete")
    text = TextConditioningEncoder().to(device)
    denoiser = ConditionalTinyUNet(use_decoder16_cross_attention=True, dropout=0.0).to(
        device
    )
    text.load_state_dict(ema["text_encoder"])
    denoiser.load_state_dict(ema["conditional_unet"])
    text.eval()
    denoiser.eval()
    return text, denoiser, identity


def frozen_plan(ca16_assets_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return the exact R11 panel and generation assignments, with no test roles."""
    root = ca16_assets_root.resolve()
    verify_ca16_assets(root)
    panel = _read(_external_path(root, CA16_REQUIRED["selection_manifest"][0]))
    output = {
        "ID_VAL": list(panel["id_entries"]),
        "COMP_VAL": list(panel["comp_entries"]),
    }
    if {role: len(rows) for role, rows in output.items()} != {
        "ID_VAL": 256,
        "COMP_VAL": 237,
    }:
        raise ValueError("R11 panel geometry differs")
    token_sources = {
        "ID_VAL": _external_path(root, CA16_REQUIRED["id_tokens"][0]),
        "COMP_VAL": _external_path(root, CA16_REQUIRED["comp_tokens"][0]),
    }
    for role, rows in output.items():
        tokens = {
            "|".join(
                str(item[field])
                for field in ("dataset_variant", "split_role", "image_id")
            ): item
            for item in (
                json.loads(line)
                for line in token_sources[role].read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        for index, row in enumerate(rows):
            if row.get("split_role") != role:
                raise ValueError("R11 prompt or generation seed assignment differs")
            generation_seed = 2026 + index
            if role == "COMP_VAL" and row.get("generation_seed") != generation_seed:
                raise ValueError("R11 prompt or generation seed assignment differs")
            if role == "ID_VAL" and "generation_seed" in row:
                raise ValueError(
                    "R11 ID prompt unexpectedly supplies a generation seed"
                )
            if "token_ids" in row or "attention_mask" in row:
                raise ValueError(
                    "selection manifest must not substitute prompt token IDs"
                )
            source = tokens.get(row["stable_record_key"])
            if source is None or any(
                source.get(key) != row.get(key)
                for key in ("caption", "split_role", "image_id")
            ):
                raise ValueError("R11 token source prompt identity differs")
            if not isinstance(source.get("token_ids"), list) or not isinstance(
                source.get("attention_mask"), list
            ):
                raise ValueError("R11 token source lacks prompt token IDs")  # noqa: TRY004
            row.update(
                token_ids=source["token_ids"],
                attention_mask=source["attention_mask"],
                generation_seed=generation_seed,
            )
    return output


def aggregate_predictions(
    plan: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Sequence[Mapping[str, int]]],
) -> dict[str, Any]:
    """R11 metric aggregation; prediction rows remain one row per prompt/sample."""
    if set(plan) != {"ID_VAL", "COMP_VAL"} or set(predictions) != set(plan):
        raise ValueError("only ID_VAL and COMP_VAL evaluation records are authorized")
    id_metrics = r11.attribute_metrics(plan["ID_VAL"], predictions["ID_VAL"])
    comp = r11.family_macro_metrics(plan["COMP_VAL"], predictions["COMP_VAL"])
    return {
        "id_maa": id_metrics["maa"],
        "id_metrics": id_metrics,
        "comp_family_macro_maa": comp["family_macro_maa"],
        "comp_macro_psr": comp["macro_psr"],
        "comp_per_family": comp["per_family"],
        "comp_family_counts": r11.EXPECTED_SELECTION_FAMILY_COUNTS,
    }


def preflight(
    *, ca16_assets_root: Path, terminal_checkpoint: Path, seed: int
) -> dict[str, Any]:
    external = verify_ca16_assets(ca16_assets_root)
    terminal = verify_terminal_checkpoint(terminal_checkpoint, seed)
    plan = frozen_plan(ca16_assets_root)
    return {
        "schema": "tiny_avatar_diffusion.h7_r11_preflight_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "READY_FOR_EXPLICIT_GPU_EXECUTION",
        "terminal": terminal,
        "r11_inputs": external,
        "panel_counts": {role: len(rows) for role, rows in plan.items()},
        "generation": dict(r11.FROZEN_GENERATION),
        "evaluator": {
            "checkpoint_sha256": r11.EVALUATOR_CHECKPOINT_SHA256,
            "freeze_sha256": r11.EVALUATOR_FREEZE_SHA256,
            "ema": True,
            "sampler": "DDPM ancestral FP32 T=1000",
        },
        "baseline_20k_comp_family_macro_maa": BASELINE_MAA[seed],
        "leakage_guard": "FINAL_OOD and ID_TEST are absent from this adapter",
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_root(seed: int) -> Path:
    return OUTPUT_ROOT / f"r11_terminal_seed{seed}"


def _prediction_identity(
    row: Mapping[str, Any], terminal: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stable_record_key": row["stable_record_key"],
        "split_role": row["split_role"],
        "generation_seed": row["generation_seed"],
        "terminal_sha256": terminal["sha256"],
    }


def execute(
    *, ca16_assets_root: Path, terminal_checkpoint: Path, seed: int
) -> dict[str, Any]:
    """Future CUDA-only R11-compatible H7 evaluation; never called implicitly."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "H7 R11 execution requires CUDA; use preflight for CPU-only validation"
        )
    report = preflight(
        ca16_assets_root=ca16_assets_root,
        terminal_checkpoint=terminal_checkpoint,
        seed=seed,
    )
    root = _output_root(seed)
    summary_path = root / "H7_R11_COMPLETE.json"
    if summary_path.exists():
        summary = _read(summary_path)
        if summary.get("terminal") != report["terminal"]:
            raise ValueError(
                "completed H7 R11 output has a different terminal identity"
            )
        return summary
    predictions_path = root / "predictions.jsonl"
    existing = (
        []
        if not predictions_path.exists()
        else [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    )
    plan = frozen_plan(ca16_assets_root)
    ordered = plan["ID_VAL"] + plan["COMP_VAL"]
    known = {row.get("stable_record_key"): row for row in existing}
    if len(known) != len(existing):
        raise ValueError("H7 prediction records have duplicate prompt identities")
    text, denoiser, _ = load_h7_terminal_ema(terminal_checkpoint, seed, device="cuda")
    evaluator, evaluator_identity = r11.load_frozen_evaluator(
        device="cuda",
        asset_root=ca16_assets_root,
        freeze_path=_external_path(
            ca16_assets_root, CA16_REQUIRED["evaluator_freeze"][0]
        ),
        checkpoint_path=_external_path(
            ca16_assets_root, CA16_REQUIRED["evaluator_checkpoint"][0]
        ),
    )
    schedule = LinearDDPMSchedule(num_timesteps=1000).to("cuda")
    for row in ordered:
        identity = _prediction_identity(row, report["terminal"])
        prior = known.get(row["stable_record_key"])
        if prior is not None:
            if prior.get("identity") != identity or not isinstance(
                prior.get("prediction"), dict
            ):
                raise ValueError("H7 prediction resume identity is corrupt")
            continue
        sample = r11.sample_ema_record(
            schedule,
            text,
            denoiser,
            row,
            device=torch.device("cuda"),
            canonical_record=row,
        )
        prediction = r11.predict_attributes(evaluator, sample.unsqueeze(0))[0]
        record = {
            "schema": "tiny_avatar_diffusion.h7_r11_prediction_v1",
            "identity": identity,
            "ground_truth_attributes": row["ground_truth_attributes"],
            "prediction": prediction,
        }
        existing.append(record)
        known[row["stable_record_key"]] = record
        atomic_jsonl(predictions_path, existing)
    by_key = {
        row["identity"]["stable_record_key"]: row["prediction"] for row in existing
    }
    metrics = aggregate_predictions(
        plan,
        {
            role: [by_key[row["stable_record_key"]] for row in rows]
            for role, rows in plan.items()
        },
    )
    difference = metrics["comp_family_macro_maa"] - BASELINE_MAA[seed]
    summary = {
        "schema": "tiny_avatar_diffusion.h7_r11_complete_v1",
        "status": "COMPLETE",
        "terminal": report["terminal"],
        "evaluator": evaluator_identity,
        "prediction_count": len(existing),
        "metrics": metrics,
        "baseline_20k_comp_family_macro_maa": BASELINE_MAA[seed],
        "paired_comp_family_macro_maa_change": difference,
        "warning": "COMP_VAL was used for historical checkpoint selection; this is not held-out confirmation",
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca16-assets-root", type=Path, required=True)
    parser.add_argument("--terminal-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--write-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.write_preflight:
        raise SystemExit("--execute and --write-preflight are mutually exclusive")
    if args.execute:
        print(
            json.dumps(
                execute(
                    ca16_assets_root=args.ca16_assets_root,
                    terminal_checkpoint=args.terminal_checkpoint,
                    seed=args.seed,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = preflight(
        ca16_assets_root=args.ca16_assets_root,
        terminal_checkpoint=args.terminal_checkpoint,
        seed=args.seed,
    )
    if args.write_preflight:
        atomic_json(OUTPUT_ROOT / f"seed{args.seed}" / "preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
