"""H7-only, resumable shared R13 image pool (no metric computation)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_avatar_diffusion.ca16_evaluation import sha256_file
from tiny_avatar_diffusion.ca16_experiment import ROOT
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.h7_r11_evaluation import (
    _external_path,
    load_h7_terminal_ema,
    verify_ca16_assets,
    verify_terminal_checkpoint,
)

PROTOCOL_RELATIVE = "data/processed/ca16_r13_quality_v1/r13_execution_protocol.json"
PROTOCOL_SHA256 = "fe292f1a67b4f755bfae0512601b206f3abd7d4ec4a2cae4bc820098631f240b"
OUTPUT_ROOT = ROOT / "results" / "h7_ca16_40k_v1"
BATCH_SIZE = 8


def pool_root(seed: int) -> Path:
    """Isolated R13 B8 namespace; never resumes the preserved B1 evidence."""
    return OUTPUT_ROOT / f"r13_shared_pool_b8_seed{seed}"


def batch_lanes(
    plan: Sequence[Mapping[str, Any]], start: int
) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    """Return CA16's real and eight sampling lanes for one canonical group."""
    real = plan[start : start + BATCH_SIZE]
    lanes = real if len(real) == BATCH_SIZE else real + real
    if len(lanes) != BATCH_SIZE or len(real) not in {4, 8}:
        raise ValueError("R13 requires full B8 groups and a repeated four-lane tail")
    return real, lanes


class _BatchedGenerator:
    """CA16 R13 B8 lanes, including the frozen repeated four-lane tail."""

    def __init__(
        self, plan: Sequence[Mapping[str, Any]], text: Any, denoiser: Any
    ) -> None:
        self.plan, self.text, self.denoiser = tuple(plan), text, denoiser
        self.schedule = LinearDDPMSchedule(1000).to("cuda")
        self.cache: dict[str, np.ndarray] = {}
        self.indices = {
            str(row["sample_id"]): index for index, row in enumerate(plan)
        }

    def _generate_group(self, start: int) -> None:
        real, lanes = batch_lanes(self.plan, start)
        ids = [row["token_ids"] for row in lanes]
        masks = [row["attention_mask"] for row in lanes]
        if len({len(row) for row in ids}) != 1 or any(
            len(a) != len(b) for a, b in zip(ids, masks, strict=True)
        ):
            raise ValueError("R13 token rows are not B8-compatible")
        generators = [
            torch.Generator(device="cuda").manual_seed(int(row["generation_seed"]))
            for row in lanes
        ]

        def noise() -> torch.Tensor:
            return torch.cat(
                [
                    torch.randn(
                        (1, 3, 32, 32),
                        device="cuda",
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
                self.text,
                self.denoiser,
                torch.tensor(ids, device="cuda"),
                torch.tensor(masks, device="cuda"),
                x_T=noise(),
                reverse_noise_provider=lambda _t, _x: noise(),
                clip_denoised=True,
            )
        for row, sample in zip(real, samples[: len(real)], strict=True):
            self.cache[str(row["sample_id"])] = (
                ((sample.detach().cpu().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )

    def __call__(self, row: Mapping[str, Any]) -> np.ndarray:
        key = str(row["sample_id"])
        if key not in self.cache:
            index = self.indices.get(key)
            if index is None or self.plan[index] != row:
                raise ValueError("R13 row is outside the canonical plan")
            self._generate_group(index - index % BATCH_SIZE)
        return self.cache.pop(key)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")  # noqa: TRY004
    return value


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_plan(ca16_assets_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hydrate exactly the frozen R13 804-record plan from CA16, never H7."""
    verify_ca16_assets(ca16_assets_root)
    protocol_path = _external_path(ca16_assets_root, PROTOCOL_RELATIVE)
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("R13 protocol hash mismatch")
    protocol = _json(protocol_path)
    manifest_path = _external_path(
        ca16_assets_root, protocol["quality_manifest"]["path"]
    )
    if sha256_file(manifest_path) != protocol["quality_manifest"]["sha256"]:
        raise ValueError("R13 quality manifest hash mismatch")
    manifest = _json(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 804:
        raise ValueError("R13 plan must contain 804 records")
    tokens = {}
    for binding in protocol["token_sources"].values():
        source = _external_path(ca16_assets_root, binding["path"])
        if sha256_file(source) != binding["sha256"]:
            raise ValueError("R13 token source hash mismatch")
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            tokens[
                f"{row['dataset_variant']}|{row['split_role']}|{row['image_id']}"
            ] = row
    plan = []
    for index, row in enumerate(entries):
        if (
            row.get("canonical_manifest_index") != index
            or row.get("generation_seed") != 2026 + index
            or row.get("split_role") not in {"ID_VAL", "COMP_VAL"}
        ):
            raise ValueError("R13 canonical record mismatch")
        token = tokens.get(row["stable_record_key"])
        if token is None or not isinstance(token.get("token_ids"), list):
            raise ValueError("R13 canonical token identity mismatch")
        plan.append(
            {
                **row,
                "token_ids": token["token_ids"],
                "attention_mask": token["attention_mask"],
                "sample_id": f"quality_{index:04d}",
                "png_relative_path": f"images/{row['split_role'].lower()}/{index:04d}.png",
            }
        )
    if {
        role: sum(row["split_role"] == role for row in plan)
        for role in ("ID_VAL", "COMP_VAL")
    } != {"ID_VAL": 466, "COMP_VAL": 338}:
        raise ValueError("R13 grouping mismatch")
    identity = {
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return plan, identity


def validate_pool(
    root: Path,
    plan: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    state = _json(root / "pool_state.json")
    if state.get("identity") != dict(identity) or state.get("terminal") != dict(
        terminal
    ):
        raise ValueError("pool identity mismatch")
    rows = [
        json.loads(line)
        for line in (root / "pool_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if len(rows) != len(plan):
        raise ValueError("pool is incomplete")
    expected_paths = {str(item["png_relative_path"]) for item in plan}
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*.png")}
    if actual_paths != expected_paths:
        raise ValueError("pool PNG membership is incomplete or has foreign images")
    for row, item in zip(rows, plan, strict=True):
        if (
            row.get("sample_id") != item["sample_id"]
            or row.get("generation_seed") != item["generation_seed"]
        ):
            raise ValueError("pool manifest mismatch")
        path = root / row["png_relative_path"]
        if not path.is_file() or sha256_file(path) != row.get("png_sha256"):
            raise ValueError("pool PNG hash mismatch")
    return rows


def execute_pool(
    *,
    ca16_assets_root: Path,
    terminal_checkpoint: Path,
    seed: int,
    generator: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
) -> dict[str, Any]:
    terminal = verify_terminal_checkpoint(terminal_checkpoint, seed)
    plan, identity = prepare_plan(ca16_assets_root)
    root = pool_root(seed)
    complete = root / "R13_POOL_COMPLETE.json"
    if complete.exists():
        summary = _json(complete)
        if summary.get("identity") != identity or summary.get("terminal") != terminal:
            raise ValueError("completed pool identity mismatch")
        validate_pool(root, plan, identity, terminal)
        return summary
    state_path = root / "pool_state.json"
    state = {"identity": identity, "terminal": terminal}
    if state_path.exists() and _json(state_path) != state:
        raise ValueError("existing pool differs")
    _atomic(state_path, state)
    rows = (
        []
        if not (root / "pool_manifest.jsonl").exists()
        else [
            json.loads(x)
            for x in (root / "pool_manifest.jsonl").read_text().splitlines()
            if x
        ]
    )
    known = {row["sample_id"]: row for row in rows}
    if len(known) != len(rows) or not set(known).issubset(
        {item["sample_id"] for item in plan}
    ):
        raise ValueError("pool manifest has duplicate or foreign sample identities")
    for item in plan:
        prior = known.get(item["sample_id"])
        if prior is not None:
            if (
                prior.get("generation_seed") != item["generation_seed"]
                or prior.get("png_relative_path") != item["png_relative_path"]
            ):
                raise ValueError("pool resume identity mismatch")
            path = root / str(prior["png_relative_path"])
            if not path.is_file() or sha256_file(path) != prior.get("png_sha256"):
                raise ValueError("pool resume PNG hash mismatch")
    if generator is None:
        if not torch.cuda.is_available():
            raise RuntimeError("R13 pool execution requires CUDA")
        text, denoiser, _ = load_h7_terminal_ema(
            terminal_checkpoint, seed, device="cuda"
        )
        generator = _BatchedGenerator(plan, text, denoiser)

    for item in plan:
        if item["sample_id"] in known:
            continue
        array = generator(item)
        if array.shape != (32, 32, 3):
            raise ValueError("R13 generator must return RGB 32x32")
        path = root / item["png_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        Image.fromarray(array.astype(np.uint8), "RGB").save(tmp, format="PNG")
        os.replace(tmp, path)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "split_role": item["split_role"],
                "generation_seed": item["generation_seed"],
                "png_relative_path": item["png_relative_path"],
                "png_sha256": sha256_file(path),
            }
        )
        (root / "pool_manifest.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows),
            encoding="utf-8",
        )
    validate_pool(root, plan, identity, terminal)
    summary = {
        "schema": "tiny_avatar_diffusion.h7_r13_shared_pool_v1",
        "status": "COMPLETE",
        "identity": identity,
        "terminal": terminal,
        "image_count": 804,
    }
    _atomic(complete, summary)
    return summary
