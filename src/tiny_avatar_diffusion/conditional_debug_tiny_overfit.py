"""DEBUG-ONLY tiny joint conditional-overfit gate for DR03."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tiny_avatar_diffusion.conditional_training import (
    conditional_diffusion_loss,
    conditional_training_step,
)
from tiny_avatar_diffusion.debug_tiny_overfit import (
    BATCH_SIZE,
    PROBE_TIMESTEPS,
    SEED,
    SUBSET_SIZE,
    TRAIN_MANIFEST,
    build_fixed_probe,
    load_train_records,
    preload_images,
    select_tiny_subset,
)
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ROOT = Path(__file__).resolve().parents[2]
TEXT_TRAIN_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "dr03_conditional_tiny_overfit_debug"
DEBUG_LR = 1e-4
DEBUG_WEIGHT_DECAY = 0.0


@dataclass(frozen=True)
class ConditionalFixedProbe:
    """DR02 fixed image/noise probe extended with aligned DR01 token IDs."""

    x_start: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def load_train_text_records(path: Path = TEXT_TRAIN_MANIFEST) -> list[dict[str, Any]]:
    """Load only materialized canonical DR01 10k TRAIN text records."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("DR01 text TRAIN manifest is empty")
    if any(record.get("split_role") != "TRAIN" for record in records):
        raise ValueError("DR01 text manifest contains a non-TRAIN record")
    return records


def select_text_conditioning(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize validated DR01 token IDs and masks without rebuilding vocabulary."""
    if not records:
        raise ValueError("No selected DR01 records")
    by_image_id = {str(record["image_id"]): record for record in load_train_text_records()}
    try:
        selected_text = [by_image_id[str(record["image_id"])] for record in records]
    except KeyError as error:
        raise ValueError("Selected TRAIN image is missing from DR01 text TRAIN artifacts") from error
    input_ids = torch.tensor([record["token_ids"] for record in selected_text], dtype=torch.long)
    attention_mask = torch.tensor(
        [record["attention_mask"] for record in selected_text], dtype=torch.long
    )
    if input_ids.shape[1] != 7 or attention_mask.shape != input_ids.shape:
        raise ValueError("DR01 text artifacts must provide [B, 7] token IDs and masks")
    if torch.any(input_ids == 3):
        raise ValueError("Canonical DR01 TRAIN captions must not contain <unk>")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("DR01 attention masks must use 1=valid and 0=padding")
    return input_ids, attention_mask


def build_conditional_fixed_probe(
    images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> ConditionalFixedProbe:
    """Reuse the DR02 fixed probe and repeat aligned DR01 text conditioning."""
    probe = build_fixed_probe(images)
    repetitions = len(PROBE_TIMESTEPS)
    return ConditionalFixedProbe(
        x_start=probe.x_start,
        timesteps=probe.timesteps,
        noise=probe.noise,
        input_ids=input_ids.repeat_interleave(repetitions, dim=0),
        attention_mask=attention_mask.repeat_interleave(repetitions, dim=0),
    )


def create_conditional_debug_optimizer(
    text_encoder: TextConditioningEncoder, denoiser: ConditionalTinyUNet
) -> torch.optim.AdamW:
    """Reuse the DR02 debug-only AdamW policy for both trainable components."""
    return torch.optim.AdamW(
        [*text_encoder.parameters(), *denoiser.parameters()],
        lr=DEBUG_LR,
        weight_decay=DEBUG_WEIGHT_DECAY,
    )


def evaluate_conditional_fixed_probe(
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    schedule: LinearDDPMSchedule,
    probe: ConditionalFixedProbe,
    *,
    device: torch.device,
) -> float:
    """Return deterministic fixed-probe epsilon MSE without changing parameters."""
    text_was_training = text_encoder.training
    denoiser_was_training = denoiser.training
    text_encoder.eval()
    denoiser.eval()
    total_squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for start in range(0, probe.x_start.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            result = conditional_diffusion_loss(
                text_encoder,
                denoiser,
                schedule,
                probe.x_start[start:end].to(device),
                probe.input_ids[start:end].to(device),
                probe.attention_mask[start:end].to(device),
                timesteps=probe.timesteps[start:end].to(device),
                noise=probe.noise[start:end].to(device),
            )
            total_squared_error += (result.predicted_noise - result.noise).square().sum().item()
            element_count += result.noise.numel()
    text_encoder.train(text_was_training)
    denoiser.train(denoiser_was_training)
    return total_squared_error / element_count


def run_conditional_debug_tiny_overfit(
    *,
    device_name: str,
    max_steps: int = 2_000,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the DR03 DEBUG-only joint conditional tiny-overfit gate."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    selected = select_tiny_subset(load_train_records())
    images = preload_images(selected)
    input_ids, attention_mask = select_text_conditioning(selected)
    probe = build_conditional_fixed_probe(images, input_ids, attention_mask)
    text_encoder = TextConditioningEncoder().to(device)
    denoiser = ConditionalTinyUNet(dropout=0.0).to(device)
    schedule = LinearDDPMSchedule().to(device)
    optimizer = create_conditional_debug_optimizer(text_encoder, denoiser)
    corruption_generator = torch.Generator(device=device.type).manual_seed(SEED)
    batch_generator = torch.Generator().manual_seed(SEED)

    initial_probe_loss = evaluate_conditional_fixed_probe(
        text_encoder, denoiser, schedule, probe, device=device
    )
    metrics: list[dict[str, float | int | None]] = [
        {"step": 0, "training_loss": None, "probe_epsilon_mse": initial_probe_loss}
    ]
    permutation = torch.empty(0, dtype=torch.long)
    batch_offset = SUBSET_SIZE
    training_loss = 0.0
    for step in range(1, max_steps + 1):
        if batch_offset >= SUBSET_SIZE:
            permutation = torch.randperm(SUBSET_SIZE, generator=batch_generator)
            batch_offset = 0
        indices = permutation[batch_offset : batch_offset + BATCH_SIZE]
        batch_offset += BATCH_SIZE
        result = conditional_training_step(
            text_encoder,
            denoiser,
            schedule,
            optimizer,
            images[indices].to(device),
            input_ids[indices].to(device),
            attention_mask[indices].to(device),
            generator=corruption_generator,
        )
        training_loss = float(result.loss.detach())
        if step % 100 == 0 or step == max_steps:
            probe_loss = evaluate_conditional_fixed_probe(
                text_encoder, denoiser, schedule, probe, device=device
            )
            metrics.append(
                {"step": step, "training_loss": training_loss, "probe_epsilon_mse": probe_loss}
            )

    final_probe_loss = float(metrics[-1]["probe_epsilon_mse"])
    summary: dict[str, Any] = {
        "debug_only": True,
        "dataset": "10k",
        "split": "TRAIN",
        "manifest_path": TRAIN_MANIFEST.relative_to(ROOT).as_posix(),
        "text_manifest_path": TEXT_TRAIN_MANIFEST.relative_to(ROOT).as_posix(),
        "subset_size": SUBSET_SIZE,
        "subset_seed": SEED,
        "selected_image_ids": [record["image_id"] for record in selected],
        "optimizer": "AdamW",
        "lr": DEBUG_LR,
        "weight_decay": DEBUG_WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_steps": max_steps,
        "optimization_steps_run": max_steps,
        "T": 1_000,
        "schedule": "linear",
        "initial_probe_epsilon_mse": initial_probe_loss,
        "final_probe_epsilon_mse": final_probe_loss,
        "probe_loss_decreased": final_probe_loss < initial_probe_loss,
        "all_finite": all(
            value is None or torch.isfinite(torch.tensor(value))
            for row in metrics
            for value in row.values()
            if isinstance(value, float)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    if not summary["all_finite"]:
        raise FloatingPointError("Conditional debug gate produced non-finite metrics")
    if not summary["probe_loss_decreased"]:
        raise RuntimeError("Conditional debug gate did not reduce the fixed probe loss")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_conditional_debug_tiny_overfit(
        device_name=args.device, max_steps=args.max_steps, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
