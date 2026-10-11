"""Controlled DR03 10k conditional learning-rate pilot; not main training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_debug_tiny_overfit import (
    ROOT,
    load_train_text_records,
)
from tiny_avatar_diffusion.conditional_training import conditional_diffusion_loss
from tiny_avatar_diffusion.debug_tiny_overfit import BATCH_SIZE, SEED, preload_images
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ID_VAL_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_val.jsonl"
COMP_VAL_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "comp_val.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "dr03_lr_pilot"
LEARNING_RATES = (5e-5, 1e-4, 2e-4)
PHYSICAL_BATCH_SIZE = 32
PILOT_STEPS = 2_000
VALIDATION_STEPS = (0, 250, 500, 1_000, 1_500, 2_000)
SMOOTHING_WINDOW = 100
LOG_INTERVAL = 50
WEIGHT_DECAY = 0.0
TIE_RELATIVE_TOLERANCE = 0.01
VALIDATION_GROWTH_LIMIT = 1.10
MAX_CONSECUTIVE_SCALER_SKIPS = 3


@dataclass(frozen=True)
class LRPilotConfig:
    """Fully frozen controls for the bounded DR03 LR comparison."""

    learning_rates: tuple[float, ...] = LEARNING_RATES
    physical_batch_size: int = PHYSICAL_BATCH_SIZE
    gradient_accumulation: int = 1
    steps: int = PILOT_STEPS
    seed: int = SEED
    validation_steps: tuple[int, ...] = VALIDATION_STEPS
    smoothing_window: int = SMOOTHING_WINDOW
    log_interval: int = LOG_INTERVAL

    def __post_init__(self) -> None:
        if self.learning_rates != LEARNING_RATES:
            raise ValueError(f"learning_rates must be exactly {LEARNING_RATES}")
        if self.physical_batch_size != PHYSICAL_BATCH_SIZE:
            raise ValueError("physical_batch_size is frozen at 32")
        if self.gradient_accumulation != 1:
            raise ValueError("gradient_accumulation is frozen at 1")
        if self.steps != PILOT_STEPS:
            raise ValueError("steps are frozen at 2000")
        if self.validation_steps != VALIDATION_STEPS:
            raise ValueError(f"validation_steps must be exactly {VALIDATION_STEPS}")
        if self.smoothing_window != SMOOTHING_WINDOW:
            raise ValueError("smoothing_window is frozen at 100")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


@dataclass(frozen=True)
class TextImageSplit:
    """Preprocessed real images with aligned materialized DR01 token artifacts."""

    name: str
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class FixedValidationSplit:
    """One split with fixed timesteps/noise shared across all LR candidates."""

    split: TextImageSplit
    timesteps: torch.Tensor
    noise: torch.Tensor


def _load_text_records(path: Path, expected_split: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"{expected_split} text manifest is empty")
    if any(record.get("split_role") != expected_split for record in records):
        raise ValueError(f"{expected_split} manifest contains another split role")
    if any(3 in record["token_ids"] for record in records):
        raise ValueError(f"{expected_split} canonical records contain <unk>")
    return records


def load_text_image_split(
    path: Path, expected_split: str, *, image_root: Path = ROOT
) -> TextImageSplit:
    """Load one existing DR01 text split and reuse the frozen image preprocessing."""
    records = _load_text_records(path, expected_split)
    images = preload_images(records, root=image_root)
    input_ids = torch.tensor([record["token_ids"] for record in records], dtype=torch.long)
    attention_mask = torch.tensor(
        [record["attention_mask"] for record in records], dtype=torch.long
    )
    if images.shape[0] != input_ids.shape[0] or input_ids.shape != attention_mask.shape:
        raise AssertionError("Image and text artifact alignment failed")
    if input_ids.shape[1] != 7:
        raise ValueError("DR01 canonical sequences must have length 7")
    return TextImageSplit(expected_split, images, input_ids, attention_mask)


def build_fixed_validation_split(split: TextImageSplit, *, seed: int) -> FixedValidationSplit:
    """Create deterministic per-example corruption tensors for one validation split."""
    generator = torch.Generator().manual_seed(seed)
    timesteps = torch.randint(0, 1_000, (split.images.shape[0],), generator=generator)
    noise = torch.randn(split.images.shape, generator=generator, dtype=split.images.dtype)
    return FixedValidationSplit(split=split, timesteps=timesteps, noise=noise)


def _batch_from_indices(
    split: TextImageSplit, indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        split.images[indices].to(device),
        split.input_ids[indices].to(device),
        split.attention_mask[indices].to(device),
    )


def evaluate_fixed_validation(
    text_encoder: TextConditioningEncoder,
    denoiser: ConditionalTinyUNet,
    schedule: LinearDDPMSchedule,
    validation: FixedValidationSplit,
    *,
    device: torch.device,
) -> float:
    """Compute deterministic epsilon MSE without optimizer updates."""
    text_was_training = text_encoder.training
    denoiser_was_training = denoiser.training
    text_encoder.eval()
    denoiser.eval()
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for start in range(0, validation.split.images.shape[0], PHYSICAL_BATCH_SIZE):
            end = min(start + PHYSICAL_BATCH_SIZE, validation.split.images.shape[0])
            indices = torch.arange(start, end)
            x_start, input_ids, attention_mask = _batch_from_indices(
                validation.split, indices, device
            )
            result = conditional_diffusion_loss(
                text_encoder,
                denoiser,
                schedule,
                x_start,
                input_ids,
                attention_mask,
                timesteps=validation.timesteps[start:end].to(device),
                noise=validation.noise[start:end].to(device),
            )
            squared_error += (result.predicted_noise - result.noise).square().sum().item()
            elements += result.noise.numel()
    text_encoder.train(text_was_training)
    denoiser.train(denoiser_was_training)
    return squared_error / elements


def _finite_gradients(*modules: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for module in modules
        for parameter in module.parameters()
    )


def _state_dicts_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _run_candidate(
    *,
    learning_rate: float,
    config: LRPilotConfig,
    train: TextImageSplit,
    id_validation: FixedValidationSplit,
    comp_validation: FixedValidationSplit,
    initial_text_state: dict[str, torch.Tensor],
    initial_denoiser_state: dict[str, torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one independent, same-initialization AMP LR candidate."""
    run_id = f"lr_{learning_rate:.0e}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    text_encoder = TextConditioningEncoder()
    denoiser = ConditionalTinyUNet(dropout=0.0)
    text_encoder.load_state_dict(initial_text_state)
    denoiser.load_state_dict(initial_denoiser_state)
    initialization_identical = _state_dicts_equal(text_encoder.state_dict(), initial_text_state) and _state_dicts_equal(
        denoiser.state_dict(), initial_denoiser_state
    )
    text_encoder.to(device)
    denoiser.to(device)
    schedule = LinearDDPMSchedule().to(device)
    optimizer = torch.optim.AdamW(
        [*text_encoder.parameters(), *denoiser.parameters()],
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda")
    initial_scaler_scale = float(scaler.get_scale())
    sampler_generator = torch.Generator().manual_seed(config.seed)
    corruption_generator = torch.Generator(device=device).manual_seed(config.seed)
    permutation = torch.empty(0, dtype=torch.long)
    batch_offset = train.images.shape[0]
    losses: list[float] = []
    train_metrics: list[dict[str, float | int]] = []
    validation_metrics: list[dict[str, float | int | str]] = []
    skipped_steps = 0
    consecutive_skips = 0
    all_losses_finite = True
    all_gradients_finite = True
    status = "completed"
    start_time = time.perf_counter()

    for step in range(0, config.steps + 1):
        if step in config.validation_steps:
            validation_metrics.extend(
                [
                    {
                        "step": step,
                        "split": "ID_VAL",
                        "epsilon_mse": evaluate_fixed_validation(
                            text_encoder, denoiser, schedule, id_validation, device=device
                        ),
                    },
                    {
                        "step": step,
                        "split": "COMP_VAL",
                        "epsilon_mse": evaluate_fixed_validation(
                            text_encoder, denoiser, schedule, comp_validation, device=device
                        ),
                    },
                ]
            )
        if step == config.steps:
            break
        if batch_offset + config.physical_batch_size > train.images.shape[0]:
            permutation = torch.randperm(train.images.shape[0], generator=sampler_generator)
            batch_offset = 0
        indices = permutation[batch_offset : batch_offset + config.physical_batch_size]
        batch_offset += config.physical_batch_size
        x_start, input_ids, attention_mask = _batch_from_indices(train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            result = conditional_diffusion_loss(
                text_encoder,
                denoiser,
                schedule,
                x_start,
                input_ids,
                attention_mask,
                generator=corruption_generator,
            )
        loss_finite = bool(torch.isfinite(result.loss))
        all_losses_finite = all_losses_finite and loss_finite
        if not loss_finite:
            status = "nonfinite_loss"
            break
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        gradients_finite = _finite_gradients(text_encoder, denoiser)
        all_gradients_finite = all_gradients_finite and gradients_finite
        if not gradients_finite:
            status = "nonfinite_gradients"
            break
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        skipped = float(scaler.get_scale()) < scale_before
        skipped_steps += int(skipped)
        consecutive_skips = consecutive_skips + 1 if skipped else 0
        if consecutive_skips >= MAX_CONSECUTIVE_SCALER_SKIPS:
            status = "persistent_scaler_skips"
            break
        loss_value = float(result.loss.detach())
        losses.append(loss_value)
        completed_step = step + 1
        if completed_step % config.log_interval == 0 or completed_step == config.steps:
            train_metrics.append(
                {
                    "step": completed_step,
                    "training_epsilon_mse": loss_value,
                    "smoothed_training_epsilon_mse": statistics.mean(
                        losses[-config.smoothing_window :]
                    ),
                    "grad_scaler_scale": float(scaler.get_scale()),
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
            )

    elapsed_seconds = time.perf_counter() - start_time
    validations = {
        (item["step"], item["split"]): float(item["epsilon_mse"])
        for item in validation_metrics
    }
    initial_smoothed_loss = statistics.mean(losses[: config.smoothing_window]) if len(losses) >= config.smoothing_window else None
    final_smoothed_loss = statistics.mean(losses[-config.smoothing_window :]) if len(losses) >= config.smoothing_window else None
    final_id = validations.get((config.steps, "ID_VAL"))
    final_comp = validations.get((config.steps, "COMP_VAL"))
    initial_id = validations.get((0, "ID_VAL"))
    initial_comp = validations.get((0, "COMP_VAL"))
    eligible = bool(
        status == "completed"
        and all_losses_finite
        and all_gradients_finite
        and skipped_steps == 0
        and initial_smoothed_loss is not None
        and final_smoothed_loss is not None
        and final_smoothed_loss < initial_smoothed_loss
        and final_id is not None
        and initial_id is not None
        and final_id <= VALIDATION_GROWTH_LIMIT * initial_id
        and final_comp is not None
        and initial_comp is not None
        and final_comp <= VALIDATION_GROWTH_LIMIT * initial_comp
    )
    score = (final_id + final_comp) / 2.0 if eligible and final_id is not None and final_comp is not None else None
    summary = {
        "run_id": run_id,
        "learning_rate": learning_rate,
        "seed": config.seed,
        "dataset": "10k",
        "train_split": "TRAIN",
        "train_size": train.images.shape[0],
        "physical_batch_size": config.physical_batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "precision": "AMP_FP16",
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "steps_requested": config.steps,
        "steps_completed": len(losses),
        "images_seen": len(losses) * config.physical_batch_size,
        "effective_epochs": len(losses) * config.physical_batch_size / train.images.shape[0],
        "wall_clock_seconds": elapsed_seconds,
        "initialization_identical": initialization_identical,
        "initial_grad_scaler_scale": initial_scaler_scale,
        "final_grad_scaler_scale": float(scaler.get_scale()),
        "skipped_scaler_steps": skipped_steps,
        "all_losses_finite": all_losses_finite,
        "all_gradients_finite": all_gradients_finite,
        "initial_smoothed_train_loss": initial_smoothed_loss,
        "final_smoothed_train_loss": final_smoothed_loss,
        "final_id_val_mse": final_id,
        "final_comp_val_mse": final_comp,
        "best_id_val_mse": min(
            (item["epsilon_mse"] for item in validation_metrics if item["split"] == "ID_VAL"),
            default=None,
        ),
        "best_comp_val_mse": min(
            (item["epsilon_mse"] for item in validation_metrics if item["split"] == "COMP_VAL"),
            default=None,
        ),
        "eligible": eligible,
        "selection_score": score,
        "status": status,
    }
    with (run_dir / "train_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "step", "training_epsilon_mse", "smoothed_training_epsilon_mse", "grad_scaler_scale", "elapsed_seconds"
        ))
        writer.writeheader()
        writer.writerows(train_metrics)
    with (run_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "split", "epsilon_mse"))
        writer.writeheader()
        writer.writerows(validation_metrics)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "step": len(losses),
            "learning_rate": learning_rate,
            "text_encoder_state_dict": text_encoder.state_dict(),
            "conditional_unet_state_dict": denoiser.state_dict(),
        },
        run_dir / "final_state.pt",
    )
    return summary


def select_learning_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the user-frozen eligibility, final-score, and 1% tie-break rule."""
    eligible = [result for result in results if result["eligible"]]
    if not eligible:
        return {"selected_learning_rate": None, "strength": "INCONCLUSIVE", "tied_run_ids": []}
    best_score = min(float(result["selection_score"]) for result in eligible)
    tied = [
        result
        for result in eligible
        if abs(float(result["selection_score"]) - best_score) / best_score <= TIE_RELATIVE_TOLERANCE
    ]
    selected = min(tied, key=lambda result: float(result["learning_rate"]))
    return {
        "selected_learning_rate": selected["learning_rate"],
        "selected_run_id": selected["run_id"],
        "best_score": best_score,
        "tied_run_ids": [result["run_id"] for result in tied],
        "strength": "WEAK" if len(tied) > 1 else "CLEAR",
    }


def _write_overview(output_dir: Path, config: LRPilotConfig, results: list[dict[str, Any],], selection: dict[str, Any]) -> None:
    """Write concise aggregate JSON and Markdown records for the completed pilot."""
    overview = {
        "pilot_type": "DR03 CONDITIONAL LR PILOT ONLY",
        "frozen_config": asdict(config),
        "selection_rule": {
            "eligibility": "finite losses/gradients; zero skips; final smoothed train loss < initial 100-step mean; final ID_VAL and COMP_VAL <= 1.10 * step-0 MSE",
            "score": "mean(final ID_VAL MSE, final COMP_VAL MSE)",
            "tie_rule": "within 1% of best score selects lower learning rate",
        },
        "results": results,
        "selection": selection,
    }
    (output_dir / "summary.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DR03 Conditional Learning-Rate Pilot",
        "",
        "Engineering/optimization pilot only; it is not a final generative-quality evaluation.",
        "",
        "| LR | Eligible | Score | ID_VAL @ 2000 | COMP_VAL @ 2000 | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['learning_rate']:.0e} | {result['eligible']} | {result['selection_score']} | "
            f"{result['final_id_val_mse']} | {result['final_comp_val_mse']} | {result['status']} |"
        )
    lines.extend(["", f"Selected learning rate: `{selection['selected_learning_rate']}`."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_lr_pilot(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_index: int = 0,
    config: LRPilotConfig | None = None,
) -> dict[str, Any]:
    """Run the exact three-point DR03 LR pilot on real 10k TRAIN data."""
    if config is None:
        config = LRPilotConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("DR03 LR pilot requires CUDA")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("device_index is not an available CUDA device")
    device = torch.device(f"cuda:{device_index}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_text_image_split(
        ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl", "TRAIN"
    )
    id_validation = build_fixed_validation_split(
        load_text_image_split(ID_VAL_MANIFEST, "ID_VAL"), seed=config.seed + 1
    )
    comp_validation = build_fixed_validation_split(
        load_text_image_split(COMP_VAL_MANIFEST, "COMP_VAL"), seed=config.seed + 2
    )
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    initial_text_state = copy.deepcopy(TextConditioningEncoder().state_dict())
    initial_denoiser_state = copy.deepcopy(ConditionalTinyUNet(dropout=0.0).state_dict())
    results = [
        _run_candidate(
            learning_rate=learning_rate,
            config=config,
            train=train,
            id_validation=id_validation,
            comp_validation=comp_validation,
            initial_text_state=initial_text_state,
            initial_denoiser_state=initial_denoiser_state,
            device=device,
            output_dir=output_dir,
        )
        for learning_rate in config.learning_rates
    ]
    selection = select_learning_rate(results)
    _write_overview(output_dir, config, results, selection)
    return {"results": results, "selection": selection}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_lr_pilot(output_dir=args.output_dir, device_index=args.device_index)


if __name__ == "__main__":
    main()
