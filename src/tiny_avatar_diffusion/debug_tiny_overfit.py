"""DEBUG-ONLY tiny unconditional overfit gate for DR02 Phase 7."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.unet import TinyUNet

ROOT = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = ROOT / "data" / "processed" / "dr01_v1" / "10k" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "dr02_tiny_overfit_debug"
SUBSET_SIZE = 16
SEED = 2026
BATCH_SIZE = 8
PROBE_TIMESTEPS = (0, 250, 500, 999)


def parse_source_image_path(source_image_path: str) -> tuple[Path, str]:
    """Split a DR01 ``archive::member`` image reference."""
    parts = source_image_path.split("::", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("source_image_path must have format archive::member")
    return Path(parts[0]), parts[1]


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the frozen DR01 preprocessing contract using Pillow as backend."""
    foreground = image.convert("RGBA")
    background = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, foreground).convert("RGB")
    resized = composited.resize((32, 32), resample=Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    tensor = pixels.reshape(32, 32, 3).permute(2, 0, 1).to(torch.float32)
    return tensor.div_(255.0).mul_(2.0).sub_(1.0)


def decode_archive_member(archive_path: Path, member_path: str) -> torch.Tensor:
    """Decode and preprocess one image without extracting it to disk."""
    with tarfile.open(archive_path, "r:gz") as archive:
        stream = archive.extractfile(member_path)
        if stream is None:
            raise FileNotFoundError(f"Archive member not found: {member_path}")
        with Image.open(stream) as image:
            image.load()
            return preprocess_image(image)


def load_train_records(manifest_path: Path = TRAIN_MANIFEST) -> list[dict[str, Any]]:
    """Load and validate only the canonical DR01 TRAIN manifest."""
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        raise ValueError("TRAIN manifest is empty")
    if any(record.get("split_role") != "TRAIN" for record in records):
        raise ValueError("TRAIN manifest contains a non-TRAIN record")
    return records


def select_tiny_subset(
    records: list[dict[str, Any]],
    *,
    subset_size: int = SUBSET_SIZE,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Select a deterministic unique subset using a dedicated generator."""
    if len(records) < subset_size:
        raise ValueError("Not enough TRAIN records for the tiny subset")
    image_ids = [str(record["image_id"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("TRAIN records must have unique image_id values")
    if any(record.get("split_role") != "TRAIN" for record in records):
        raise ValueError("Tiny subset selection accepts TRAIN records only")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator)[:subset_size].tolist()
    return [records[index] for index in indices]


def preload_images(
    records: list[dict[str, Any]], *, root: Path = ROOT
) -> torch.Tensor:
    """Decode the tiny subset directly from its archives into memory."""
    images: list[torch.Tensor] = []
    with ExitStack() as stack:
        archives: dict[Path, tarfile.TarFile] = {}
        for record in records:
            archive_reference, member_path = parse_source_image_path(
                str(record["source_image_path"])
            )
            archive_path = (
                archive_reference
                if archive_reference.is_absolute()
                else root / archive_reference
            )
            if archive_path not in archives:
                archives[archive_path] = stack.enter_context(
                    tarfile.open(archive_path, "r:gz")
                )
            stream = archives[archive_path].extractfile(member_path)
            if stream is None:
                raise FileNotFoundError(f"Archive member not found: {member_path}")
            with Image.open(stream) as image:
                image.load()
                images.append(preprocess_image(image))
    return torch.stack(images)


@dataclass(frozen=True)
class FixedProbe:
    """Deterministic diagnostic inputs reused throughout one debug run."""

    x_start: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor


def build_fixed_probe(images: torch.Tensor, *, seed: int = SEED) -> FixedProbe:
    """Build all image/timestep pairs and their fixed Gaussian noise."""
    if images.shape != (SUBSET_SIZE, 3, 32, 32):
        raise ValueError("images must have shape [16, 3, 32, 32]")
    x_start = images.repeat_interleave(len(PROBE_TIMESTEPS), dim=0)
    timesteps = torch.tensor(PROBE_TIMESTEPS, dtype=torch.long).repeat(SUBSET_SIZE)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(x_start.shape, generator=generator, dtype=torch.float32)
    return FixedProbe(x_start=x_start, timesteps=timesteps, noise=noise)


def evaluate_fixed_probe(
    model: nn.Module,
    schedule: LinearDDPMSchedule,
    probe: FixedProbe,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fixed epsilon and reconstructed-x0 MSE without changing model state."""
    was_training = model.training
    model.eval()
    epsilon_squared_error = 0.0
    x0_squared_error = 0.0
    element_count = 0
    x0_squared_error_by_timestep = {
        timestep: 0.0 for timestep in PROBE_TIMESTEPS
    }
    element_count_by_timestep = {
        timestep: 0 for timestep in PROBE_TIMESTEPS
    }
    with torch.no_grad():
        for start in range(0, probe.x_start.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            x_start = probe.x_start[start:end].to(device)
            timesteps = probe.timesteps[start:end].to(device)
            noise = probe.noise[start:end].to(device)
            x_t = schedule.q_sample(x_start, timesteps, noise=noise)
            predicted_noise = model(x_t, timesteps)
            predicted_x_start = schedule.predict_x_start_from_noise(
                x_t, timesteps, predicted_noise
            )
            epsilon_squared_error += (predicted_noise - noise).square().sum().item()
            x0_error = (predicted_x_start - x_start).square()
            x0_squared_error += x0_error.sum().item()
            element_count += noise.numel()
            for timestep in PROBE_TIMESTEPS:
                timestep_mask = timesteps == timestep
                x0_squared_error_by_timestep[timestep] += x0_error[
                    timestep_mask
                ].sum().item()
                element_count_by_timestep[timestep] += x0_error[
                    timestep_mask
                ].numel()
    model.train(was_training)
    return {
        "epsilon_mse": epsilon_squared_error / element_count,
        "x0_mse": x0_squared_error / element_count,
        "x0_mse_by_timestep": {
            timestep: x0_squared_error_by_timestep[timestep]
            / element_count_by_timestep[timestep]
            for timestep in PROBE_TIMESTEPS
        },
    }


def create_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Create the frozen DEBUG-only optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)


def training_step(
    model: nn.Module,
    schedule: LinearDDPMSchedule,
    optimizer: torch.optim.Optimizer,
    x_start: torch.Tensor,
    *,
    generator: torch.Generator,
) -> float:
    """Execute one real epsilon-prediction optimizer step."""
    model.train()
    timesteps = torch.randint(
        0,
        schedule.betas.shape[0],
        (x_start.shape[0],),
        device=x_start.device,
        generator=generator,
    )
    noise = torch.randn(
        x_start.shape,
        device=x_start.device,
        dtype=x_start.dtype,
        generator=generator,
    )
    x_t = schedule.q_sample(x_start, timesteps, noise=noise)
    optimizer.zero_grad(set_to_none=True)
    predicted_noise = model(x_t, timesteps)
    if not torch.isfinite(predicted_noise).all():
        raise FloatingPointError("Training prediction contains NaN or Inf")
    loss = F.mse_loss(predicted_noise, noise)
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss contains NaN or Inf")
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Training gradients contain NaN or Inf")
    if not any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
        raise RuntimeError("All training gradients are zero")
    optimizer.step()
    return float(loss.detach())


def sample_fixed_preview(
    model: nn.Module,
    schedule: LinearDDPMSchedule,
    *,
    device: torch.device,
    seed: int = SEED,
) -> torch.Tensor:
    """Run the validated T=1000 sampler with locally fixed RNG state."""
    if schedule.betas.shape[0] != 1_000:
        raise ValueError("Sampling preview requires the real T=1000 schedule")
    was_training = model.training
    model.eval()
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        x_T = torch.randn(1, 3, 32, 32, device=device)
        sample = schedule.sample_loop_from_noise_predictor(x_T, model)
    model.train(was_training)
    return sample


def save_preview_png(sample: torch.Tensor, path: Path) -> None:
    """Clamp only for display conversion and save one RGB preview."""
    display = (
        sample[0]
        .detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
    )
    image = Image.frombytes("RGB", (32, 32), bytes(display.reshape(-1).tolist()))
    image.save(path)


def run_debug_tiny_overfit(
    *,
    device_name: str,
    max_steps: int = 2_000,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sampling_preview: bool = False,
) -> dict[str, Any]:
    """Run the deliberately small DR02 DEBUG-only correctness experiment."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    records = load_train_records()
    selected = select_tiny_subset(records)
    images = preload_images(selected)
    probe = build_fixed_probe(images)

    model = TinyUNet(dropout=0.0).to(device)
    schedule = LinearDDPMSchedule().to(device)
    optimizer = create_optimizer(model)
    corruption_generator = torch.Generator(device=device.type).manual_seed(SEED)
    batch_generator = torch.Generator().manual_seed(SEED)

    output_dir.mkdir(parents=True, exist_ok=True)
    untrained_preview = None
    if sampling_preview:
        untrained_preview = sample_fixed_preview(model, schedule, device=device)

    initial_probe = evaluate_fixed_probe(model, schedule, probe, device=device)
    metrics: list[dict[str, float | int | None]] = [
        {
            "step": 0,
            "training_loss": None,
            "probe_epsilon_mse": initial_probe["epsilon_mse"],
            "probe_x0_mse": initial_probe["x0_mse"],
            **{
                f"probe_x0_mse_t{timestep}": initial_probe["x0_mse_by_timestep"][
                    timestep
                ]
                for timestep in PROBE_TIMESTEPS
            },
        }
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
        x_start = images[indices].to(device)
        training_loss = training_step(
            model,
            schedule,
            optimizer,
            x_start,
            generator=corruption_generator,
        )
        if step % 100 == 0 or step == max_steps:
            probe_metrics = evaluate_fixed_probe(model, schedule, probe, device=device)
            metrics.append(
                {
                    "step": step,
                    "training_loss": training_loss,
                    "probe_epsilon_mse": probe_metrics["epsilon_mse"],
                    "probe_x0_mse": probe_metrics["x0_mse"],
                    **{
                        f"probe_x0_mse_t{timestep}": probe_metrics[
                            "x0_mse_by_timestep"
                        ][timestep]
                        for timestep in PROBE_TIMESTEPS
                    },
                }
            )

    final_probe = {
        "epsilon_mse": float(metrics[-1]["probe_epsilon_mse"]),
        "x0_mse": float(metrics[-1]["probe_x0_mse"]),
    }
    archive_reference, _ = parse_source_image_path(str(selected[0]["source_image_path"]))
    summary: dict[str, Any] = {
        "debug_only": True,
        "pillow_version": PIL.__version__,
        "preprocessing_backend": "Pillow",
        "dataset": "10k",
        "split": "TRAIN",
        "manifest_path": TRAIN_MANIFEST.relative_to(ROOT).as_posix(),
        "archive_path": archive_reference.as_posix(),
        "subset_size": SUBSET_SIZE,
        "subset_seed": SEED,
        "selected_image_ids": [record["image_id"] for record in selected],
        "selected_source_paths": [record["source_image_path"] for record in selected],
        "resolution": 32,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dropout": 0.0,
        "T": 1_000,
        "schedule": "linear",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "max_steps": max_steps,
        "probe_timesteps": list(PROBE_TIMESTEPS),
        "device": str(device),
        "torch_version": torch.__version__,
        "initial_probe_epsilon_mse": initial_probe["epsilon_mse"],
        "final_probe_epsilon_mse": final_probe["epsilon_mse"],
        "initial_probe_x0_mse": initial_probe["x0_mse"],
        "final_probe_x0_mse": final_probe["x0_mse"],
        "initial_probe_x0_mse_by_timestep": {
            str(timestep): initial_probe["x0_mse_by_timestep"][timestep]
            for timestep in PROBE_TIMESTEPS
        },
        "final_probe_x0_mse_by_timestep": {
            str(timestep): float(metrics[-1][f"probe_x0_mse_t{timestep}"])
            for timestep in PROBE_TIMESTEPS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    if sampling_preview:
        assert untrained_preview is not None
        trained_preview = sample_fixed_preview(model, schedule, device=device)
        save_preview_png(untrained_preview, output_dir / "sample_untrained.png")
        save_preview_png(trained_preview, output_dir / "sample_trained.png")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, choices=("cpu", "cuda"))
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sampling-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the DEBUG-only gate."""
    args = _parse_args()
    run_debug_tiny_overfit(
        device_name=args.device,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        sampling_preview=args.sampling_preview,
    )


if __name__ == "__main__":
    main()
