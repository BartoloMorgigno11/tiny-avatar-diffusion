from __future__ import annotations

import inspect
import io
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from tiny_avatar_diffusion.debug_tiny_overfit import (
    PROBE_TIMESTEPS,
    build_fixed_probe,
    create_optimizer,
    decode_archive_member,
    evaluate_fixed_probe,
    parse_source_image_path,
    preprocess_image,
    run_debug_tiny_overfit,
    select_tiny_subset,
    training_step,
)
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule


def _records(count: int = 32) -> list[dict[str, object]]:
    return [
        {
            "image_id": f"image-{index}",
            "source_image_path": f"archive.tgz::images/image-{index}.png",
            "split_role": "TRAIN",
        }
        for index in range(count)
    ]


def test_parse_and_decode_archive_member_with_pillow(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.tgz"
    image_bytes = io.BytesIO()
    Image.new("RGBA", (5, 7), (10, 20, 30, 255)).save(image_bytes, format="PNG")
    payload = image_bytes.getvalue()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("images/example.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    parsed_archive, member = parse_source_image_path(
        f"{archive_path}::images/example.png"
    )
    tensor = decode_archive_member(parsed_archive, member)

    assert parsed_archive == archive_path
    assert member == "images/example.png"
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_preprocessing_composites_alpha_on_white_and_is_deterministic() -> None:
    transparent = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    partial = Image.new("RGBA", (4, 4), (0, 0, 0, 128))

    transparent_tensor = preprocess_image(transparent)
    partial_tensor = preprocess_image(partial)
    repeated = preprocess_image(partial)

    assert transparent_tensor.shape == (3, 32, 32)
    assert transparent_tensor.dtype == torch.float32
    assert torch.equal(transparent_tensor, torch.ones_like(transparent_tensor))
    expected_partial = torch.tensor(127 / 255 * 2 - 1, dtype=torch.float32)
    assert torch.allclose(partial_tensor, torch.full_like(partial_tensor, expected_partial))
    assert torch.equal(partial_tensor, repeated)
    assert torch.isfinite(partial_tensor).all()
    assert partial_tensor.min() >= -1.0
    assert partial_tensor.max() <= 1.0


def test_preprocessing_uses_bilinear_resize() -> None:
    image = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
    image.putpixel((1, 0), (255, 255, 255, 255))

    tensor = preprocess_image(image)

    assert torch.unique(tensor[0]).numel() > 2


def test_tiny_subset_is_deterministic_unique_and_train_only() -> None:
    records = _records()

    first = select_tiny_subset(records)
    second = select_tiny_subset(records)

    assert [record["image_id"] for record in first] == [
        record["image_id"] for record in second
    ]
    assert len(first) == 16
    assert len({record["image_id"] for record in first}) == 16
    assert all(record["split_role"] == "TRAIN" for record in first)
    assert first != records[:16]
    invalid = records.copy()
    invalid[0] = {**invalid[0], "split_role": "ID_VAL"}
    with pytest.raises(ValueError, match="TRAIN"):
        select_tiny_subset(invalid)


def test_fixed_probe_has_frozen_count_timesteps_and_noise() -> None:
    images = torch.zeros(16, 3, 32, 32)

    first = build_fixed_probe(images)
    second = build_fixed_probe(images)

    assert first.x_start.shape == (64, 3, 32, 32)
    assert first.timesteps.shape == (64,)
    assert set(first.timesteps.tolist()) == set(PROBE_TIMESTEPS)
    assert torch.equal(first.noise, second.noise)


class _ProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.observed_training: list[bool] = []
        self.observed_grad_enabled: list[bool] = []

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        self.observed_training.append(self.training)
        self.observed_grad_enabled.append(torch.is_grad_enabled())
        return self.conv(x)


class _TimestepDependentProbeModel(nn.Module):
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.to(dtype=x.dtype).reshape(-1, 1, 1, 1).expand_as(x)


def test_training_step_updates_parameter_with_finite_loss() -> None:
    model = _ProbeModel()
    schedule = LinearDDPMSchedule()
    optimizer = create_optimizer(model)
    generator = torch.Generator().manual_seed(2026)
    x_start = torch.zeros(8, 3, 32, 32)
    weight_before = model.conv.weight.detach().clone()

    loss = training_step(
        model, schedule, optimizer, x_start, generator=generator
    )

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.conv.weight, weight_before)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_probe_is_finite_deterministic_and_does_not_change_parameters() -> None:
    model = _ProbeModel()
    model.train()
    schedule = LinearDDPMSchedule()
    probe = build_fixed_probe(torch.zeros(16, 3, 32, 32))
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    first = evaluate_fixed_probe(model, schedule, probe, device=torch.device("cpu"))
    second = evaluate_fixed_probe(model, schedule, probe, device=torch.device("cpu"))

    assert first == second
    assert torch.isfinite(torch.tensor(first["epsilon_mse"]))
    assert torch.isfinite(torch.tensor(first["x0_mse"]))
    assert model.training
    assert model.observed_training and not any(model.observed_training)
    assert model.observed_grad_enabled and not any(model.observed_grad_enabled)
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(model.parameters(), parameters_before, strict=True)
    )


def test_probe_reports_x0_mse_for_each_fixed_timestep() -> None:
    model = _TimestepDependentProbeModel()
    schedule = LinearDDPMSchedule()
    probe = build_fixed_probe(torch.zeros(16, 3, 32, 32))
    noise_before = probe.noise.clone()
    timesteps_before = probe.timesteps.clone()

    metrics = evaluate_fixed_probe(model, schedule, probe, device=torch.device("cpu"))

    x_t = schedule.q_sample(probe.x_start, probe.timesteps, noise=probe.noise)
    predicted_noise = model(x_t, probe.timesteps)
    predicted_x_start = schedule.predict_x_start_from_noise(
        x_t, probe.timesteps, predicted_noise
    )
    x0_error = (predicted_x_start - probe.x_start).square()
    expected_by_timestep = {
        timestep: x0_error[probe.timesteps == timestep].mean().item()
        for timestep in PROBE_TIMESTEPS
    }

    assert set(metrics["x0_mse_by_timestep"]) == set(PROBE_TIMESTEPS)
    assert all(torch.isfinite(torch.tensor(value)) for value in expected_by_timestep.values())
    assert all(
        metrics["x0_mse_by_timestep"][timestep]
        == pytest.approx(expected_by_timestep[timestep])
        for timestep in PROBE_TIMESTEPS
    )
    assert metrics["x0_mse"] == pytest.approx(x0_error.mean().item())
    assert len(set(metrics["x0_mse_by_timestep"].values())) == len(PROBE_TIMESTEPS)
    assert torch.equal(probe.noise, noise_before)
    assert torch.equal(probe.timesteps, timesteps_before)


def test_sampling_preview_is_disabled_by_default_and_no_forbidden_features() -> None:
    signature = inspect.signature(run_debug_tiny_overfit)
    source = inspect.getsource(inspect.getmodule(run_debug_tiny_overfit))

    assert signature.parameters["sampling_preview"].default is False
    assert "autocast" not in source
    assert "GradScaler" not in source
    assert "clip_grad" not in source
    assert "lr_scheduler" not in source
    assert "EMA" not in source
