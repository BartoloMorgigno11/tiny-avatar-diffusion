import csv
import json

import pytest
import torch

from tiny_avatar_diffusion.profile_gpu import (
    CONDITIONAL_BATCH_SIZES,
    CONDITIONAL_MEASURED_STEPS,
    ConditionalProfileConfig,
    _conditional_profile_batch,
    _write_conditional_profile_artifacts,
    validate_conditional_profile_config,
)


def test_conditional_profile_defaults_and_validation() -> None:
    config = ConditionalProfileConfig()

    assert config.batch_sizes == CONDITIONAL_BATCH_SIZES == (4, 8, 16, 32)
    assert config.warmup_steps == 5
    assert config.measured_steps == CONDITIONAL_MEASURED_STEPS == 10
    with pytest.raises(ValueError, match="increasing"):
        validate_conditional_profile_config(ConditionalProfileConfig(batch_sizes=(8, 4)))
    with pytest.raises(ValueError, match="measured_steps"):
        validate_conditional_profile_config(ConditionalProfileConfig(measured_steps=0))


def test_conditional_profile_batch_cycles_real_subset_tensors_without_changing_contract() -> None:
    images = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    input_ids = torch.tensor([[1, 4, 2, 0, 0, 0, 0], [1, 5, 2, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])

    batch_images, batch_ids, batch_mask = _conditional_profile_batch(
        images, input_ids, mask, 4, device=torch.device("cpu")
    )

    assert batch_images.shape == (4, 3, 32, 32)
    assert batch_ids.tolist() == [input_ids[0].tolist(), input_ids[1].tolist()] * 2
    assert torch.equal(batch_mask, mask.repeat(2, 1))


def test_conditional_profile_artifact_serialization(tmp_path) -> None:
    row = {
        "precision": "FP32",
        "batch_size": 4,
        "status": "success",
        "warmup_steps": 5,
        "measured_steps": 10,
        "mean_step_ms": 1.0,
        "median_step_ms": 1.0,
        "std_step_ms": 0.0,
        "images_per_second": 4_000.0,
        "peak_allocated_mb": 100.0,
        "peak_reserved_mb": 128.0,
        "total_vram_mb": 8_000.0,
        "allocated_fraction": 0.0125,
        "reserved_fraction": 0.016,
        "finite_loss": True,
        "finite_gradients": True,
        "grad_scaler_initial_scale": None,
        "grad_scaler_final_scale": None,
        "warmup_skipped_steps": 0,
        "measured_skipped_steps": 0,
    }
    summary = {"profile_type": "DR03 CONDITIONAL FULL-TRAINING ENGINEERING PILOT ONLY"}
    parity = {"amp_loss_finite": True}

    _write_conditional_profile_artifacts(tmp_path, summary, [row], [], parity)

    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert json.loads((tmp_path / "amp_parity.json").read_text()) == parity
    with (tmp_path / "fp32_results.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["batch_size"] == "4"
