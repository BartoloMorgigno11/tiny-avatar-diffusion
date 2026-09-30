from __future__ import annotations

import csv
import json

import pytest

from tiny_avatar_diffusion.profile_gpu import (
    AMP_MEASUREMENT_FIELDS,
    MEASUREMENT_FIELDS,
    OPTIONAL_BATCH_MEMORY_FRACTION,
    PRECISIONS,
    ProfileConfig,
    backend_precision_metadata,
    compare_precision_results,
    images_per_second,
    should_try_optional_batch_128,
    validate_config,
    validate_precision,
    write_amp_pilot_artifacts,
    write_profile_artifacts,
)


def test_profile_config_rejects_invalid_step_counts() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        validate_config(ProfileConfig(warmup_steps=-1))
    with pytest.raises(ValueError, match="measured_steps"):
        validate_config(ProfileConfig(measured_steps=0))


def test_images_per_second_uses_median_step_latency() -> None:
    assert images_per_second(32, 4.0) == pytest.approx(8_000.0)
    with pytest.raises(ValueError):
        images_per_second(0, 4.0)
    with pytest.raises(ValueError):
        images_per_second(32, 0.0)


def test_optional_batch_policy_requires_success_and_low_memory() -> None:
    total_vram_mb = 8_000.0
    success = {"status": "success", "peak_allocated_mb": 0.59 * total_vram_mb}
    high_memory = {"status": "success", "peak_allocated_mb": 0.60 * total_vram_mb}
    oom = {"status": "OOM", "peak_allocated_mb": None}

    assert should_try_optional_batch_128(success, total_vram_mb=total_vram_mb)
    assert not should_try_optional_batch_128(high_memory, total_vram_mb=total_vram_mb)
    assert not should_try_optional_batch_128(oom, total_vram_mb=total_vram_mb)
    assert OPTIONAL_BATCH_MEMORY_FRACTION == 0.60


def test_profile_artifact_serialization(tmp_path) -> None:
    summary = {"profile_type": "MODEL-COMPUTE-ONLY PROFILE", "precision": "FP32"}
    rows = [
        {
            "batch_size": 8,
            "status": "success",
            "warmup_steps": 5,
            "measured_steps": 20,
            "mean_step_ms": 1.0,
            "median_step_ms": 1.0,
            "std_step_ms": 0.0,
            "images_per_second": 8_000.0,
            "peak_allocated_mb": 100.0,
            "peak_reserved_mb": 128.0,
            "total_vram_mb": 8_000.0,
            "allocated_fraction": 0.0125,
        }
    ]

    write_profile_artifacts(tmp_path, summary, rows)

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
    with (tmp_path / "batch_profile.csv").open(newline="", encoding="utf-8") as stream:
        serialized_rows = list(csv.DictReader(stream))
    assert serialized_rows[0]["batch_size"] == "8"
    assert serialized_rows[0]["status"] == "success"


def test_precision_configuration_and_comparison_formulas() -> None:
    validate_precision("FP32")
    validate_precision("AMP_FP16")
    with pytest.raises(ValueError, match="precision"):
        validate_precision("BF16")

    comparison = compare_precision_results(
        {
            "median_step_ms": 20.0,
            "images_per_second": 1_600.0,
            "peak_allocated_mb": 500.0,
            "peak_reserved_mb": 600.0,
        },
        {
            "median_step_ms": 10.0,
            "images_per_second": 3_200.0,
            "peak_allocated_mb": 300.0,
            "peak_reserved_mb": 400.0,
        },
    )

    assert comparison["speedup"] == pytest.approx(2.0)
    assert comparison["throughput_gain_percent"] == pytest.approx(100.0)
    assert comparison["allocated_memory_reduction_percent"] == pytest.approx(40.0)
    assert comparison["reserved_memory_reduction_percent"] == pytest.approx(
        100.0 / 3.0
    )


def test_fp32_and_amp_share_the_core_measurement_contract() -> None:
    assert AMP_MEASUREMENT_FIELDS[0] == "precision"
    assert AMP_MEASUREMENT_FIELDS[1 : 1 + len(MEASUREMENT_FIELDS)] == (
        MEASUREMENT_FIELDS
    )
    assert PRECISIONS == ("FP32", "AMP_FP16")


def test_tf32_metadata_records_flags_without_changing_them() -> None:
    before = backend_precision_metadata()
    after = backend_precision_metadata()

    assert set(before) == {"cudnn_allow_tf32", "cuda_matmul_allow_tf32"}
    assert all(isinstance(value, bool) for value in before.values())
    assert after == before


def test_amp_pilot_artifact_serialization(tmp_path) -> None:
    row = {
        "precision": "FP32",
        "batch_size": 32,
        "status": "success",
        "warmup_steps": 5,
        "measured_steps": 20,
        "mean_step_ms": 4.0,
        "median_step_ms": 4.0,
        "std_step_ms": 0.1,
        "images_per_second": 8_000.0,
        "peak_allocated_mb": 500.0,
        "peak_reserved_mb": 600.0,
        "total_vram_mb": 8_000.0,
        "allocated_fraction": 0.0625,
        "grad_scaler_initial_scale": None,
        "grad_scaler_final_scale": None,
        "warmup_skipped_steps": 0,
        "measured_skipped_steps": 0,
    }
    rows = [row, {**row, "precision": "AMP_FP16"}]
    summary = {
        "profile_type": "MODEL-COMPUTE-ONLY AMP FP16 PARITY + BENEFIT PILOT",
        "cudnn_allow_tf32": True,
        "cuda_matmul_allow_tf32": False,
    }
    parity = {
        "fp32_loss": 1.0,
        "amp_loss": 1.001,
        "representative_gradients": {
            "input_projection.weight": {
                "fp32_finite": True,
                "amp_finite": True,
            }
        },
    }

    write_amp_pilot_artifacts(tmp_path, summary, rows, parity)

    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert json.loads((tmp_path / "parity.json").read_text()) == parity
    with (tmp_path / "precision_profile.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        serialized_rows = list(csv.DictReader(stream))
    assert [item["precision"] for item in serialized_rows] == [
        "FP32",
        "AMP_FP16",
    ]
