from __future__ import annotations

import csv
import json

import pytest

from tiny_avatar_diffusion.profile_gpu import (
    OPTIONAL_BATCH_MEMORY_FRACTION,
    ProfileConfig,
    images_per_second,
    should_try_optional_batch_128,
    validate_config,
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
