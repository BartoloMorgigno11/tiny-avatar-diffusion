from __future__ import annotations

from scripts.dr05_final_runtime_benchmark import (
    MEASURED, PRIMARY, PROMPT, PROMPT_ID, SECONDARY, WARMUPS,
    benchmark_seeds, conditional_overhead, latency_summary, memory_summary,
)


def test_frozen_benchmark_namespaces_seeds_and_prompt():
    assert WARMUPS == 3 and MEASURED == 10
    assert PRIMARY == {"batch_size": 1, "namespace": "latency_primary_b1", "seed_base": 602026}
    assert SECONDARY == {"batch_size": 16, "namespace": "latency_secondary_b16", "seed_base": 612026}
    assert benchmark_seeds(PRIMARY["seed_base"]) == ([602026, 602027, 602028], list(range(602029, 602039)))
    assert benchmark_seeds(SECONDARY["seed_base"]) == ([612026, 612027, 612028], list(range(612029, 612039)))
    assert PROMPT_ID == "cs10002930376453395152"
    assert PROMPT == "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"


def test_latency_overhead_and_memory_summaries_are_exact():
    rows = [{"seconds": value, "pre_sampling_allocated_bytes": 10, "peak_allocated_bytes": 10 + value, "peak_reserved_bytes": 30 + value, "incremental_peak_allocated_bytes": value} for value in range(1, 11)]
    latency = latency_summary(rows, 1)
    assert latency["median_seconds_per_image"] == 5.5
    assert latency["mean_seconds_per_image"] == 5.5
    assert latency["median_images_per_second"] > 0
    memory = memory_summary(rows)
    assert memory == {"maximum_peak_allocated_bytes": 20, "median_peak_allocated_bytes": 15.5, "maximum_peak_reserved_bytes": 40, "median_pre_sampling_allocated_bytes": 10, "median_incremental_peak_allocated_bytes": 5.5}
    overhead = conditional_overhead({"median_seconds_per_image": 6.0}, {"median_seconds_per_image": 5.0})
    assert overhead == {"absolute_median_seconds_per_image": 1.0, "percent_median_overhead": 20.0}
