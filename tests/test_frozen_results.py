from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tiny_avatar_diffusion.frozen_results import ROOT, _read_json, load_frozen_results

SOURCE_PATHS = (
    "results/runs/dr04_final_conditioning_v1/final_conditioning_summary.json",
    "results/runs/dr05_final_quality_v1/final_quality_summary.json",
    "results/runs/dr05_final_diversity_v1/diversity_summary.json",
    "results/runs/dr05_final_efficiency_v1/parameter_count_summary.json",
    "results/runs/dr05_final_efficiency_v1/latency_summary.json",
    "results/runs/dr05_final_efficiency_v1/conditional_b1.json",
    "results/runs/dr05_final_efficiency_v1/unconditional_b1.json",
    "results/runs/dr05_final_efficiency_v1/conditional_b16.json",
    "results/runs/dr05_final_efficiency_v1/unconditional_b16.json",
)


def _numbers(value: object) -> list[int | float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [number for child in value.values() for number in _numbers(child)]
    if isinstance(value, list):
        return [number for child in value for number in _numbers(child)]
    return []


def _evidence_copy(tmp_path: Path) -> Path:
    for relative in SOURCE_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def _mutate_json(root: Path, relative: str, mutate: object) -> None:
    path = root / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_results_parse_required_headline_evidence_without_mutation() -> None:
    before = {relative: (ROOT / relative).read_bytes() for relative in SOURCE_PATHS}
    results = load_frozen_results()
    after = {relative: (ROOT / relative).read_bytes() for relative in SOURCE_PATHS}
    assert after == before

    assert set(results) == {
        "conditioning",
        "quality_global",
        "quality_by_split",
        "lpips_diversity_global_prompt_macro",
        "parameter_counts",
        "latency",
        "memory",
    }
    assert set(results["conditioning"]) == {
        "id_maa",
        "ood_maa",
        "absolute_maa_gap",
        "macro_psr",
        "id_maa_bootstrap_95",
        "ood_maa_bootstrap_95",
    }
    assert set(results["quality_global"]) == {"conditional", "unconditional"}
    assert set(results["quality_global"]["conditional"]) == {"kid", "fid"}
    assert set(results["quality_by_split"]) == {"conditional", "unconditional"}
    assert set(results["quality_by_split"]["conditional"]) == {"id", "ood"}
    assert set(results["quality_by_split"]["conditional"]["id"]) == {"kid", "fid"}
    assert "gap_ci" not in results["conditioning"]
    assert set(results["parameter_counts"]) == {"conditional", "unconditional"}
    assert set(results["latency"]["conditional"]) == {"batch_1", "batch_16"}
    assert set(results["memory"]["unconditional"]) == {"batch_1", "batch_16"}
    assert all(math.isfinite(number) for number in _numbers(results))


def test_extended_values_are_read_from_canonical_evidence() -> None:
    dr04 = json.loads((ROOT / SOURCE_PATHS[0]).read_text(encoding="utf-8"))
    quality = json.loads((ROOT / SOURCE_PATHS[1]).read_text(encoding="utf-8"))
    results = load_frozen_results()

    assert results["conditioning"]["id_maa_bootstrap_95"] == dict(
        zip(("lower", "upper"), dr04["bootstrap"]["scopes"]["ID"]["percentile_95"], strict=True)
    )
    assert results["conditioning"]["ood_maa_bootstrap_95"] == dict(
        zip(("lower", "upper"), dr04["bootstrap"]["scopes"]["OOD"]["percentile_95"], strict=True)
    )
    assert results["conditioning"]["absolute_maa_gap"] == abs(
        dr04["primary_conditioning"]["ID_MAA"]
        - dr04["primary_conditioning"]["OOD_MAA"]
    )
    for model in ("conditional", "unconditional"):
        for split in ("ID", "OOD"):
            source = quality["state"]["phase3"]["metrics"][f"{model}:{split}"]["result"]
            assert results["quality_by_split"][model][split.lower()] == {
                "kid": source["kid"],
                "fid": source["fid"],
            }


def test_extended_surface_validates_intervals_and_split_quality(tmp_path: Path) -> None:
    invalid_interval = _evidence_copy(tmp_path / "interval")
    _mutate_json(
        invalid_interval,
        SOURCE_PATHS[0],
        lambda payload: payload["bootstrap"]["scopes"]["ID"].update(
            percentile_95=[0.6, 0.7]
        ),
    )
    with pytest.raises(ValueError, match="estimate must lie within"):
        load_frozen_results(invalid_interval)

    missing_split = _evidence_copy(tmp_path / "quality")
    _mutate_json(
        missing_split,
        SOURCE_PATHS[1],
        lambda payload: payload["state"]["phase3"]["metrics"]["conditional:ID"][
            "result"
        ].pop("kid"),
    )
    with pytest.raises(ValueError, match="field is missing"):
        load_frozen_results(missing_split)


def test_frozen_results_reader_fails_clearly_on_missing_malformed_or_wrong_schema(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Canonical evidence is missing"):
        _read_json(tmp_path, "missing.json", "expected")
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _read_json(tmp_path, "bad.json", "expected")
    (tmp_path / "wrong.json").write_text('{"schema": "wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        _read_json(tmp_path, "wrong.json", "expected")


def test_frozen_results_rejects_missing_nested_quality_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    root = _evidence_copy(tmp_path)
    quality_path = SOURCE_PATHS[1]
    _mutate_json(
        root,
        quality_path,
        lambda payload: payload["state"]["phase3"]["metrics"]["conditional:GLOBAL"][
            "result"
        ].pop("kid"),
    )
    with pytest.raises(ValueError, match="field is missing"):
        load_frozen_results(root)

    root = _evidence_copy(tmp_path / "nonfinite")
    _mutate_json(
        root,
        quality_path,
        lambda payload: payload["state"]["phase3"]["metrics"]["conditional:GLOBAL"][
            "result"
        ].update(kid=float("nan")),
    )
    with pytest.raises(ValueError, match="must be finite"):
        load_frozen_results(root)


def test_frozen_results_rejects_missing_duplicate_or_incomplete_latency_conditions(
    tmp_path: Path,
) -> None:
    latency_path = SOURCE_PATHS[4]
    missing = _evidence_copy(tmp_path / "missing")
    _mutate_json(missing, latency_path, lambda payload: payload["conditions"].pop())
    with pytest.raises(ValueError, match="four conditions"):
        load_frozen_results(missing)

    duplicate = _evidence_copy(tmp_path / "duplicate")
    _mutate_json(
        duplicate,
        latency_path,
        lambda payload: payload["conditions"][1].update(
            model_type=payload["conditions"][0]["model_type"],
            batch_size=payload["conditions"][0]["batch_size"],
        ),
    )
    with pytest.raises(ValueError, match="Duplicate canonical latency"):
        load_frozen_results(duplicate)

    incomplete = _evidence_copy(tmp_path / "incomplete")
    _mutate_json(
        incomplete,
        latency_path,
        lambda payload: payload["conditions"][0]["statistics"].pop(
            "median_seconds_per_image"
        ),
    )
    with pytest.raises(ValueError, match="field is missing"):
        load_frozen_results(incomplete)


def test_frozen_results_rejects_invalid_missing_duplicate_or_malformed_memory(
    tmp_path: Path,
) -> None:
    missing = _evidence_copy(tmp_path / "missing")
    (missing / SOURCE_PATHS[5]).unlink()
    with pytest.raises(FileNotFoundError, match="Canonical evidence is missing"):
        load_frozen_results(missing)

    duplicate = _evidence_copy(tmp_path / "duplicate")
    _mutate_json(
        duplicate,
        SOURCE_PATHS[7],
        lambda payload: payload.update(model_type="conditional", batch_size=1),
    )
    with pytest.raises(ValueError, match="Duplicate canonical memory"):
        load_frozen_results(duplicate)

    invalid = _evidence_copy(tmp_path / "invalid")
    _mutate_json(invalid, SOURCE_PATHS[5], lambda payload: payload.update(batch_size=2))
    with pytest.raises(ValueError, match="Unexpected canonical memory"):
        load_frozen_results(invalid)

    malformed = _evidence_copy(tmp_path / "malformed")
    _mutate_json(
        malformed,
        SOURCE_PATHS[5],
        lambda payload: payload.update(peak_allocated_bytes="not-a-number"),
    )
    with pytest.raises(ValueError, match="must be finite"):
        load_frozen_results(malformed)


def test_frozen_results_rejects_defined_version_mismatch(tmp_path: Path) -> None:
    root = _evidence_copy(tmp_path)
    _mutate_json(root, SOURCE_PATHS[3], lambda payload: payload.update(version=2))
    with pytest.raises(ValueError, match="version mismatch"):
        load_frozen_results(root)
