from __future__ import annotations

import itertools
import json
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import pytest
from PIL import Image

from tiny_avatar_diffusion.dr05_diversity import aggregate_pairs, bootstrap_ood_family_scopes, bootstrap_prompt_scopes, enumerate_pairs, ood_family_prompt_rows, png_to_lpips_tensor, reject_overlapping_output_root, validate_images, validate_population


SEEDS = (2026, 2027, 2028, 2029, 2030)


def canonical_rows() -> list[dict]:
    rows = []
    index = 0
    for split, families, count in (("ID", (None,), 152), ("OOD", ("OOD1", "OOD2", "OOD3", "OOD4"), 25)):
        for family in families:
            for prompt in range(count):
                prompt_id = f"{split}_{family}_{prompt}" if family else f"ID_{prompt}"
                for seed in SEEDS:
                    rows.append({"sample_id": f"s{index}", "prompt_id": prompt_id, "seed": seed, "panel_split": split, "ood_family": family, "control_role": None, "png_path": f"images/s{index}.png"})
                    index += 1
    return rows


def test_pair_combinatorics_and_canonical_geometry():
    groups = validate_population(canonical_rows())
    pairs = enumerate_pairs(groups)
    assert len(groups) == 252 and len(pairs) == 2520
    assert sum(pair["panel_split"] == "ID" for pair in pairs) == 1520
    assert sum(pair["panel_split"] == "OOD" for pair in pairs) == 1000
    assert {family: sum(pair["ood_family"] == family for pair in pairs) for family in ("OOD1", "OOD2", "OOD3", "OOD4")} == {family: 250 for family in ("OOD1", "OOD2", "OOD3", "OOD4")}
    first = [pair for pair in pairs if pair["prompt_id"] == "ID_0"]
    assert [(pair["seed_a"], pair["seed_b"]) for pair in first] == list(itertools.combinations(SEEDS, 2))
    assert len({pair["pair_id"] for pair in pairs}) == 2520
    assert all(pair["seed_a"] < pair["seed_b"] for pair in pairs)


@pytest.mark.parametrize("mutation", [lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[0])), lambda rows: rows.__setitem__(0, {**rows[0], "seed": 999})])
def test_seed_validation_rejects_missing_duplicate_and_wrong_sets(mutation):
    rows = canonical_rows(); mutation(rows)
    with pytest.raises(ValueError):
        validate_population(rows)


def test_image_and_transform_validation(tmp_path: Path):
    rgb = tmp_path / "rgb.png"; Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8), "RGB").save(rgb)
    tensor = png_to_lpips_tensor(rgb)
    assert tensor.shape == (1, 3, 32, 32) and tensor.dtype.is_floating_point and float(tensor.min()) == float(tensor.max()) == 1.0
    wrong_mode = tmp_path / "gray.png"; Image.new("L", (32, 32)).save(wrong_mode)
    wrong_size = tmp_path / "large.png"; Image.new("RGB", (31, 32)).save(wrong_size)
    for path in (wrong_mode, wrong_size):
        with pytest.raises(ValueError): png_to_lpips_tensor(path)
    for name in ("rgb.png", "gray.png", "large.png"):
        row = {"png_path": name}
        if name == "rgb.png": validate_images([row], tmp_path)
        else:
            with pytest.raises(ValueError): validate_images([row], tmp_path)


def test_prompt_and_split_aggregation_are_macro_and_ood_is_equal_family_macro():
    rows = []
    for family, value in zip(("OOD1", "OOD2", "OOD3", "OOD4"), (1.0, 2.0, 3.0, 4.0), strict=True):
        for pair in range(10): rows.append({"prompt_id": family, "panel_split": "OOD", "ood_family": family, "control_role": None, "lpips": value, "l1": value, "pair_id": f"{family}-{pair}"})
    rows.extend({"prompt_id": "id_a", "panel_split": "ID", "ood_family": None, "control_role": None, "lpips": 0.0, "l1": 0.0, "pair_id": f"id-a-{pair}"} for pair in range(10))
    rows.extend({"prompt_id": "id_b", "panel_split": "ID", "ood_family": None, "control_role": None, "lpips": 10.0, "l1": 10.0, "pair_id": f"id-b-{pair}"} for pair in range(10))
    prompts, aggregates = aggregate_pairs(rows)
    assert len(prompts) == 6
    assert aggregates["ID_prompt_macro_lpips"] == 5.0
    assert aggregates["OOD_prompt_macro_lpips"] == 2.5
    assert aggregates["OOD_equal_family_macro_lpips"] == 2.5
    bootstrap = bootstrap_prompt_scopes(prompts)
    assert bootstrap["seed"] == 202605 and bootstrap["resamples"] == 10000 and set(bootstrap["scopes"]) == {"ID", "OOD", "OOD1", "OOD2", "OOD3", "OOD4"}
    rows[0]["lpips"] = float("nan")
    with pytest.raises(ValueError): aggregate_pairs(rows)


def test_ood_family_selection_excludes_matched_id_controls_from_aggregate_and_bootstrap():
    ood_rows = [{"prompt_id": f"ood-{index}", "panel_split": "OOD", "ood_family": "OOD1", "control_role": None, "mean_lpips": 1.0} for index in range(25)]
    controls = [{"prompt_id": f"a-{index}", "panel_split": "ID", "ood_family": "OOD1", "control_role": "A_FIXED", "mean_lpips": 100.0} for index in range(19)]
    controls += [{"prompt_id": f"b-{index}", "panel_split": "ID", "ood_family": "OOD1", "control_role": "B_FIXED", "mean_lpips": 100.0} for index in range(19)]
    rows = ood_rows + controls
    assert len(ood_family_prompt_rows(rows, "OOD1")) == 25
    assert np.mean([row["mean_lpips"] for row in ood_family_prompt_rows(rows, "OOD1")]) == 1.0
    first = bootstrap_ood_family_scopes(rows)["OOD1"]
    second = bootstrap_ood_family_scopes(ood_rows + [{**row, "mean_lpips": -100.0} for row in controls])["OOD1"]
    assert first == second == {"estimate": 1.0, "percentile_95": [1.0, 1.0]}


def test_balanced_ood_family_macro_equals_overall_ood_macro():
    rows = [
        {"prompt_id": f"{family}-{index}", "panel_split": "OOD", "ood_family": family, "control_role": None, "mean_lpips": float(value)}
        for family, value in zip(("OOD1", "OOD2", "OOD3", "OOD4"), (1, 2, 3, 4), strict=True)
        for index in range(25)
    ]
    family_macro = np.mean([np.mean([row["mean_lpips"] for row in ood_family_prompt_rows(rows, family)]) for family in ("OOD1", "OOD2", "OOD3", "OOD4")])
    overall_macro = np.mean([row["mean_lpips"] for row in rows if row["panel_split"] == "OOD"])
    assert family_macro == overall_macro


def test_output_root_cannot_overlap_dr04_input(tmp_path: Path):
    source = tmp_path / "dr04"; source.mkdir()
    with pytest.raises(ValueError): reject_overlapping_output_root(source, source)
    with pytest.raises(ValueError): reject_overlapping_output_root(source, source / "child")
    reject_overlapping_output_root(source, tmp_path / "dr05")


def test_cli_script_entrypoint_reaches_synthetic_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Exercise the script-level ``__main__`` guard without final DR04 data."""
    source_root, output_root = tmp_path / "source", tmp_path / "output"
    calls: list[str] = []
    pairs = [{"pair_id": f"pair-{index}"} for index in range(20)]
    prompts = [{"prompt_id": f"prompt-{index}"} for index in range(2)]
    fake = types.ModuleType("tiny_avatar_diffusion.dr05_diversity")
    fake.DEFAULT_OUTPUT_ROOT = output_root
    fake.SOURCE_ROOT = source_root
    fake.reject_overlapping_output_root = lambda source, output: calls.append("root_check")
    fake.source_artifact_hashes = lambda source: calls.append("source_hashes")
    fake.load_jsonl = lambda path: (calls.append("load_manifest") or [{"synthetic": True}])
    fake.validate_population = lambda rows: (calls.append("validate_population") or {"synthetic": rows})
    fake.validate_images = lambda rows, source: calls.append("validate_images")
    fake.enumerate_pairs = lambda groups: (calls.append("enumerate_pairs") or pairs)
    fake.lpips_alexnet_metric = lambda device: (calls.append("build_metric") or (lambda left, right: 0.0))
    fake.measure_pairs = lambda rows, source, metric: (calls.append("measure_pairs") or rows)
    fake.aggregate_pairs = lambda rows: (calls.append("aggregate_pairs") or (prompts, {"synthetic": 1.0}))
    fake.bootstrap_prompt_scopes = lambda rows: (calls.append("bootstrap") or {"synthetic": True})

    def write_results(**kwargs):
        calls.append("write_results")
        kwargs["output_root"].mkdir()
        for name, payload in (
            ("pair_records.jsonl", "\n".join(json.dumps(row) for row in kwargs["pair_rows"]) + "\n"),
            ("prompt_records.jsonl", "\n".join(json.dumps(row) for row in kwargs["prompt_rows"]) + "\n"),
            ("diversity_summary.json", json.dumps({"pair_rows": len(kwargs["pair_rows"]), "prompt_rows": len(kwargs["prompt_rows"])})),
        ):
            (kwargs["output_root"] / name).write_text(payload, encoding="utf-8")

    fake.write_results = write_results
    monkeypatch.setitem(sys.modules, "tiny_avatar_diffusion.dr05_diversity", fake)
    monkeypatch.setattr(sys, "argv", ["dr05_final_diversity_evaluation.py", "--source-root", str(source_root), "--output-root", str(output_root)])

    runpy.run_path(Path("scripts/dr05_final_diversity_evaluation.py"), run_name="__main__")

    assert calls == ["root_check", "source_hashes", "load_manifest", "validate_population", "validate_images", "enumerate_pairs", "build_metric", "measure_pairs", "aggregate_pairs", "bootstrap", "write_results"]
    assert json.loads((output_root / "diversity_summary.json").read_text(encoding="utf-8")) == {"pair_rows": 20, "prompt_rows": 2}
    output = capsys.readouterr().out
    markers = ["START", "PROTOCOL_OK", "INPUT_OK", "LPIPS_READY", "PAIRS_COMPLETE", "AGGREGATION_COMPLETE", "BOOTSTRAP_COMPLETE", "WRITE_COMPLETE", "COMPLETE"]
    positions = [output.index(f"[DR05-DIVERSITY] {marker}") for marker in markers]
    assert positions == sorted(positions)


def test_cli_script_entrypoint_fails_closed_when_writer_omits_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """A writer return without canonical artifacts must not yield a successful CLI run."""
    source_root, output_root = tmp_path / "source", tmp_path / "output"
    fake = types.ModuleType("tiny_avatar_diffusion.dr05_diversity")
    fake.DEFAULT_OUTPUT_ROOT = output_root
    fake.SOURCE_ROOT = source_root
    fake.reject_overlapping_output_root = lambda source, output: None
    fake.source_artifact_hashes = lambda source: None
    fake.load_jsonl = lambda path: [{"synthetic": True}]
    fake.validate_population = lambda rows: {"synthetic": rows}
    fake.validate_images = lambda rows, source: None
    fake.enumerate_pairs = lambda groups: [{"pair_id": "pair-0"}]
    fake.lpips_alexnet_metric = lambda device: lambda left, right: 0.0
    fake.measure_pairs = lambda rows, source, metric: rows
    fake.aggregate_pairs = lambda rows: ([{"prompt_id": "prompt-0"}], {"synthetic": 1.0})
    fake.bootstrap_prompt_scopes = lambda rows: {"synthetic": True}
    fake.write_results = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "tiny_avatar_diffusion.dr05_diversity", fake)
    monkeypatch.setattr(sys, "argv", ["dr05_final_diversity_evaluation.py", "--source-root", str(source_root), "--output-root", str(output_root)])

    with pytest.raises(FileNotFoundError, match="output root is missing"):
        runpy.run_path(Path("scripts/dr05_final_diversity_evaluation.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "[DR05-DIVERSITY] WRITE_COMPLETE" in output
    assert "[DR05-DIVERSITY] COMPLETE" not in output
    assert "[DR05-DIVERSITY] FAILED stage=WRITE_COMPLETE" in output
