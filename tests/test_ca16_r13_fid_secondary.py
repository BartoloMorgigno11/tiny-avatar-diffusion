from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import tiny_avatar_diffusion.ca16_r13_fid_secondary as fid_secondary


def _protocol() -> dict[str, object]:
    return {
        "schema": "tiny_avatar_diffusion.ca16_r13_fid_secondary_protocol_v1",
        "protocol_id": fid_secondary.PROTOCOL_ID,
        "status": "FROZEN",
        "scientific_scope": {"interpretation": "synthetic secondary diagnostic"},
        "fid": {
            "metrics": {"FID_ID": 466, "FID_COMP": 338, "FID_GLOBAL": 804},
            "implementation": {
                "package": "clean-fid",
                "version": fid_secondary.CLEAN_FID_VERSION,
                "mode": "clean",
                "feature_extractor": "standard Inception-v3",
                "feature_batch_size": 8,
                "num_workers": 0,
            },
        },
        "source_r13": {
            "annotated_tag": "source-tag",
            "frozen_commit": "source",
            "root": "source",
            "r13_protocol_path": "r13.json",
            "r13_protocol_sha256": "r13",
            "quality_manifest_path": "quality.json",
            "quality_manifest_sha256": "quality",
            "real_reference_manifest_path": "real.jsonl",
            "real_reference_manifest_sha256": "real",
            "completed_summary_sha256": {},
        },
        "output_root": "results/r13_fid_secondary_v1",
    }


def test_preflight_is_read_only_and_rejects_existing_output(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    monkeypatch.setattr(fid_secondary, "load_protocol", lambda: protocol)
    monkeypatch.setattr(
        fid_secondary.importlib.metadata,
        "version",
        lambda _name: fid_secondary.CLEAN_FID_VERSION,
    )
    monkeypatch.setattr(
        fid_secondary, "_git_state", lambda: ("freeze", "exp/ca16-ablation", True)
    )
    monkeypatch.setattr(
        fid_secondary.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "source\n"})(),
    )
    monkeypatch.setattr(
        fid_secondary,
        "verify_r13_source",
        lambda _protocol: {"runs": [{"run_id": f"run-{index}"} for index in range(6)]},
    )
    output = tmp_path / "unused"
    result = fid_secondary.validate_preflight(
        output_root=output, expected_eval_sha="freeze"
    )
    assert result["status"] == "preflight_pass"
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError, match="unused"):
        fid_secondary.validate_preflight(output_root=output, expected_eval_sha="freeze")


def test_load_protocol_rejects_an_unbound_protocol_hash(monkeypatch) -> None:
    monkeypatch.setattr(fid_secondary, "sha256_file", lambda _path: "wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        fid_secondary.load_protocol()


def test_verify_pool_rejects_missing_or_hash_mismatched_member(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "images" / "id_val" / "0000.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "0000.png").write_bytes(b"source")
    rows = [{"planned_sample_index": 0, "png_relative_path": "images/id_val/0000.png"}]
    fid_secondary._verify_pool(pool, source_root, rows, name="synthetic")
    (pool / "0000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        fid_secondary._verify_pool(pool, source_root, rows, name="synthetic")


def test_real_reference_validation_rejects_pixel_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    image = source_root / "real_references" / "id_val" / "0000.png"
    image.parent.mkdir(parents=True)
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(image)
    plan = [
        {
            "sample_id": "quality_0000",
            "planned_sample_index": 0,
            "stable_record_key": "10k|ID_VAL|image",
            "split_role": "ID_VAL",
        }
    ]
    row = {
        "schema": "tiny_avatar_diffusion.ca16_r13_real_reference_v1",
        "sample_id": "quality_0000",
        "planned_sample_index": 0,
        "stable_record_key": "10k|ID_VAL|image",
        "split_role": "ID_VAL",
        "png_relative_path": "real_references/id_val/0000.png",
        "status": "complete",
        "png_sha256": fid_secondary.sha256_file(image),
        "pixel_sha256": "wrong",
    }
    with pytest.raises(ValueError, match="pixels are invalid"):
        fid_secondary._verify_real_references(source_root, plan, [row])


def test_execute_writes_only_secondary_result(monkeypatch, tmp_path: Path) -> None:
    protocol = _protocol()
    output = tmp_path / "r13_fid_secondary_v1"
    monkeypatch.setattr(fid_secondary, "load_protocol", lambda: protocol)
    monkeypatch.setattr(fid_secondary.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        fid_secondary,
        "validate_preflight",
        lambda **kwargs: {"status": "preflight_pass"},
    )
    monkeypatch.setattr(
        fid_secondary,
        "verify_r13_source",
        lambda _protocol: {"source_root": tmp_path, "real_pools": {}, "runs": []},
    )
    monkeypatch.setattr(
        fid_secondary,
        "compute_fid_metrics",
        lambda _verified, **kwargs: [{"run_id": "synthetic", "fid": {}}],
    )
    monkeypatch.setattr(fid_secondary, "summarize_fid_results", lambda _results: {})
    monkeypatch.setattr(fid_secondary, "sha256_file", lambda _path: "protocol-hash")
    payload = fid_secondary.execute_fid(
        output_root=output, expected_eval_sha="freeze", expected_tag="tag"
    )
    assert payload["status"] == "COMPLETE"
    assert (output / fid_secondary.OUTPUT_NAME).is_file()
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_fid_from_features_rejects_nonfinite(monkeypatch) -> None:
    class FakeFID:
        @staticmethod
        def frechet_distance(*_args: object) -> float:
            return float("nan")

    monkeypatch.setitem(
        __import__("sys").modules, "cleanfid", type("Module", (), {"fid": FakeFID})()
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        fid_secondary._fid_from_features(np.zeros((2, 2)), np.zeros((2, 2)))


def test_compute_fid_metrics_produces_three_values_for_each_run(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeFID:
        @staticmethod
        def build_feature_extractor(**_kwargs: object) -> object:
            return object()

        @staticmethod
        def get_folder_features(*_args: object, **_kwargs: object) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.float64)

        @staticmethod
        def frechet_distance(*_args: object) -> float:
            return 0.25

    monkeypatch.setitem(
        __import__("sys").modules, "cleanfid", type("Module", (), {"fid": FakeFID})()
    )
    verified = {
        "real_pools": {
            name: tmp_path / name for name in ("FID_ID", "FID_COMP", "FID_GLOBAL")
        },
        "runs": [
            {"run_id": f"run-{index}", "checkpoint": {"index": index}, "root": tmp_path}
            for index in range(6)
        ],
    }
    results = fid_secondary.compute_fid_metrics(verified, device=torch.device("cpu"))
    assert len(results) == 6
    for result in results:
        assert set(result["fid"]) == {"FID_ID", "FID_COMP", "FID_GLOBAL"}
        assert result["fid"]["FID_ID"]["real_count"] == 466
        assert result["fid"]["FID_COMP"]["generated_count"] == 338
        assert result["fid"]["FID_GLOBAL"]["value"] == 0.25


def test_fid_summary_uses_paired_ca16_minus_ca8() -> None:
    results = []
    for arm, base in (("ca8", 0.2), ("ca8_ca16", 0.1)):
        for seed in (2026, 2027, 2028):
            results.append(
                {
                    "checkpoint": {"architecture_arm": arm, "training_seed": seed},
                    "fid": {
                        name: {"value": base}
                        for name in ("FID_ID", "FID_COMP", "FID_GLOBAL")
                    },
                }
            )
    summary = fid_secondary.summarize_fid_results(results)
    assert summary["paired_differences"]["2026"]["FID_ID"] == pytest.approx(-0.1)
    assert summary["means"]["FID_GLOBAL"]["ca8_ca16_minus_ca8"] == pytest.approx(-0.1)
