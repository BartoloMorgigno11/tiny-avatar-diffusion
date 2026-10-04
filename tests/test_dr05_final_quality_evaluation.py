from __future__ import annotations

import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import linalg

import scripts.dr05_final_quality_evaluation as quality

from scripts.dr05_final_quality_evaluation import (
    BATCH_SIZE,
    CLEAN_FID_VERSION,
    CONDITIONAL,
    CONDITIONAL_SHA256,
    DRY_RUN_NAMESPACE,
    DRY_RUN_SEED_BASE,
    FINAL_ALLOCATIONS,
    FINAL_NAMESPACE,
    FINAL_SEED_BASE,
    KID_SUBSET_SEED,
    GENERATED_MANIFEST_FIELDS,
    REAL_MANIFEST_FIELDS,
    UNCONDITIONAL,
    UNCONDITIONAL_SHA256,
    assert_dry_run_records,
    canonicalize_real_image,
    deterministic_kid_rng,
    dry_run_generation_plan,
    final_generation_plan,
    generated_to_rgb_uint8,
    metric_config,
    pixel_sha256,
    save_canonical_png,
    sha256_file,
    validate_mode,
)


@pytest.fixture
def retained_i19_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal, synthetic retained-pool fixture for recovery tests."""
    monkeypatch.setattr(quality, "ROOT", Path(tmp_path.anchor))
    root = tmp_path / "retained_i19"
    ids = ["cs10002930376453395152", *[f"cs1001_synthetic_{index:02d}" for index in range(1, 31)], "cs10096894922495846065"]
    real_rows: list[dict[str, object]] = []
    generated_rows = {"conditional": [], "unconditional": []}
    for index, reference_id in enumerate(ids):
        for pool, rows in (("real_id_val", real_rows), ("conditional", generated_rows["conditional"]), ("unconditional", generated_rows["unconditional"])):
            pixels = np.full((32, 32, 3), (index * 17 + len(pool)) % 256, dtype=np.uint8)
            relative = Path(pool) / f"{pool}_{index:02d}.png"
            hashes = save_canonical_png(pixels, root / relative)
            if pool == "real_id_val":
                rows.append({
                    "reference_sample_id": reference_id, "source_split": "ID_VAL",
                    "source_image_reference": f"synthetic.tar.gz::{reference_id}.png",
                    "canonical_shape": [32, 32, 3], "mode": "RGB", "dtype": "uint8",
                    "canonical_png_relative_path": relative.as_posix(), **hashes, "status": "complete",
                })
            else:
                conditional = pool == "conditional"
                checkpoint = quality.CONDITIONAL if conditional else quality.UNCONDITIONAL
                rows.append({
                    "sample_id": f"{pool}_{index:02d}", "model": pool,
                    "checkpoint_path": checkpoint.relative_to(quality.ROOT).as_posix(),
                    "checkpoint_sha256": quality.CONDITIONAL_SHA256 if conditional else quality.UNCONDITIONAL_SHA256,
                    "evaluation_role": "NONSCIENTIFIC_DRYRUN", "split": "ID_VAL",
                    "reference_sample_id": reference_id, "prompt": f"synthetic prompt {index}",
                    "generation_namespace": quality.DRY_RUN_NAMESPACE, "batch_index": index // 16,
                    "batch_seed": quality.DRY_RUN_SEED_BASE + index // 16, "sample_index_in_batch": index % 16,
                    "image_shape": [32, 32, 3], "image_mode": "RGB", "dtype": "uint8",
                    "png_relative_path": relative.as_posix(), **hashes, "status": "complete",
                    "conditioning_consumed": conditional,
                })

    manifest_rows = {
        "real_id_val_manifest.jsonl": real_rows,
        "conditional_manifest.jsonl": generated_rows["conditional"],
        "unconditional_manifest.jsonl": generated_rows["unconditional"],
    }
    manifest_hashes = {}
    for name, rows in manifest_rows.items():
        path = root / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        manifest_hashes[name.removesuffix("_manifest.jsonl").replace("real_id_val", "real")] = sha256_file(path)
    monkeypatch.setattr(quality, "RETAINED_I19_MANIFEST_SHA256", manifest_hashes)
    return root


def _accept_test_manifest_hashes(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quality,
        "RETAINED_I19_MANIFEST_SHA256",
        {
            "real": sha256_file(root / "manifests" / "real_id_val_manifest.jsonl"),
            "conditional": sha256_file(root / "manifests" / "conditional_manifest.jsonl"),
            "unconditional": sha256_file(root / "manifests" / "unconditional_manifest.jsonl"),
        },
    )


def test_frozen_allocations_and_final_nested_batch_prefix() -> None:
    assert FINAL_ALLOCATIONS == {
        5_000: {"id": 2_884, "ood": 2_116},
        10_000: {"id": 5_767, "ood": 4_233},
        25_000: {"id": 14_418, "ood": 10_582},
        50_000: {"id": 28_837, "ood": 21_163},
    }
    assert sum(FINAL_ALLOCATIONS[10_000].values()) == 10_000
    plan_10k = final_generation_plan(10_000)
    plan_50k = final_generation_plan(50_000)
    assert len(plan_10k) == 625
    assert len(plan_50k) == 3125
    assert plan_10k == plan_50k[:625]
    assert plan_10k[0] == {"batch_index": 0, "batch_seed": FINAL_SEED_BASE}
    assert plan_10k[-1]["batch_seed"] == FINAL_SEED_BASE + 624
    assert BATCH_SIZE == 16 and FINAL_NAMESPACE == "quality_generation"


def test_d30_final_plan_is_metadata_only_deterministic_and_hash_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repository"
    id_test = root / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_test.jsonl"
    final_ood = root / "data" / "processed" / "dr01_v1" / "text" / "10k" / "final_ood.jsonl"
    plan = root / "data" / "processed" / "dr05_final_eval_v1" / "dr05_main_quality_generation_plan.jsonl"
    metadata = root / "data" / "processed" / "dr05_final_eval_v1" / "dr05_main_quality_generation_plan_meta.json"
    id_test.parent.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    id_test.write_text(
        "".join(json.dumps({"image_id": f"id_{index:03d}", "caption": f"ID caption {index}", "split_role": "ID_TEST"}) + "\n" for index in range(466)),
        encoding="utf-8",
    )
    final_ood.write_text(
        "".join(json.dumps({"image_id": f"ood_{index:03d}", "caption": f"OOD caption {index}", "split_role": "FINAL_OOD"}) + "\n" for index in range(342)),
        encoding="utf-8",
    )
    monkeypatch.setattr(quality, "ROOT", root)
    monkeypatch.setattr(quality, "ID_TEST", id_test)
    monkeypatch.setattr(quality, "FINAL_OOD", final_ood)
    monkeypatch.setattr(quality, "ID_TEST_METADATA_SHA256", sha256_file(id_test))
    monkeypatch.setattr(quality, "FINAL_OOD_METADATA_SHA256", sha256_file(final_ood))
    monkeypatch.setattr(quality, "MAIN_PLAN", plan)
    monkeypatch.setattr(quality, "MAIN_PLAN_META", metadata)
    monkeypatch.setattr(quality, "_decode_real_pixels", lambda _record: pytest.fail("D30 must not decode image content"))

    materialized = quality.materialize_main_quality_generation_plan()
    rows = quality._read_jsonl(plan)
    assert len(rows) == 10_000
    assert [row["planned_sample_index"] for row in rows] == list(range(10_000))
    assert [row["split"] for row in rows[:5767]] == ["ID"] * 5767
    assert [row["split"] for row in rows[5767:]] == ["OOD"] * 4233
    assert rows[5767]["batch_index"] == 360
    assert rows[5767]["sample_index_in_batch"] == 7
    assert rows[5767]["batch_seed"] == 502386
    assert Counter(row["reference_sample_id"] for row in rows[:5767]).values() and sorted(Counter(row["reference_sample_id"] for row in rows[:5767]).values()) == [12] * 291 + [13] * 175
    assert sorted(Counter(row["reference_sample_id"] for row in rows[5767:]).values()) == [12] * 213 + [13] * 129
    assert quality.verify_frozen_main_quality_generation_plan()["generation_plan_sha256"] == materialized["generation_plan_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        quality.materialize_main_quality_generation_plan()

    parsed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    parsed_metadata["generation"]["seed_base"] = 1
    metadata.write_text(json.dumps(parsed_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source, allocation, or seed policy"):
        quality.verify_frozen_main_quality_generation_plan()


def test_dry_run_plan_is_engineering_only_and_separate() -> None:
    plan = dry_run_generation_plan()
    assert plan == [
        {"batch_index": 0, "batch_seed": 702026},
        {"batch_index": 1, "batch_seed": 702027},
    ]
    assert DRY_RUN_NAMESPACE == "quality_pipeline_dryrun"
    assert DRY_RUN_SEED_BASE != FINAL_SEED_BASE


def test_generated_transform_png_round_trip_and_pixel_hash(tmp_path: Path) -> None:
    source = torch.full((3, 32, 32), 2.0)
    source[:, 0, 0] = -2.0
    source[:, 0, 1] = 0.0
    pixels = generated_to_rgb_uint8(source)
    assert pixels.shape == (32, 32, 3)
    assert pixels.dtype == np.uint8
    assert np.all(pixels[0, 0] == 0)
    assert np.all(pixels[0, 1] == 128)
    target = tmp_path / "generated.png"
    hashes = save_canonical_png(pixels, target)
    with Image.open(target) as image:
        reloaded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(pixels, reloaded)
    assert hashes["pixel_sha256"] == pixel_sha256(reloaded)
    assert hashes["png_sha256"] == sha256_file(target)


def test_real_transform_is_rgb_bilinear_32_and_deterministic() -> None:
    rgba = Image.new("RGBA", (9, 7), (20, 30, 40, 127))
    first = canonicalize_real_image(rgba)
    second = canonicalize_real_image(rgba)
    assert first.shape == (32, 32, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert pixel_sha256(first) == pixel_sha256(second)


def test_metric_contract_and_version_gate() -> None:
    config = metric_config(32, 32)
    assert importlib.metadata.version("clean-fid") == CLEAN_FID_VERSION
    assert importlib.metadata.version("scipy") == "1.15.3"
    assert importlib.metadata.version("lpips") == "0.1.4"
    assert config["mode"] == "clean"
    assert config["kid"] == {
        "num_subsets": 100,
        "subset_size": 32,
        "sampling": "without_replacement_within_subset",
        "subset_seed": KID_SUBSET_SEED,
    }
    assert config["fid"]["role"] == "secondary"


def test_scipy_sqrtm_accepts_clean_fid_disp_argument() -> None:
    root, diagnostic = linalg.sqrtm(np.array([[4.0, 0.0], [0.0, 9.0]]), disp=False)
    assert root.shape == (2, 2)
    assert np.isfinite(root).all()
    assert np.isfinite(diagnostic)


def test_deterministic_kid_rng_is_repeatable_and_restores_numpy_state() -> None:
    np.random.seed(91)
    expected_next = np.random.random()
    np.random.seed(91)
    with deterministic_kid_rng(KID_SUBSET_SEED):
        first = np.random.choice(32, 8, replace=False)
    assert np.random.random() == expected_next
    with deterministic_kid_rng(KID_SUBSET_SEED):
        second = np.random.choice(32, 8, replace=False)
    assert np.array_equal(first, second)


def test_dry_run_rejects_sealed_content_and_final_has_opt_in() -> None:
    with pytest.raises(ValueError):
        validate_mode(None)
    with pytest.raises(PermissionError):
        validate_mode("final")
    assert validate_mode("final", confirm_sealed_final_eval=True) == "final"
    assert validate_mode("dry-run") == "dry-run"
    record = {"image_id": "sealed", "split_role": "ID_TEST", "source_image_path": "a.tgz::x.png"}
    with pytest.raises(ValueError):
        assert_dry_run_records([record] * 32)


def test_canonical_checkpoint_integrity() -> None:
    assert sha256_file(CONDITIONAL) == CONDITIONAL_SHA256
    assert sha256_file(UNCONDITIONAL) == UNCONDITIONAL_SHA256


def test_manifest_sorting_contract_is_machine_readable(tmp_path: Path) -> None:
    assert {"conditioning_consumed", "png_sha256", "pixel_sha256"}.issubset(GENERATED_MANIFEST_FIELDS)
    assert {"source_image_reference", "png_sha256", "pixel_sha256"}.issubset(REAL_MANIFEST_FIELDS)
    # JSON key ordering is deterministic; the implementation validates full row schemas at write time.
    row = {"sample_id": "conditional_x", "conditioning_consumed": True, "png_sha256": "a", "pixel_sha256": "b"}
    encoded = json.dumps(row, sort_keys=True)
    assert list(json.loads(encoded)) == sorted(row)


def test_retained_pool_audit_validates_i19_pngs_and_cross_pool_bookkeeping(retained_i19_copy: Path) -> None:
    audit = quality.audit_retained_dry_run_pools(retained_i19_copy)
    assert audit["pool_counts"] == {"real": 32, "conditional": 32, "unconditional": 32}
    assert audit["png_and_pixel_sha_checked"] == 96
    assert audit["selected_reference_ids"][0] == "cs10002930376453395152"
    assert audit["selected_reference_ids"][-1] == "cs10096894922495846065"


def test_recovery_skips_generation_and_source_loading_and_writes_summary(retained_i19_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "_decode_real_pixels", lambda _: pytest.fail("source decoding is forbidden in recovery"))
    monkeypatch.setattr(quality, "generate_dry_run_pool", lambda *_args, **_kwargs: pytest.fail("generation is forbidden in recovery"))
    monkeypatch.setattr(quality, "compute_clean_fid_metrics", lambda *_args, **_kwargs: {"configuration": metric_config(32, 32), "results": {"kid": 1.0, "fid": 2.0}})
    monkeypatch.setattr(quality, "verify_recovery_environment", lambda: {"fixture": "synthetic"})
    summary = quality.run_dry_run_recovery(retained_i19_copy)
    assert summary["recovery"]["generation_executed_in_recovery"] is False
    assert summary["recovery"]["original_source_images_opened_in_recovery"] == 0
    assert (retained_i19_copy / "dryrun_summary.json").is_file()


def test_recovery_rejects_bad_manifest_hash(retained_i19_copy: Path) -> None:
    manifest = retained_i19_copy / "manifests" / "conditional_manifest.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_recovery_rejects_missing_png(retained_i19_copy: Path) -> None:
    next((retained_i19_copy / "conditional").glob("*.png")).unlink()
    with pytest.raises(ValueError, match="exactly 32"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_recovery_rejects_png_sha_mismatch(retained_i19_copy: Path) -> None:
    target = next((retained_i19_copy / "conditional").glob("*.png"))
    target.write_bytes(target.read_bytes() + b"i20")
    with pytest.raises(ValueError, match="PNG SHA256"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_recovery_rejects_pixel_sha_mismatch(retained_i19_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "pixel_sha256", lambda _pixels: "mismatch")
    with pytest.raises(ValueError, match="pixel integrity"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_recovery_rejects_wrong_row_count(retained_i19_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = retained_i19_copy / "manifests" / "real_id_val_manifest.jsonl"
    manifest.write_text("\n".join(manifest.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
    _accept_test_manifest_hashes(retained_i19_copy, monkeypatch)
    with pytest.raises(ValueError, match="exactly 32 rows"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_recovery_rejects_wrong_conditioning_or_seed_namespace(retained_i19_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = retained_i19_copy / "manifests" / "unconditional_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["conditioning_consumed"] = True
    rows[1]["generation_namespace"] = "quality_generation"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _accept_test_manifest_hashes(retained_i19_copy, monkeypatch)
    with pytest.raises(ValueError, match="conditioning flag"):
        quality.audit_retained_dry_run_pools(retained_i19_copy)


def test_final_recovery_requires_explicit_sealed_authorization() -> None:
    with pytest.raises(PermissionError):
        validate_mode("final", confirm_sealed_final_eval=False)
