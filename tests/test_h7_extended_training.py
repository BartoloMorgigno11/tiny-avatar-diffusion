from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tiny_avatar_diffusion
import tiny_avatar_diffusion.h7_extended_training as h7
from tiny_avatar_diffusion import conditional_lr_pilot
from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec
from tiny_avatar_diffusion.conditional_main_training import ConditionalEMA
from tiny_avatar_diffusion.debug_tiny_overfit import preload_images
from tiny_avatar_diffusion.h7_extended_training import (
    TARGET_STEP,
    CA16AssetSource,
    H7ContinuationSpec,
    current_data_provenance,
    h7_command,
    verify_source_checkpoint,
)


def _asset_root(tmp_path: Path) -> CA16AssetSource:
    (tmp_path / ".git").mkdir(parents=True)
    (tmp_path / "results" / "runs").mkdir(parents=True)
    return CA16AssetSource(tmp_path)


def test_h7_is_limited_to_ca16_seeds_and_terminal_budget() -> None:
    assert H7ContinuationSpec(2026).target_successful_steps == TARGET_STEP
    with pytest.raises(ValueError):
        H7ContinuationSpec(1)
    with pytest.raises(ValueError):
        H7ContinuationSpec(2026, 20_000)
    assert (
        H7ContinuationSpec(2027).source_spec.architecture_arm
        is ArchitectureArm.CA8_CA16
    )


def test_h7_source_verification_fails_closed_before_deserialization(
    tmp_path: Path,
) -> None:
    assets = _asset_root(tmp_path / "assets")
    source = assets.root / "source.pt"
    source.write_bytes(b"not a checkpoint")
    spec = H7ContinuationSpec(2026)
    # A mismatching digest must reject the source without accepting filename identity.
    with pytest.raises(ValueError, match="SHA256"):
        verify_source_checkpoint(spec, assets, source)


def test_r10_config_stays_20k() -> None:
    config = CA16RunSpec(ArchitectureArm.CA8_CA16, 2028).main_training_config()
    assert asdict(config)["budget_steps"] == 20_000
    with pytest.raises(ValueError):
        type(config)(run_id="x", budget_steps=40_000)


def test_h7_output_is_isolated(tmp_path: Path) -> None:
    result = H7ContinuationSpec(2026).output_dir()
    assert result.parent.name == "h7_ca16_40k_v1"


def test_wrong_source_root_and_output_escape_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        CA16AssetSource(Path(h7.ROOT))
    with pytest.raises(ValueError, match="fixed"):
        H7ContinuationSpec(2026).output_dir(tmp_path)


def test_safe_command_is_argument_vector(tmp_path: Path) -> None:
    command = h7_command(H7ContinuationSpec(2026), _asset_root(tmp_path / "assets"))
    assert command[:2] == ("-m", "tiny_avatar_diffusion.h7_extended_training")
    assert "--preflight" in command and "--execute" not in command


def test_missing_assets_and_import_identity_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        current_data_provenance(_asset_root(tmp_path / "assets"))
    assert (
        Path(tiny_avatar_diffusion.__file__).resolve().parents[1]
        == Path(h7.__file__).resolve().parents[1]
    )


def test_text_image_loader_accepts_an_explicit_external_image_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [{"token_ids": [1] * 7, "attention_mask": [1] * 7}]
    observed: list[Path] = []
    monkeypatch.setattr(conditional_lr_pilot, "_load_text_records", lambda *_: records)
    monkeypatch.setattr(
        conditional_lr_pilot,
        "preload_images",
        lambda rows, *, root: observed.append(root) or torch.zeros((1, 3, 32, 32)),
    )
    split = conditional_lr_pilot.load_text_image_split(
        tmp_path / "train.jsonl", "TRAIN", image_root=tmp_path
    )
    assert split.images.shape == (1, 3, 32, 32)
    assert observed == [tmp_path]


def test_image_preload_uses_the_explicit_archive_root(tmp_path: Path) -> None:
    ca16 = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")
    manifest = ca16 / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl"
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    relative_archive = Path(record["source_image_path"].split("::", 1)[0])
    assert (ca16 / relative_archive).is_file()

    # A separate runner worktree without the archive must not silently fall back.
    with pytest.raises(FileNotFoundError):
        preload_images([record], root=tmp_path / "separate-h7-root")

    # This also models an integrated repository whose explicit root contains it.
    image = preload_images([record], root=ca16)
    assert image.shape == (1, 3, 32, 32)


def test_h7_runner_binds_training_images_to_the_explicit_ca16_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca16 = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")

    class LoaderCalled(Exception):
        pass

    def loader(path: Path, split: str, *, image_root: Path) -> None:
        assert (
            path
            == ca16 / "data" / "processed" / "dr01_v1" / "text" / "10k" / "train.jsonl"
        )
        assert split == "TRAIN"
        assert image_root == ca16
        raise LoaderCalled

    monkeypatch.setattr(h7, "preflight", lambda *_: {})
    monkeypatch.setattr(h7, "verify_source_checkpoint", lambda *_: {})
    monkeypatch.setattr(
        H7ContinuationSpec,
        "output_dir",
        lambda spec, root: tmp_path / spec.run_id,
    )
    monkeypatch.setattr(
        h7, "_restore_source", lambda *_: (None,) * 7 + (SimpleNamespace(step=20_000),)
    )
    monkeypatch.setattr(h7, "load_text_image_split", loader)
    monkeypatch.setattr(h7.torch.cuda, "is_available", lambda: True)
    with pytest.raises(LoaderCalled):
        h7.run(H7ContinuationSpec(2026), assets=CA16AssetSource(ca16), execute=True)


def test_restored_ema_is_aligned_once_without_reinitializing_values() -> None:
    text, denoiser = nn.Linear(2, 2), nn.BatchNorm1d(2)
    ema = ConditionalEMA(text, denoiser)
    ema.text_encoder = {
        key: value.detach().clone().to(torch.float64)
        for key, value in ema.text_encoder.items()
    }
    ema.denoiser = {
        key: value.detach().clone().to(torch.float64)
        for key, value in ema.denoiser.items()
    }
    before = ema.state_dict()
    rng_before = torch.get_rng_state().clone()
    h7._align_restored_ema(ema, text, denoiser, torch.device("cpu"))
    after = ema.state_dict()
    assert set(after) == {"text_encoder", "conditional_unet"}
    for shadows, model in ((ema.text_encoder, text), (ema.denoiser, denoiser)):
        assert set(shadows) == set(model.state_dict())
        assert all(value.device.type == "cpu" for value in shadows.values())
    for component in after:
        for key in after[component]:
            assert after[component][key].dtype == before[component][key].dtype
            assert torch.equal(after[component][key], before[component][key])
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_restored_ema_fails_closed_when_a_component_is_incomplete() -> None:
    text, denoiser = nn.Linear(2, 2), nn.BatchNorm1d(2)
    ema = ConditionalEMA(text, denoiser)
    ema.text_encoder.pop(next(iter(ema.text_encoder)))
    with pytest.raises(ValueError, match="EMA state is incomplete"):
        h7._align_restored_ema(ema, text, denoiser, torch.device("cpu"))
