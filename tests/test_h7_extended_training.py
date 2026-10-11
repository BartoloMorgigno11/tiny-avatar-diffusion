from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

import tiny_avatar_diffusion
import tiny_avatar_diffusion.h7_extended_training as h7
from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec
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
        CA16AssetSource(Path(__file__).resolve().parents[1])
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
