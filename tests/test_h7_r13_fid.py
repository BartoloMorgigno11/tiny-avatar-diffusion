from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from tiny_avatar_diffusion import ca16_r13_fid_secondary as ca16_fid
from tiny_avatar_diffusion import h7_r13_fid as h7_fid

CA16 = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")


def test_historical_fid_baseline_is_frozen_and_seed_specific() -> None:
    baseline = h7_fid.historical_baseline(CA16, 2026)
    assert baseline["protocol_sha256"] == ca16_fid.PROTOCOL_SHA256
    assert set(baseline["metrics"]) == {"FID_ID", "FID_COMP", "FID_GLOBAL"}
    assert baseline["metrics"]["FID_GLOBAL"]["generated_count"] == 804


def test_verified_pool_helper_uses_frozen_clean_fid_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"ID": 466, "COMP": 338, "GLOBAL": 804}
    real, generated = {}, {}
    for role, count in counts.items():
        for name, target in (("real", real), ("generated", generated)):
            directory = tmp_path / name / role
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index:04d}.png").touch()
            target[role] = directory
    calls: list[tuple[str, object]] = []
    fake_fid = types.SimpleNamespace(
        build_feature_extractor=lambda **kwargs: (
            calls.append(("build", kwargs)) or "model"
        ),
        get_folder_features=lambda path, **kwargs: calls.append((path, kwargs)) or path,
    )
    monkeypatch.setitem(sys.modules, "cleanfid", types.SimpleNamespace(fid=fake_fid))
    monkeypatch.setattr(ca16_fid, "_fid_from_features", lambda real, generated: 7.5)
    result = ca16_fid.compute_fid_from_verified_pools(
        real, generated, device=torch.device("cpu")
    )
    assert set(result) == {"FID_ID", "FID_COMP", "FID_GLOBAL"}
    assert all(value["value"] == 7.5 for value in result.values())
    assert calls[0] == (
        "build",
        {"mode": "clean", "device": torch.device("cpu"), "use_dataparallel": False},
    )
    assert len(calls) == 7
    assert all(call[1]["num_workers"] == 0 for call in calls[1:])
    assert all(call[1]["batch_size"] == 8 for call in calls[1:])


def test_fid_execute_refuses_cpu_before_pool_or_metric_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(h7_fid.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        h7_fid.execute(
            ca16_assets_root=CA16, terminal_checkpoint=Path("missing.pt"), seed=2026
        )
