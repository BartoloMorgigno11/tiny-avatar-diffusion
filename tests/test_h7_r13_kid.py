from pathlib import Path

import pytest

from tiny_avatar_diffusion import h7_r13_kid as kid

CA16 = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")


def test_historical_kid_baseline_is_authoritative_and_complete():
    baseline = kid.historical_baseline(CA16, 2026)
    assert set(baseline["metrics"]) == {"KID_ID", "KID_COMP", "KID_GLOBAL"}
    assert baseline["metrics"]["KID_COMP"]["num_subsets"] == 100


def test_kid_execute_refuses_cpu_without_pool_or_generation(monkeypatch):
    monkeypatch.setattr(kid.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        kid.execute(
            ca16_assets_root=CA16, terminal_checkpoint=Path("missing.pt"), seed=2026
        )
