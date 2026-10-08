from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def pilot():
    path = Path(__file__).parents[1] / "scripts" / "run_ca16_r11_batched_pilot.py"
    spec = importlib.util.spec_from_file_location("r11_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_round_trip_rejects_identity_digest_and_partial(
    tmp_path: Path, pilot
) -> None:
    identity = {"schema": "test", "arm": "ca8", "checkpoint_sha256": "a" * 64}
    path = tmp_path / "reference.pt"
    tensor = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
    pilot._write_cached_reference(path, identity, tensor)
    loaded = pilot._load_cached_reference(path, identity, torch.device("cpu"))
    assert torch.equal(loaded, tensor)
    with pytest.raises(FileExistsError):
        pilot._write_cached_reference(path, identity, tensor)
    with pytest.raises(RuntimeError, match="identity"):
        pilot._load_cached_reference(
            path, {**identity, "arm": "ca8_ca16"}, torch.device("cpu")
        )
    payload = torch.load(path, weights_only=True)
    payload["tensor_sha256"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="corrupt"):
        pilot._load_cached_reference(path, identity, torch.device("cpu"))
    path.unlink()
    path.with_suffix(".pt.tmp").write_bytes(b"interrupted")
    with pytest.raises(RuntimeError, match="incomplete"):
        pilot._load_cached_reference(path, identity, torch.device("cpu"))


def test_writer_loads_real_torch_version_with_weights_only(
    tmp_path: Path, pilot
) -> None:
    path = tmp_path / "real-version.pt"
    identity = {"runtime": {"torch": str(torch.__version__)}}
    tensor = torch.zeros((3, 32, 32), dtype=torch.float32)
    pilot._write_cached_reference(path, identity, tensor)
    assert torch.equal(
        pilot._load_cached_reference(path, identity, torch.device("cpu")), tensor
    )
