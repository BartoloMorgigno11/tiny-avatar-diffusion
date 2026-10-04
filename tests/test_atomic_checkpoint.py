from __future__ import annotations

import copy
import errno
from pathlib import Path

import pytest
import torch

import tiny_avatar_diffusion.conditional_main_training as checkpoint_module


def _payload(counter: int) -> dict[str, object]:
    return {"counter": counter, "tensor": torch.tensor([counter, counter + 1])}


def _temp_files(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_save_creates_and_replaces_loadable_destination(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    checkpoint_module.atomic_save_checkpoint(path, _payload(1))
    assert torch.load(path, weights_only=False)["counter"] == 1
    checkpoint_module.atomic_save_checkpoint(path, _payload(2))
    loaded = torch.load(path, weights_only=False)
    assert loaded["counter"] == 2 and torch.equal(loaded["tensor"], torch.tensor([2, 3]))
    assert not _temp_files(path)


def test_atomic_save_ignores_stale_temp_and_does_not_mutate_payload(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    stale = path.parent / f".{path.name}.stale.tmp"
    stale.write_bytes(b"unrelated stale file")
    payload = _payload(7)
    before = copy.deepcopy(payload)
    checkpoint_module.atomic_save_checkpoint(path, payload)
    assert torch.equal(payload["tensor"], before["tensor"]) and payload["counter"] == before["counter"]
    assert stale.read_bytes() == b"unrelated stale file"
    assert torch.load(path, weights_only=False)["counter"] == 7


def test_atomic_save_retries_first_transient_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.pt"
    real_replace, calls = checkpoint_module.os.replace, []

    def fail_once(source: Path, destination: Path) -> None:
        calls.append(Path(source))
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "simulated sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_once)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    checkpoint_module.atomic_save_checkpoint(path, _payload(3))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert torch.load(path, weights_only=False)["counter"] == 3
    assert not _temp_files(path)


def test_atomic_save_retries_multiple_transient_failures_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.pt"
    real_replace, calls = checkpoint_module.os.replace, 0

    def fail_three_times(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise PermissionError(13, "simulated access denied")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_three_times)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    checkpoint_module.atomic_save_checkpoint(path, _payload(4))
    assert calls == 4 and torch.load(path, weights_only=False)["counter"] == 4


def test_atomic_save_retry_exhaustion_preserves_old_destination_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.pt"
    checkpoint_module.atomic_save_checkpoint(path, _payload(10))

    def always_fail(source: Path, destination: Path) -> None:
        raise PermissionError(13, "simulated persistent access denied")

    monkeypatch.setattr(checkpoint_module.os, "replace", always_fail)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="after 6 attempts"):
        checkpoint_module.atomic_save_checkpoint(path, _payload(11))
    assert torch.load(path, weights_only=False)["counter"] == 10
    assert not _temp_files(path)


def test_atomic_save_propagates_immediate_non_transient_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "latest.pt"
    checkpoint_module.atomic_save_checkpoint(path, _payload(10))
    calls: list[Path] = []
    non_transient = OSError(errno.EIO, "simulated disk error")

    def fail_non_transient(source: Path, destination: Path) -> None:
        calls.append(Path(source))
        raise non_transient

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_non_transient)
    with pytest.raises(OSError) as raised:
        checkpoint_module.atomic_save_checkpoint(path, _payload(11))
    assert raised.value is non_transient
    assert len(calls) == 1
    assert torch.load(path, weights_only=False)["counter"] == 10
    assert not _temp_files(path)


def test_atomic_save_propagates_non_transient_error_after_transient_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "latest.pt"
    checkpoint_module.atomic_save_checkpoint(path, _payload(10))
    calls = 0
    non_transient = OSError(errno.EIO, "simulated disk error")

    def fail_transient_then_non_transient(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError(errno.EACCES, "simulated sharing violation")
        raise non_transient

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_transient_then_non_transient)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    with pytest.raises(OSError) as raised:
        checkpoint_module.atomic_save_checkpoint(path, _payload(11))
    assert raised.value is non_transient
    assert calls == 2
    assert torch.load(path, weights_only=False)["counter"] == 10
    assert not _temp_files(path)
