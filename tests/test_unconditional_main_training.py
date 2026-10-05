from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import tiny_avatar_diffusion.unconditional_main_training as unconditional_training
from tiny_avatar_diffusion.conditional_main_training import (
    TrainingProgress,
    atomic_save_checkpoint,
)
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.unconditional_main_training import (
    CANONICAL_RUN_ID,
    CHECKPOINT_SCHEMA,
    EXPECTED_PARAMETERS,
    MILESTONES,
    TRAIN_MANIFEST,
    UnconditionalEMA,
    UnconditionalMainConfig,
    build_unconditional_checkpoint,
    canonical_checkpoint_ready,
    finalize_unconditional_amp_attempt,
    load_unconditional_checkpoint,
    unconditional_diffusion_loss,
)
from tiny_avatar_diffusion.unet import TinyUNet


class FakeScaler:
    def __init__(self, *, skipped: bool) -> None:
        self.scale_value = 8.0
        self.skipped = skipped

    def get_scale(self) -> float:
        return self.scale_value

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.skipped:
            optimizer.step()

    def update(self) -> None:
        if self.skipped:
            self.scale_value = 4.0


def _components() -> tuple[TinyUNet, torch.optim.AdamW, UnconditionalEMA, TrainingProgress]:
    torch.manual_seed(7)
    model = TinyUNet()
    return model, torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0), UnconditionalEMA(model), TrainingProgress()


def test_d17_architecture_and_frozen_configuration():
    config = UnconditionalMainConfig()
    model = TinyUNet()
    assert config.run_id == CANONICAL_RUN_ID
    assert asdict(config) == {
        "run_id": CANONICAL_RUN_ID, "dataset_variant": "10k", "train_count": 8388,
        "resolution": (32, 32), "channels": 3, "physical_batch_size": 32,
        "learning_rate": 2e-4, "weight_decay": 0.0, "seed": 2026,
        "diffusion_timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02,
        "precision": "AMP_FP16", "successful_steps": 20000, "ema_decay": 0.9999,
        "milestones": MILESTONES,
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert model(torch.randn(1, 3, 32, 32), torch.tensor([500])).shape == (1, 3, 32, 32)
    assert not any("text" in name or "cross_attention" in name for name, _ in model.named_parameters())
    assert TRAIN_MANIFEST.as_posix().endswith("data/processed/dr01_v1/text/10k/train.jsonl")
    with pytest.raises(ValueError):
        UnconditionalMainConfig(learning_rate=1e-4)


def test_unconditional_loss_uses_epsilon_mse_and_adamw_frozen_controls():
    model, optimizer, _, _ = _components()
    schedule = LinearDDPMSchedule()
    generator = torch.Generator().manual_seed(3)
    loss = unconditional_diffusion_loss(model, schedule, torch.randn(2, 3, 32, 32), generator=generator)
    assert torch.isfinite(loss)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_successful_and_skipped_amp_attempts_have_correct_ema_accounting():
    model, optimizer, ema, progress = _components()
    initial_ema = ema.state_dict()
    loss = model(torch.randn(1, 3, 32, 32), torch.tensor([1])).square().mean()
    loss.backward()
    assert not finalize_unconditional_amp_attempt(scaler=FakeScaler(skipped=False), optimizer=optimizer, gradients_finite=True, progress=progress, ema=ema, model=model, ema_decay=0.9999)
    assert progress.step == 1 and progress.skipped_step_count == 0
    assert any(not torch.equal(ema.state_dict()[name], initial_ema[name]) for name in initial_ema)
    before_skip = ema.state_dict()
    assert finalize_unconditional_amp_attempt(scaler=FakeScaler(skipped=True), optimizer=optimizer, gradients_finite=True, progress=progress, ema=ema, model=model, ema_decay=0.9999)
    assert progress.step == 1 and progress.skipped_step_count == 1
    assert all(torch.equal(ema.state_dict()[name], before_skip[name]) for name in before_skip)


def test_checkpoint_round_trip_restores_raw_ema_optimizer_config_and_progress(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(unconditional_training, "_git_head", lambda: "a" * 40)
    config = UnconditionalMainConfig()
    model, optimizer, ema, progress = _components()
    progress.step, progress.images_seen, progress.batch_offset = 2, 64, 32
    progress.permutation = torch.arange(32)
    progress.train_losses = [1.0, 0.5]
    sampler = torch.Generator().manual_seed(9)
    corruption = torch.Generator().manual_seed(10)
    scaler = torch.amp.GradScaler("cpu")
    checkpoint = build_unconditional_checkpoint(config=config, progress=progress, model=model, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler, corruption_generator=corruption)
    path = tmp_path / "resume.pt"
    atomic_save_checkpoint(path, checkpoint)
    restored_model, restored_optimizer, restored_ema, _ = _components()
    restored_scaler = torch.amp.GradScaler("cpu")
    restored_progress = load_unconditional_checkpoint(torch.load(path, weights_only=False), config=config, model=restored_model, ema=restored_ema, optimizer=restored_optimizer, scaler=restored_scaler, sampler_generator=torch.Generator(), corruption_generator=torch.Generator())
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["config"] == asdict(config)
    assert checkpoint["provenance"]["split_role"] == "TRAIN"
    assert checkpoint["provenance"]["git_commit"] == "a" * 40
    assert restored_progress.step == 2 and restored_progress.images_seen == 64
    assert all(torch.equal(model.state_dict()[key], restored_model.state_dict()[key]) for key in model.state_dict())
    assert all(torch.equal(ema.state_dict()[key], restored_ema.state_dict()[key]) for key in ema.state_dict())
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]


def test_d18_only_admits_ema_at_exactly_20000_successful_updates():
    progress = TrainingProgress(step=19_999)
    assert not canonical_checkpoint_ready(progress)
    progress.step = 20_000
    assert canonical_checkpoint_ready(progress)
    progress.step = 20_001
    assert not canonical_checkpoint_ready(progress)


def test_synthetic_resume_matches_uninterrupted_without_counter_or_ema_duplication(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(unconditional_training, "_git_head", lambda: "a" * 40)
    def update(model, optimizer, ema, progress, generator):
        optimizer.zero_grad(set_to_none=True)
        loss = unconditional_diffusion_loss(model, LinearDDPMSchedule(), torch.full((1, 3, 32, 32), 0.25), generator=generator)
        loss.backward()
        assert not finalize_unconditional_amp_attempt(scaler=FakeScaler(skipped=False), optimizer=optimizer, gradients_finite=True, progress=progress, ema=ema, model=model, ema_decay=0.9999)

    full_model, full_optimizer, full_ema, full_progress = _components()
    full_generator = torch.Generator().manual_seed(123)
    update(full_model, full_optimizer, full_ema, full_progress, full_generator)
    update(full_model, full_optimizer, full_ema, full_progress, full_generator)

    interrupted_model, interrupted_optimizer, interrupted_ema, interrupted_progress = _components()
    interrupted_generator = torch.Generator().manual_seed(123)
    update(interrupted_model, interrupted_optimizer, interrupted_ema, interrupted_progress, interrupted_generator)
    config = UnconditionalMainConfig()
    checkpoint = build_unconditional_checkpoint(config=config, progress=interrupted_progress, model=interrupted_model, ema=interrupted_ema, optimizer=interrupted_optimizer, scaler=torch.amp.GradScaler("cpu"), sampler_generator=torch.Generator().manual_seed(1), corruption_generator=interrupted_generator)
    resumed_model, resumed_optimizer, resumed_ema, _ = _components()
    resumed_generator = torch.Generator()
    resumed_progress = load_unconditional_checkpoint(checkpoint, config=config, model=resumed_model, ema=resumed_ema, optimizer=resumed_optimizer, scaler=torch.amp.GradScaler("cpu"), sampler_generator=torch.Generator(), corruption_generator=resumed_generator)
    update(resumed_model, resumed_optimizer, resumed_ema, resumed_progress, resumed_generator)

    assert resumed_progress.step == full_progress.step == 2
    assert resumed_progress.skipped_step_count == full_progress.skipped_step_count == 0
    assert all(torch.equal(full_model.state_dict()[key], resumed_model.state_dict()[key]) for key in full_model.state_dict())
    assert all(torch.equal(full_ema.state_dict()[key], resumed_ema.state_dict()[key]) for key in full_ema.state_dict())
