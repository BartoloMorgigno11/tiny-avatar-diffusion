import random
from pathlib import Path

import pytest
import torch

from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec
from tiny_avatar_diffusion.conditional_main_training import (
    CA16_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA,
    ConditionalEMA,
    TrainingProgress,
    _nested_exact,
    _new_models,
    _new_models_for_spec,
    audit_paired_initialization,
    build_checkpoint,
    finalize_amp_attempt,
    load_checkpoint,
)


def _checkpoint_parts(spec: CA16RunSpec):
    text = torch.nn.Linear(2, 2)
    denoiser = torch.nn.Linear(2, 2)
    ema = ConditionalEMA(text, denoiser)
    optimizer = torch.optim.AdamW([*text.parameters(), *denoiser.parameters()], lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = torch.Generator().manual_seed(spec.training_seed)
    corruption = torch.Generator().manual_seed(spec.training_seed)
    return text, denoiser, ema, optimizer, scaler, sampler, corruption


@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_paired_initialization_and_runtime_streams_are_equal(seed: int) -> None:
    report = audit_paired_initialization(seed)
    assert all(report.values())


def test_spec_constructs_the_requested_architecture_arm() -> None:
    control = CA16RunSpec(ArchitectureArm.CA8, 2026)
    treatment = CA16RunSpec(ArchitectureArm.CA8_CA16, 2026)
    _, control_denoiser = _new_models_for_spec(torch.device("cpu"), control)
    _, treatment_denoiser = _new_models_for_spec(torch.device("cpu"), treatment)
    assert not hasattr(control_denoiser, "decoder16_image_text_cross_attention")
    assert hasattr(treatment_denoiser, "decoder16_image_text_cross_attention")


def test_legacy_model_constructor_remains_ca8_only() -> None:
    _, denoiser = _new_models(torch.device("cpu"), 2026)
    assert not hasattr(denoiser, "decoder16_image_text_cross_attention")


def test_ca16_checkpoint_identity_round_trip_and_schema_boundaries(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8_CA16, 2026)
    config = spec.main_training_config()
    parts = _checkpoint_parts(spec)
    checkpoint = build_checkpoint(
        config=config,
        progress=TrainingProgress(
            step=1,
            attempted_batch_count=2,
            skipped_step_count=1,
            skipped_attempt_positions=[2],
        ),
        text_encoder=parts[0],
        denoiser=parts[1],
        ema=parts[2],
        optimizer=parts[3],
        scaler=parts[4],
        sampler_generator=parts[5],
        corruption_generator=parts[6],
        validation_history=[],
        experiment_spec=spec,
        execution_git_sha="test-r6-sha",
        execution_branch="test-r6-branch",
    )
    assert checkpoint["schema"] == CA16_CHECKPOINT_SCHEMA
    assert checkpoint["experiment"]["architecture_arm"] == "ca8_ca16"
    assert checkpoint["experiment"]["git_commit"] == "test-r6-sha"
    path = tmp_path / "ca16.pt"
    torch.save(checkpoint, path)

    restored = _checkpoint_parts(spec)
    progress, history = load_checkpoint(
        path,
        config=config,
        text_encoder=restored[0],
        denoiser=restored[1],
        ema=restored[2],
        optimizer=restored[3],
        scaler=restored[4],
        sampler_generator=restored[5],
        corruption_generator=restored[6],
        map_location=torch.device("cpu"),
        experiment_spec=spec,
    )
    assert progress.step == 1
    assert progress.attempted_batch_count == 2
    assert progress.skipped_step_count == 1
    assert progress.skipped_attempt_positions == [2]
    assert history == []

    with pytest.raises(ValueError, match="checkpoint is not compatible"):
        load_checkpoint(
            path,
            config=CA16RunSpec(ArchitectureArm.CA8, 2026).main_training_config(),
            text_encoder=restored[0],
            denoiser=restored[1],
            ema=restored[2],
            optimizer=restored[3],
            scaler=restored[4],
            sampler_generator=restored[5],
            corruption_generator=restored[6],
            map_location=torch.device("cpu"),
            experiment_spec=CA16RunSpec(ArchitectureArm.CA8, 2026),
        )

    legacy = build_checkpoint(
        config=config,
        progress=TrainingProgress(),
        text_encoder=parts[0],
        denoiser=parts[1],
        ema=parts[2],
        optimizer=parts[3],
        scaler=parts[4],
        sampler_generator=parts[5],
        corruption_generator=parts[6],
        validation_history=[],
    )
    assert legacy["schema"] == CHECKPOINT_SCHEMA
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="checkpoint is not compatible"):
        load_checkpoint(
            legacy_path,
            config=config,
            text_encoder=restored[0],
            denoiser=restored[1],
            ema=restored[2],
            optimizer=restored[3],
            scaler=restored[4],
            sampler_generator=restored[5],
            corruption_generator=restored[6],
            map_location=torch.device("cpu"),
            experiment_spec=spec,
        )


def test_ca8_checkpoint_cannot_resume_as_ca8_ca16(tmp_path: Path) -> None:
    control = CA16RunSpec(ArchitectureArm.CA8, 2026)
    treatment = CA16RunSpec(ArchitectureArm.CA8_CA16, 2026)
    parts = _checkpoint_parts(control)
    checkpoint = build_checkpoint(
        config=control.main_training_config(),
        progress=TrainingProgress(),
        text_encoder=parts[0],
        denoiser=parts[1],
        ema=parts[2],
        optimizer=parts[3],
        scaler=parts[4],
        sampler_generator=parts[5],
        corruption_generator=parts[6],
        validation_history=[],
        experiment_spec=control,
        execution_git_sha="test-r6-sha",
    )
    path = tmp_path / "control-ca16-schema.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="checkpoint is not compatible"):
        load_checkpoint(
            path,
            config=treatment.main_training_config(),
            text_encoder=parts[0],
            denoiser=parts[1],
            ema=parts[2],
            optimizer=parts[3],
            scaler=parts[4],
            sampler_generator=parts[5],
            corruption_generator=parts[6],
            map_location=torch.device("cpu"),
            experiment_spec=treatment,
        )


def test_ca16_loader_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8_CA16, 2026)
    parts = _checkpoint_parts(spec)
    checkpoint = build_checkpoint(
        config=spec.main_training_config(),
        progress=TrainingProgress(),
        text_encoder=parts[0],
        denoiser=parts[1],
        ema=parts[2],
        optimizer=parts[3],
        scaler=parts[4],
        sampler_generator=parts[5],
        corruption_generator=parts[6],
        validation_history=[],
        experiment_spec=spec,
        execution_git_sha="test-r6-sha",
    )
    checkpoint["experiment"]["manifests"][0]["sha256"] = "0" * 64
    path = tmp_path / "tampered.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="CA16 experiment identity"):
        load_checkpoint(
            path,
            config=spec.main_training_config(),
            text_encoder=parts[0],
            denoiser=parts[1],
            ema=parts[2],
            optimizer=parts[3],
            scaler=parts[4],
            sampler_generator=parts[5],
            corruption_generator=parts[6],
            map_location=torch.device("cpu"),
            experiment_spec=spec,
        )


class _SequenceScaler:
    def __init__(self, skipped: list[bool]) -> None:
        self.skipped = iter(skipped)
        self.scale = 8.0

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if next(self.skipped):
            self.scale /= 2
        else:
            optimizer.step()

    def update(self) -> None:
        return None


def test_attempt_provenance_tracks_skips_and_legacy_defaults() -> None:
    text = torch.nn.Linear(2, 2)
    denoiser = torch.nn.Linear(2, 2)
    ema = ConditionalEMA(text, denoiser)
    optimizer = torch.optim.SGD([*text.parameters(), *denoiser.parameters()], lr=0.1)
    progress = TrainingProgress()
    scaler = _SequenceScaler([True, False, True])

    for skipped in (True, False, True):
        for parameter in [*text.parameters(), *denoiser.parameters()]:
            parameter.grad = torch.ones_like(parameter)
        assert finalize_amp_attempt(
            scaler=scaler,
            optimizer=optimizer,
            gradients_finite=not skipped,
            progress=progress,
            ema=ema,
            text_encoder=text,
            denoiser=denoiser,
            ema_decay=0.9999,
        ) is skipped

    assert progress.attempted_batch_count == 3
    assert progress.skipped_step_count == 2
    assert progress.step == 1
    assert progress.skipped_attempt_positions == [1, 3]
    assert progress.attempted_batch_count == progress.step + progress.skipped_step_count


def test_oldest_legacy_checkpoint_attempt_fields_use_compatibility_fallback(tmp_path: Path) -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8, 2026)
    parts = _checkpoint_parts(spec)
    checkpoint = build_checkpoint(
        config=spec.main_training_config(),
        progress=TrainingProgress(step=3, skipped_step_count=2),
        text_encoder=parts[0],
        denoiser=parts[1],
        ema=parts[2],
        optimizer=parts[3],
        scaler=parts[4],
        sampler_generator=parts[5],
        corruption_generator=parts[6],
        validation_history=[],
    )
    del checkpoint["progress"]["attempted_batch_count"]
    del checkpoint["progress"]["skipped_attempt_positions"]
    del checkpoint["progress"]["skipped_step_count"]
    path = tmp_path / "old-ca16.pt"
    torch.save(checkpoint, path)
    restored = _checkpoint_parts(spec)
    progress, _ = load_checkpoint(
        path,
        config=spec.main_training_config(),
        text_encoder=restored[0],
        denoiser=restored[1],
        ema=restored[2],
        optimizer=restored[3],
        scaler=restored[4],
        sampler_generator=restored[5],
        corruption_generator=restored[6],
        map_location=torch.device("cpu"),
    )
    assert progress.attempted_batch_count == 3
    assert progress.skipped_step_count == 0
    assert progress.skipped_attempt_positions == []


def test_pairing_audit_preserves_caller_rng_state(monkeypatch) -> None:
    random.seed(314159)
    torch.manual_seed(271828)
    python_state = random.getstate()
    cpu_state = torch.get_rng_state().clone()

    audit_paired_initialization(2026)

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), cpu_state)

    def fail_after_rng_mutation(*_args, **_kwargs):
        random.seed(1)
        torch.manual_seed(2)
        raise RuntimeError("controlled audit failure")

    monkeypatch.setattr(
        "tiny_avatar_diffusion.conditional_main_training._new_models_for_spec",
        fail_after_rng_mutation,
    )
    with pytest.raises(RuntimeError, match="controlled audit failure"):
        audit_paired_initialization(2026)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), cpu_state)


def test_nested_exact_compares_tensor_sequences_without_structural_tensor_equality() -> None:
    left = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    equal = [tensor.clone() for tensor in left]
    unequal = [left[0].clone(), torch.tensor([3, 5])]

    assert _nested_exact(left, equal)
    assert not _nested_exact(left, unequal)
