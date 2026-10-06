import pytest

from tiny_avatar_diffusion.ca16_experiment import (
    FROZEN_MANIFESTS,
    ArchitectureArm,
    CA16RunSpec,
    ca16_run_matrix,
    verify_frozen_manifests,
)


def test_matrix_has_exact_canonical_order_and_unique_ids() -> None:
    matrix = ca16_run_matrix()
    assert [(spec.architecture_arm.value, spec.training_seed) for spec in matrix] == [
        ("ca8", 2026),
        ("ca8_ca16", 2026),
        ("ca8", 2027),
        ("ca8_ca16", 2027),
        ("ca8", 2028),
        ("ca8_ca16", 2028),
    ]
    assert [spec.run_id for spec in matrix] == [
        "ca16_ablation_ca8_seed2026",
        "ca16_ablation_ca8_ca16_seed2026",
        "ca16_ablation_ca8_seed2027",
        "ca16_ablation_ca8_ca16_seed2027",
        "ca16_ablation_ca8_seed2028",
        "ca16_ablation_ca8_ca16_seed2028",
    ]
    assert len({spec.run_id for spec in matrix}) == 6


def test_architecture_mapping_and_seed_validation() -> None:
    assert not CA16RunSpec(ArchitectureArm.CA8, 2026).use_decoder16_cross_attention
    assert CA16RunSpec(ArchitectureArm.CA8_CA16, 2026).use_decoder16_cross_attention
    with pytest.raises(ValueError, match="unknown architecture arm"):
        CA16RunSpec("unknown", 2026)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported training seed"):
        CA16RunSpec(ArchitectureArm.CA8, 2025)


def test_all_specs_share_frozen_protocol_and_derive_v1_config() -> None:
    matrix = ca16_run_matrix()
    protocols = [spec.frozen_protocol for spec in matrix]
    assert all(protocol == protocols[0] for protocol in protocols)
    assert protocols[0] == {
        "dataset_variant": "10k",
        "resolution": (32, 32),
        "channels": 3,
        "physical_batch_size": 32,
        "gradient_accumulation": 1,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "diffusion_timesteps": 1000,
        "schedule": "linear",
        "prediction_target": "epsilon",
        "precision": "AMP_FP16",
        "validation_interval": 1000,
        "milestones": (5000, 10000, 15000, 20000),
        "budget_steps": 20000,
        "ema_decay": 0.9999,
        "denoiser_dropout": 0.0,
        "cfg": False,
        "conditioning_dropout": False,
    }
    assert all(
        (
            config.learning_rate,
            config.physical_batch_size,
            config.gradient_accumulation,
            config.weight_decay,
            config.diffusion_timesteps,
            config.schedule,
            config.precision,
            config.validation_interval,
            config.milestones,
            config.budget_steps,
            config.ema_decay,
        )
        == (2e-4, 32, 1, 0.0, 1000, "linear", "AMP_FP16", 1000, (5000, 10000, 15000, 20000), 20000, 0.9999)
        for config in (spec.main_training_config() for spec in matrix)
    )


def test_frozen_protocol_metadata_matches_each_derived_training_config() -> None:
    fields = (
        "dataset_variant",
        "resolution",
        "channels",
        "physical_batch_size",
        "gradient_accumulation",
        "learning_rate",
        "weight_decay",
        "diffusion_timesteps",
        "schedule",
        "precision",
        "validation_interval",
        "milestones",
        "budget_steps",
        "ema_decay",
    )
    for spec in ca16_run_matrix():
        main = spec.main_training_config()
        metadata = spec.frozen_protocol
        assert all(getattr(main, field) == metadata[field] for field in fields)
        assert main.seed == spec.training_seed
        assert main.run_id == spec.run_id
        assert metadata["prediction_target"] == "epsilon"
        assert metadata["denoiser_dropout"] == 0.0
        assert metadata["cfg"] is False
        assert metadata["conditioning_dropout"] is False


def test_manifest_references_are_identical_and_match_files() -> None:
    matrix = ca16_run_matrix()
    assert all(spec.manifests == FROZEN_MANIFESTS for spec in matrix)
    verify_frozen_manifests()


def test_provenance_requires_dynamic_execution_commit_and_records_required_fields() -> None:
    spec = CA16RunSpec(ArchitectureArm.CA8_CA16, 2026)
    with pytest.raises(ValueError, match="execution_git_sha"):
        spec.provenance(execution_git_sha="")
    provenance = spec.provenance(
        execution_git_sha="future-r6-r7-r8-sha",
        branch="future-experiment",
        environment={"python": "3.12"},
    )
    assert provenance["git_commit"] == "future-r6-r7-r8-sha"
    assert provenance["run_id"] == "ca16_ablation_ca8_ca16_seed2026"
    assert provenance["architecture_arm"] == "ca8_ca16"
    assert provenance["environment"] == {"python": "3.12"}


def test_declarative_configuration_has_no_runtime_side_effects(monkeypatch) -> None:
    import torch

    forbidden = (
        "tiny_avatar_diffusion.unet.ConditionalTinyUNet",
        "tiny_avatar_diffusion.text_conditioning.TextConditioningEncoder",
        "tiny_avatar_diffusion.conditional_main_training.run_main_training",
        "tiny_avatar_diffusion.conditional_sampling.conditional_sample",
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("declarative configuration invoked forbidden runtime code")

    for target in forbidden:
        module_name, attribute = target.rsplit(".", 1)
        monkeypatch.setattr(f"{module_name}.{attribute}", fail)
    monkeypatch.setattr(torch, "Generator", fail)
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)

    matrix = ca16_run_matrix()
    for spec in matrix:
        assert spec.architecture_arm in {
            ArchitectureArm.CA8,
            ArchitectureArm.CA8_CA16,
        }
        assert spec.main_training_config().seed == spec.training_seed
        assert spec.manifests
        assert spec.provenance(execution_git_sha="future-execution-sha")["run_id"] == spec.run_id
