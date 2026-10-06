"""Declarative configuration for the frozen CA16 ablation matrix."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tiny_avatar_diffusion.conditional_main_training import MainTrainingConfig

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "ca16_ablation"
SUPPORTED_SEEDS = (2026, 2027, 2028)


class ArchitectureArm(str, Enum):
    """The two frozen CA16 architecture arms."""

    CA8 = "ca8"
    CA8_CA16 = "ca8_ca16"


@dataclass(frozen=True)
class ManifestReference:
    """A frozen R2 artifact and its required content hash."""

    path: str
    sha256: str

    def verify(self, *, root: Path = ROOT) -> None:
        artifact = root / self.path
        if not artifact.is_file():
            raise FileNotFoundError(f"frozen manifest is missing: {artifact}")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != self.sha256:
            raise ValueError(
                f"frozen manifest hash mismatch for {self.path}: "
                f"expected {self.sha256}, got {digest}"
            )


FROZEN_MANIFESTS = (
    ManifestReference(
        "data/processed/ca16_eval_v1/robust_conditioning_manifest.json",
        "e3319f65f67c19ae8b458f7450ff8e6b86e5913e0fd542e9c0c3009802451aec",
    ),
    ManifestReference(
        "data/processed/ca16_eval_v1/checkpoint_selection_manifest.json",
        "192191235af7992b8c332019b7e948414ba0b87c5f8df8d70f044b74e85e3d40",
    ),
    ManifestReference(
        "data/processed/ca16_eval_v1/quality_guardrail_manifest.json",
        "e34db5251d8949a5fb37a905f6e9e4791602047f3153a9a9908d14d9cd85e84f",
    ),
    ManifestReference(
        "data/processed/ca16_eval_v1/manifest_audit.json",
        "b693521cc5e1430b6f5d44acd3bca4bae11eb67746fc06e6e43b22f21b69c35f",
    ),
)


@dataclass(frozen=True)
class CA16RunSpec:
    """One declarative, deterministic run in the six-run CA16 matrix."""

    architecture_arm: ArchitectureArm
    training_seed: int

    def __post_init__(self) -> None:
        try:
            arm = ArchitectureArm(self.architecture_arm)
        except ValueError as error:
            raise ValueError(f"unknown architecture arm: {self.architecture_arm!r}") from error
        if arm is not self.architecture_arm:
            object.__setattr__(self, "architecture_arm", arm)
        if self.training_seed not in SUPPORTED_SEEDS:
            raise ValueError(f"unsupported training seed: {self.training_seed}")

    @property
    def run_id(self) -> str:
        return f"{EXPERIMENT_ID}_{self.architecture_arm.value}_seed{self.training_seed}"

    @property
    def use_decoder16_cross_attention(self) -> bool:
        return self.architecture_arm is ArchitectureArm.CA8_CA16

    @property
    def manifests(self) -> tuple[ManifestReference, ...]:
        return FROZEN_MANIFESTS

    @property
    def text_encoder(self) -> Mapping[str, int | float | str]:
        return {
            "initialization": "from scratch",
            "d_model": 64,
            "layers": 2,
            "heads": 4,
            "ffn": 256,
            "dropout": 0.1,
        }

    @property
    def frozen_protocol(self) -> Mapping[str, object]:
        return {
            "dataset_variant": "10k",
            "resolution": (32, 32),
            "channels": 3,
            "physical_batch_size": 32,
            "gradient_accumulation": 1,
            "learning_rate": 2e-4,
            "weight_decay": 0.0,
            "diffusion_timesteps": 1_000,
            "schedule": "linear",
            "prediction_target": "epsilon",
            "precision": "AMP_FP16",
            "validation_interval": 1_000,
            "milestones": (5_000, 10_000, 15_000, 20_000),
            "budget_steps": 20_000,
            "ema_decay": 0.9999,
            "denoiser_dropout": 0.0,
            "cfg": False,
            "conditioning_dropout": False,
        }

    def main_training_config(self) -> MainTrainingConfig:
        """Derive the existing frozen V1 training configuration for this run."""
        return MainTrainingConfig(run_id=self.run_id, seed=self.training_seed)

    def validate_config(self, config: MainTrainingConfig) -> None:
        """Reject a training configuration that does not identify this run."""
        if config != self.main_training_config():
            raise ValueError("training config does not match the requested CA16 run spec")

    def verify_manifests(self) -> None:
        """Verify the configured frozen R2 artifacts before a run starts."""
        verify_frozen_manifests()

    def provenance(
        self,
        *,
        execution_git_sha: str,
        branch: str | None = None,
        environment: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build future-run provenance without assuming the execution commit."""
        if not execution_git_sha:
            raise ValueError("execution_git_sha is required")
        return {
            "run_id": self.run_id,
            "experiment_id": EXPERIMENT_ID,
            "architecture_arm": self.architecture_arm.value,
            "use_decoder16_cross_attention": self.use_decoder16_cross_attention,
            "training_seed": self.training_seed,
            "frozen_training": dict(self.frozen_protocol),
            "text_encoder": dict(self.text_encoder),
            "manifests": [
                {"path": reference.path, "sha256": reference.sha256}
                for reference in self.manifests
            ],
            "git_commit": execution_git_sha,
            "branch": branch,
            "environment": dict(environment or {}),
        }


def ca16_run_matrix() -> tuple[CA16RunSpec, ...]:
    """Return the complete matrix in canonical paired/interleaved order."""
    matrix = tuple(
        CA16RunSpec(arm, seed)
        for seed in SUPPORTED_SEEDS
        for arm in (ArchitectureArm.CA8, ArchitectureArm.CA8_CA16)
    )
    if len({spec.run_id for spec in matrix}) != len(matrix):
        raise AssertionError("CA16 run matrix contains duplicate run IDs")
    return matrix


def verify_frozen_manifests(*, root: Path = ROOT) -> None:
    """Verify every configured R2 manifest against its required SHA256."""
    for reference in FROZEN_MANIFESTS:
        reference.verify(root=root)
