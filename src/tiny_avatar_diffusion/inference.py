"""Shared inference facade for the frozen Tiny Avatar Diffusion V1 models."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from tiny_avatar_diffusion.captions import generate_caption
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import (
    LinearDDPMSchedule,
    ObservedSample,
    SamplingSnapshot,
    normalize_snapshot_timesteps,
    unconditional_sample,
)
from tiny_avatar_diffusion.evaluator import (
    ARCHITECTURE_ID,
    CLASS_COUNTS,
    PREPROCESSING_ID,
    AttributeEvaluator,
    preprocess_cached,
)
from tiny_avatar_diffusion.semantic_mapping import (
    CANONICAL_ATTRIBUTE_ORDER,
    SEMANTIC_MAPPING,
    semantic_value,
)
from tiny_avatar_diffusion.text_conditioning import (
    TextConditioningConfig,
    TextConditioningEncoder,
)
from tiny_avatar_diffusion.tokenizer import UNK_ID, WhitespaceTokenizer
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, TinyUNet

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"
VOCABULARY_PATH = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "vocab.json"

CONDITIONAL_CHECKPOINT = CHECKPOINT_ROOT / "conditional_main_ema_step20000.pt"
UNCONDITIONAL_CHECKPOINT = CHECKPOINT_ROOT / "unconditional_main_ema_step20000.pt"
EVALUATOR_CHECKPOINT = CHECKPOINT_ROOT / "dr04_evaluator_epoch41.pt"

CONDITIONAL_SHA256 = "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"
UNCONDITIONAL_SHA256 = "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e"
EVALUATOR_SHA256 = "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"

CONDITIONAL_PARAMETER_COUNT = 4_723_907
UNCONDITIONAL_PARAMETER_COUNT = 4_595_779
EVALUATOR_PARAMETER_COUNT = 293_064


@dataclass(frozen=True)
class PromptAttributes:
    """Raw DR01 IDs for the four controlled attributes."""

    face_shape: int
    glasses: int
    hair_color: int
    face_color: int

    def __post_init__(self) -> None:
        for attribute in CANONICAL_ATTRIBUTE_ORDER:
            value = getattr(self, attribute)
            if type(value) is not int:
                raise ValueError(f"{attribute} must be an integer DR01 value")
            semantic_value(attribute, value)

    @property
    def caption(self) -> str:
        return generate_caption(asdict(self))


@dataclass(frozen=True)
class ConditionalInferenceBundle:
    text_encoder: TextConditioningEncoder
    model: ConditionalTinyUNet
    schedule: LinearDDPMSchedule
    tokenizer: WhitespaceTokenizer
    device: torch.device
    requested_device: str = "auto"


@dataclass(frozen=True)
class UnconditionalInferenceBundle:
    model: TinyUNet
    schedule: LinearDDPMSchedule
    device: torch.device
    requested_device: str = "auto"


@dataclass(frozen=True)
class EvaluatorInferenceBundle:
    model: AttributeEvaluator
    device: torch.device
    requested_device: str = "auto"


@dataclass(frozen=True)
class GenerationProvenance:
    """JSON-compatible schema-v1 provenance for one in-memory generation."""

    schema_version: int
    generation: dict[str, Any]
    model: dict[str, Any]
    diffusion: dict[str, Any]
    runtime: dict[str, Any]
    software: dict[str, Any]
    observability: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    """In-memory generated sample with observations and separate diagnostics."""

    image: Image.Image
    final_tensor: torch.Tensor
    provenance: GenerationProvenance
    initial_x_T: torch.Tensor
    snapshots: tuple[SamplingSnapshot, ...] = ()
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairingMetadata:
    comparison_label: str
    seed: int
    shared_initial_x_T: bool
    equivalent_post_x_T_generator_state: bool


@dataclass(frozen=True)
class PairedGenerationResult:
    conditional: GenerationResult
    unconditional: GenerationResult
    pairing: PairingMetadata


def resolve_device(requested: Literal["auto", "cuda", "cpu"] | str = "auto") -> torch.device:
    """Resolve the frozen public device policy without silent CUDA fallback."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return torch.device("cuda")
    raise ValueError("device must be one of: auto, cuda, cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_artifact(
    path: Path,
    *,
    expected_sha256: str,
    artifact_type: str,
    model_role: str,
    expected_parameter_count: int,
    selected_field: str,
    selected_value: int,
    require_ema: bool,
    expected_architecture: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical checkpoint is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Canonical checkpoint SHA256 mismatch: {path.name}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError("Canonical checkpoint must contain a dictionary artifact")
    required = {
        "artifact_format_version",
        "artifact_type",
        "model_role",
        "model_state_dict",
        "architecture",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_git_commit",
        "expected_parameter_count",
        selected_field,
    }
    if not required.issubset(artifact) or artifact.get("artifact_format_version") != 1:
        raise ValueError("Invalid canonical inference checkpoint schema")
    if artifact.get("artifact_type") != artifact_type:
        raise ValueError("Canonical checkpoint artifact type mismatch")
    if artifact.get("model_role") != model_role:
        raise ValueError("Canonical checkpoint model role mismatch")
    if artifact.get(selected_field) != selected_value:
        raise ValueError(f"Canonical checkpoint {selected_field} mismatch")
    if artifact.get("expected_parameter_count") != expected_parameter_count:
        raise ValueError("Canonical checkpoint parameter-count identity mismatch")
    if require_ema and artifact.get("weight_variant") != "EMA":
        raise ValueError("Canonical generator checkpoint must contain EMA weights")
    if artifact.get("architecture") != expected_architecture:
        raise ValueError("Canonical checkpoint architecture metadata mismatch")
    if not isinstance(artifact.get("model_state_dict"), dict) or not artifact["model_state_dict"]:
        raise TypeError("Canonical checkpoint model_state_dict must be a dictionary")
    source_path = artifact.get("source_checkpoint_path")
    if not isinstance(source_path, str) or not source_path or Path(source_path).is_absolute():
        raise ValueError("Canonical checkpoint source path must be repository-relative")
    source_sha256 = artifact.get("source_checkpoint_sha256")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("Canonical checkpoint source SHA256 metadata is malformed")
    source_commit = artifact.get("source_git_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Canonical checkpoint source Git commit metadata is malformed")
    forbidden = ("optimizer", "scheduler", "scaler", "rng", "progress", "history", "dataloader")
    if any(any(token in key.lower() for token in forbidden) for key in artifact):
        raise ValueError("Resume-only metadata found in canonical inference checkpoint")
    return artifact


def _parameter_count(*modules: torch.nn.Module) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _load_tokenizer(path: Path = VOCABULARY_PATH) -> WhitespaceTokenizer:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Canonical vocabulary is missing: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Canonical vocabulary is malformed: {path}") from error
    if (
        artifact.get("schema") != "tiny_avatar_diffusion.dr01_train_vocabulary"
        or artifact.get("version") != "dr01_v1"
        or artifact.get("dataset_variant") != "10k"
        or artifact.get("source_split") != "TRAIN"
        or artifact.get("vocab_size") != 45
        or not isinstance(artifact.get("token_to_id"), dict)
    ):
        raise ValueError("Canonical vocabulary identity or schema mismatch")
    return WhitespaceTokenizer(artifact["token_to_id"])


def load_conditional_bundle(device: str = "auto") -> ConditionalInferenceBundle:
    """Load the only canonical V1 conditional checkpoint on CPU, then move it."""
    resolved = resolve_device(device)
    expected_architecture = {
        "text_encoder": {
            "class": "TextConditioningEncoder",
            "config": asdict(TextConditioningConfig()),
        },
        "conditional_unet": {"class": "ConditionalTinyUNet", "dropout": 0.0},
    }
    artifact = _load_artifact(
        CONDITIONAL_CHECKPOINT,
        expected_sha256=CONDITIONAL_SHA256,
        artifact_type="canonical_inference_checkpoint",
        model_role="conditional_main",
        expected_parameter_count=CONDITIONAL_PARAMETER_COUNT,
        selected_field="selected_step",
        selected_value=20_000,
        require_ema=True,
        expected_architecture=expected_architecture,
    )
    text_encoder = TextConditioningEncoder(TextConditioningConfig())
    model = ConditionalTinyUNet(dropout=0.0)
    state = artifact["model_state_dict"]
    try:
        text_encoder.load_state_dict(state["text_encoder"], strict=True)
        model.load_state_dict(state["conditional_unet"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("Canonical conditional state_dict is incompatible") from error
    if _parameter_count(text_encoder, model) != CONDITIONAL_PARAMETER_COUNT:
        raise ValueError("Loaded conditional model parameter count mismatch")
    schedule = LinearDDPMSchedule()
    text_encoder.to(resolved).eval()
    model.to(resolved).eval()
    schedule.to(resolved).eval()
    return ConditionalInferenceBundle(
        text_encoder, model, schedule, _load_tokenizer(), resolved, device
    )


def load_unconditional_bundle(device: str = "auto") -> UnconditionalInferenceBundle:
    """Load the only canonical V1 unconditional checkpoint on CPU, then move it."""
    resolved = resolve_device(device)
    expected_architecture = {"tiny_unet": {"class": "TinyUNet", "dropout": 0.0}}
    artifact = _load_artifact(
        UNCONDITIONAL_CHECKPOINT,
        expected_sha256=UNCONDITIONAL_SHA256,
        artifact_type="canonical_inference_checkpoint",
        model_role="unconditional_main",
        expected_parameter_count=UNCONDITIONAL_PARAMETER_COUNT,
        selected_field="selected_step",
        selected_value=20_000,
        require_ema=True,
        expected_architecture=expected_architecture,
    )
    model = TinyUNet(dropout=0.0)
    try:
        model.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("Canonical unconditional state_dict is incompatible") from error
    if _parameter_count(model) != UNCONDITIONAL_PARAMETER_COUNT:
        raise ValueError("Loaded unconditional model parameter count mismatch")
    schedule = LinearDDPMSchedule()
    model.to(resolved).eval()
    schedule.to(resolved).eval()
    return UnconditionalInferenceBundle(model, schedule, resolved, device)


def load_evaluator_bundle(device: str = "auto") -> EvaluatorInferenceBundle:
    """Load the frozen DR04 evaluator inference artifact."""
    resolved = resolve_device(device)
    expected_architecture = {
        "class": "AttributeEvaluator",
        "architecture_id": ARCHITECTURE_ID,
        "preprocessing_id": PREPROCESSING_ID,
        "class_counts": CLASS_COUNTS,
    }
    artifact = _load_artifact(
        EVALUATOR_CHECKPOINT,
        expected_sha256=EVALUATOR_SHA256,
        artifact_type="canonical_evaluator_inference_checkpoint",
        model_role="dr04_evaluator",
        expected_parameter_count=EVALUATOR_PARAMETER_COUNT,
        selected_field="selected_epoch",
        selected_value=41,
        require_ema=False,
        expected_architecture=expected_architecture,
    )
    model = AttributeEvaluator()
    try:
        model.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("Canonical evaluator state_dict is incompatible") from error
    if _parameter_count(model) != EVALUATOR_PARAMETER_COUNT:
        raise ValueError("Loaded evaluator parameter count mismatch")
    model.to(resolved).eval()
    return EvaluatorInferenceBundle(model, resolved, device)


def validate_raw_caption(caption: str) -> PromptAttributes:
    """Accept only an exact canonical English DR01 caption."""
    if not isinstance(caption, str):
        raise TypeError("raw caption must be a string")
    tokens = caption.split(" ")
    if len(tokens) != 5 or tokens[0] != "avatar" or "" in tokens:
        raise ValueError("raw caption must exactly match the five-token canonical grammar")
    values: dict[str, int] = {}
    for attribute, token in zip(CANONICAL_ATTRIBUTE_ORDER, tokens[1:], strict=True):
        matches = [
            raw_id
            for raw_id, value in SEMANTIC_MAPPING[attribute].items()
            if value.canonical_model_token == token
        ]
        if len(matches) != 1:
            raise ValueError(f"invalid canonical token for {attribute}: {token!r}")
        values[attribute] = matches[0]
    attributes = PromptAttributes(**values)
    if caption != attributes.caption:
        raise ValueError("raw caption is not in exact canonical form")
    return attributes


def _prompt_caption(
    attributes: PromptAttributes | None, raw_caption: str | None
) -> tuple[str, PromptAttributes]:
    if (attributes is None) == (raw_caption is None):
        raise ValueError("provide exactly one of attributes or raw_caption")
    if attributes is not None:
        if not isinstance(attributes, PromptAttributes):
            raise ValueError("attributes must be PromptAttributes")
        return attributes.caption, attributes
    parsed = validate_raw_caption(raw_caption)  # type: ignore[arg-type]
    return parsed.caption, parsed


def _generator(device: torch.device, seed: int) -> torch.Generator:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return torch.Generator(device=device).manual_seed(seed)


def tensor_to_rgb_image(sample: torch.Tensor) -> Image.Image:
    """Apply the frozen generated-tensor to uint8 RGB conversion."""
    if sample.shape != (3, 32, 32):
        raise ValueError("sample must have shape [3, 32, 32]")
    pixels = (
        ((sample.detach().cpu().to(torch.float32).clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def pixel_content_sha256(image: Image.Image) -> str:
    """Hash canonical RGB uint8 pixels in contiguous row-major HWC byte order."""
    if image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("pixel hashing requires one RGB 32x32 image")
    return hashlib.sha256(image.tobytes()).hexdigest()


def _git_metadata() -> tuple[str | None, bool | None]:
    """Return the base revision and dirty state, or null values on any failure."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            return None, None
        porcelain = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return revision, bool(porcelain.strip())


@lru_cache(maxsize=1)
def _software_metadata() -> dict[str, str | bool | None]:
    try:
        package_version = version("tiny-avatar-diffusion")
    except PackageNotFoundError:
        package_version = None
    git_commit, git_dirty = _git_metadata()
    return {
        "package_version": package_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _attribute_metadata(attributes: PromptAttributes) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attribute in CANONICAL_ATTRIBUTE_ORDER:
        raw_id = getattr(attributes, attribute)
        value = semantic_value(attribute, raw_id)
        metadata[attribute] = {
            "raw_id": raw_id,
            "label": value.canonical_label,
            "model_token": value.canonical_model_token,
        }
    return metadata


def _diffusion_metadata(schedule: LinearDDPMSchedule) -> dict[str, Any]:
    if type(schedule) is not LinearDDPMSchedule:
        raise ValueError("canonical inference requires LinearDDPMSchedule")
    if schedule.betas.dtype != torch.float32:
        raise ValueError("canonical inference schedule must use FP32 coefficients")
    return {
        "image_size": [32, 32],
        "timesteps": int(schedule.betas.shape[0]),
        "schedule_identity": "linear_beta_schedule",
        "beta_start": float(schedule.betas[0].detach().cpu().item()),
        "beta_end": float(schedule.betas[-1].detach().cpu().item()),
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
        "sampler_identity": "ancestral_ddpm",
        "clip_denoised": True,
        "precision": "fp32",
    }


def _model_metadata(model_variant: Literal["conditional", "unconditional"]) -> dict[str, Any]:
    if model_variant == "conditional":
        checkpoint = CONDITIONAL_CHECKPOINT
        digest = CONDITIONAL_SHA256
    else:
        checkpoint = UNCONDITIONAL_CHECKPOINT
        digest = UNCONDITIONAL_SHA256
    return {
        "checkpoint_identity": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": digest,
        "selected_step": 20_000,
        "weights_variant": "EMA",
    }


def _sample_parts(
    output: torch.Tensor | ObservedSample,
) -> tuple[torch.Tensor, tuple[SamplingSnapshot, ...]]:
    if isinstance(output, ObservedSample):
        return output.sample, output.snapshots
    return output, ()


def _build_generation_result(
    *,
    sample_output: torch.Tensor | ObservedSample,
    initial_x_T: torch.Tensor,
    model_variant: Literal["conditional", "unconditional"],
    canonical_prompt: str | None,
    attributes: PromptAttributes | None,
    seed: int,
    schedule: LinearDDPMSchedule,
    requested_device: str,
    resolved_device: torch.device,
    requested_snapshots: tuple[int, ...],
) -> GenerationResult:
    sample, snapshots = _sample_parts(sample_output)
    if sample.shape != (1, 3, 32, 32) or sample.dtype != torch.float32:
        raise RuntimeError("canonical sampler returned an invalid final tensor")
    final_tensor = sample[0].detach().clone().cpu()
    image = tensor_to_rgb_image(final_tensor)
    pixel_hash = pixel_content_sha256(image)
    observability = None
    if requested_snapshots:
        observability = {
            "requested_snapshot_timesteps": list(requested_snapshots),
            "captured_snapshot_timesteps": [item.timestep for item in snapshots],
        }
    provenance = GenerationProvenance(
        schema_version=1,
        generation={
            "model_variant": model_variant,
            "canonical_prompt": canonical_prompt,
            "attributes": _attribute_metadata(attributes) if attributes is not None else None,
            "seed": seed,
            "pixel_content_sha256": pixel_hash,
        },
        model=_model_metadata(model_variant),
        diffusion=_diffusion_metadata(schedule),
        runtime={
            "requested_device": requested_device,
            "resolved_device": str(resolved_device),
        },
        software=dict(_software_metadata()),
        observability=observability,
    )
    return GenerationResult(
        image=image,
        final_tensor=final_tensor,
        provenance=provenance,
        initial_x_T=initial_x_T.detach().clone().cpu(),
        snapshots=snapshots,
    )


def _conditional_from_noise(
    bundle: ConditionalInferenceBundle,
    *,
    seed: int,
    caption: str,
    attributes: PromptAttributes,
    x_T: torch.Tensor,
    generator: torch.Generator,
    snapshot_timesteps: tuple[int, ...],
) -> GenerationResult:
    encoded = bundle.tokenizer.encode(caption)
    if UNK_ID in encoded.token_ids:
        raise ValueError("canonical caption encoded to <unk>")
    input_ids = torch.tensor([encoded.token_ids], dtype=torch.long, device=bundle.device)
    attention_mask = torch.tensor(
        [encoded.attention_mask], dtype=torch.long, device=bundle.device
    )
    output = conditional_sample(
        bundle.schedule,
        bundle.text_encoder,
        bundle.model,
        input_ids,
        attention_mask,
        x_T=x_T,
        generator=generator,
        clip_denoised=True,
        snapshot_timesteps=snapshot_timesteps or None,
    )
    return _build_generation_result(
        sample_output=output,
        initial_x_T=x_T,
        model_variant="conditional",
        canonical_prompt=caption,
        attributes=attributes,
        seed=seed,
        schedule=bundle.schedule,
        requested_device=bundle.requested_device,
        resolved_device=bundle.device,
        requested_snapshots=snapshot_timesteps,
    )


def _unconditional_from_noise(
    bundle: UnconditionalInferenceBundle,
    *,
    seed: int,
    x_T: torch.Tensor,
    generator: torch.Generator,
    snapshot_timesteps: tuple[int, ...],
) -> GenerationResult:
    output = unconditional_sample(
        bundle.schedule,
        bundle.model,
        x_T=x_T,
        generator=generator,
        clip_denoised=True,
        snapshot_timesteps=snapshot_timesteps or None,
    )
    return _build_generation_result(
        sample_output=output,
        initial_x_T=x_T,
        model_variant="unconditional",
        canonical_prompt=None,
        attributes=None,
        seed=seed,
        schedule=bundle.schedule,
        requested_device=bundle.requested_device,
        resolved_device=bundle.device,
        requested_snapshots=snapshot_timesteps,
    )


def generate_conditional(
    bundle: ConditionalInferenceBundle,
    *,
    seed: int,
    attributes: PromptAttributes | None = None,
    raw_caption: str | None = None,
    snapshot_timesteps: Iterable[int] | None = None,
) -> GenerationResult:
    """Generate one conditional result from structured input or a strict caption."""
    caption, parsed_attributes = _prompt_caption(attributes, raw_caption)
    requested = normalize_snapshot_timesteps(
        snapshot_timesteps, bundle.schedule.betas.shape[0]
    )
    generator = _generator(bundle.device, seed)
    x_T = torch.randn(
        (1, 3, 32, 32), device=bundle.device, dtype=torch.float32, generator=generator
    )
    return _conditional_from_noise(
        bundle,
        seed=seed,
        caption=caption,
        attributes=parsed_attributes,
        x_T=x_T,
        generator=generator,
        snapshot_timesteps=requested,
    )


def generate_unconditional(
    bundle: UnconditionalInferenceBundle,
    *,
    seed: int,
    snapshot_timesteps: Iterable[int] | None = None,
) -> GenerationResult:
    """Generate one unconditional result with the canonical V1 sampler."""
    requested = normalize_snapshot_timesteps(
        snapshot_timesteps, bundle.schedule.betas.shape[0]
    )
    generator = _generator(bundle.device, seed)
    x_T = torch.randn(
        (1, 3, 32, 32), device=bundle.device, dtype=torch.float32, generator=generator
    )
    return _unconditional_from_noise(
        bundle,
        seed=seed,
        x_T=x_T,
        generator=generator,
        snapshot_timesteps=requested,
    )


def generate_paired(
    conditional_bundle: ConditionalInferenceBundle,
    unconditional_bundle: UnconditionalInferenceBundle,
    *,
    seed: int,
    attributes: PromptAttributes | None = None,
    raw_caption: str | None = None,
    snapshot_timesteps: Iterable[int] | None = None,
) -> PairedGenerationResult:
    """Run a qualitative pair from shared ``x_T`` and cloned post-``x_T`` RNG state."""
    if conditional_bundle.device != unconditional_bundle.device:
        raise ValueError("paired generation requires both models on the same device")
    if not torch.equal(
        conditional_bundle.schedule.betas, unconditional_bundle.schedule.betas
    ):
        raise ValueError("paired generation requires identical diffusion schedules")
    caption, parsed_attributes = _prompt_caption(attributes, raw_caption)
    requested = normalize_snapshot_timesteps(
        snapshot_timesteps, conditional_bundle.schedule.betas.shape[0]
    )
    root_generator = _generator(conditional_bundle.device, seed)
    shared_x_T = torch.randn(
        (1, 3, 32, 32),
        device=conditional_bundle.device,
        dtype=torch.float32,
        generator=root_generator,
    )
    post_x_T_state = root_generator.get_state().clone()
    conditional_generator = torch.Generator(device=conditional_bundle.device)
    conditional_generator.set_state(post_x_T_state.clone())
    unconditional_generator = torch.Generator(device=conditional_bundle.device)
    unconditional_generator.set_state(post_x_T_state.clone())

    conditional_result = _conditional_from_noise(
        conditional_bundle,
        seed=seed,
        caption=caption,
        attributes=parsed_attributes,
        x_T=shared_x_T.clone(),
        generator=conditional_generator,
        snapshot_timesteps=requested,
    )
    unconditional_result = _unconditional_from_noise(
        unconditional_bundle,
        seed=seed,
        x_T=shared_x_T.clone(),
        generator=unconditional_generator,
        snapshot_timesteps=requested,
    )
    return PairedGenerationResult(
        conditional=conditional_result,
        unconditional=unconditional_result,
        pairing=PairingMetadata(
            comparison_label="QUALITATIVE PAIRED COMPARISON",
            seed=seed,
            shared_initial_x_T=True,
            equivalent_post_x_T_generator_state=True,
        ),
    )


def evaluator_diagnostic(
    bundle: EvaluatorInferenceBundle,
    image: Image.Image,
    *,
    requested: PromptAttributes | None = None,
) -> dict[str, Any]:
    """Return per-head labels and uncalibrated evaluator scores for one image."""
    if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("evaluator input must be one RGB 32x32 PIL image")
    tensor = preprocess_cached(image).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(tensor)
    attributes: dict[str, Any] = {}
    matches = 0
    for attribute in CANONICAL_ATTRIBUTE_ORDER:
        head_logits = logits[attribute][0].detach().cpu().to(torch.float32)
        scores = torch.softmax(head_logits, dim=0)
        predicted_id = int(torch.argmax(head_logits).item())
        predicted = semantic_value(attribute, predicted_id)
        requested_id = getattr(requested, attribute) if requested is not None else None
        match = predicted_id == requested_id if requested_id is not None else None
        if match:
            matches += 1
        attributes[attribute] = {
            "predicted_raw_id": predicted_id,
            "predicted_label": predicted.canonical_label,
            "predicted_model_token": predicted.canonical_model_token,
            "raw_logits": head_logits.tolist(),
            "uncalibrated_evaluator_scores": [
                {
                    "raw_id": raw_id,
                    "label": semantic_value(attribute, raw_id).canonical_label,
                    "model_token": semantic_value(attribute, raw_id).canonical_model_token,
                    "score": float(scores[raw_id].item()),
                }
                for raw_id in range(len(scores))
            ],
            "requested_raw_id": requested_id,
            "requested_label": (
                semantic_value(attribute, requested_id).canonical_label
                if requested_id is not None
                else None
            ),
            "match": match,
        }
    return {
        "score_label": "UNCALIBRATED EVALUATOR SCORE",
        "attributes": attributes,
        "attributes_matched": matches if requested is not None else None,
        "attributes_total": 4 if requested is not None else None,
    }
