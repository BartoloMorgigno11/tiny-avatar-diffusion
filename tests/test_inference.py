from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from tiny_avatar_diffusion import inference
from tiny_avatar_diffusion.diffusion import (
    LinearDDPMSchedule,
    ObservedSample,
    unconditional_sample,
)
from tiny_avatar_diffusion.inference import (
    CONDITIONAL_PARAMETER_COUNT,
    EVALUATOR_PARAMETER_COUNT,
    UNCONDITIONAL_PARAMETER_COUNT,
    ConditionalInferenceBundle,
    EvaluatorInferenceBundle,
    PromptAttributes,
    UnconditionalInferenceBundle,
    evaluator_diagnostic,
    generate_conditional,
    generate_paired,
    generate_unconditional,
    load_conditional_bundle,
    load_evaluator_bundle,
    load_unconditional_bundle,
    pixel_content_sha256,
    resolve_device,
    tensor_to_rgb_image,
    validate_raw_caption,
)


class ZeroNoiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.timesteps: list[int] = []

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.timesteps.append(int(timesteps[0].item()))
        return torch.zeros_like(image) + self.anchor * 0


class FixedEvaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        counts = {"face_shape": 7, "glasses": 12, "hair_color": 10, "face_color": 11}
        outputs = {}
        for index, (attribute, count) in enumerate(counts.items()):
            logits = torch.arange(count, dtype=torch.float32, device=image.device)
            logits[index] = 100.0
            outputs[attribute] = logits.unsqueeze(0)
        return outputs


def _manual_frozen_unconditional_loop(
    schedule: LinearDDPMSchedule, model: nn.Module, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.randn((1, 3, 32, 32), dtype=torch.float32, generator=generator)
    with torch.no_grad():
        for value in range(schedule.betas.shape[0] - 1, -1, -1):
            timesteps = torch.full((1,), value, dtype=torch.long)
            noise = torch.randn(image.shape, dtype=torch.float32, generator=generator)
            image = schedule.p_sample_from_noise(
                image,
                timesteps,
                model(image, timesteps),
                noise=noise,
                clip_denoised=True,
            )["sample"]
    return image


def test_device_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cuda") == torch.device("cuda")
    with pytest.raises(ValueError, match="auto, cuda, cpu"):
        resolve_device("mps")


def test_structured_prompt_and_strict_raw_caption_round_trip() -> None:
    attributes = PromptAttributes(face_shape=5, glasses=6, hair_color=9, face_color=0)
    expected = (
        "avatar face_shape_wide glasses_thick_rectangular_glasses "
        "hair_color_silver face_color_deep_brown"
    )
    assert attributes.caption == expected
    assert validate_raw_caption(expected) == attributes


@pytest.mark.parametrize(
    "caption",
    [
        "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver",
        "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown extra",
        "avatar glasses_thick_rectangular_glasses face_shape_wide hair_color_silver face_color_deep_brown",
        "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_marrone",
        "avatar face_shape_wide <unk> hair_color_silver face_color_deep_brown",
        "Avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown",
        "avatar  face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown",
    ],
)
def test_strict_raw_caption_rejects_noncanonical_input(caption: str) -> None:
    with pytest.raises(ValueError):
        validate_raw_caption(caption)


@pytest.mark.parametrize(
    "values",
    [
        {"face_shape": 7, "glasses": 0, "hair_color": 0, "face_color": 0},
        {"face_shape": 0, "glasses": -1, "hair_color": 0, "face_color": 0},
        {"face_shape": 0, "glasses": 0, "hair_color": "red", "face_color": 0},
    ],
)
def test_structured_prompt_rejects_invalid_attribute(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PromptAttributes(**values)  # type: ignore[arg-type]


def _artifact(path: Path, **overrides: object) -> tuple[Path, str]:
    payload = {
        "artifact_format_version": 1,
        "artifact_type": "canonical_inference_checkpoint",
        "model_role": "unconditional_main",
        "selected_step": 20_000,
        "weight_variant": "EMA",
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "architecture": {},
        "source_checkpoint_path": "source.pt",
        "source_checkpoint_sha256": "0" * 64,
        "source_git_commit": "0" * 40,
        "expected_parameter_count": 1,
    }
    payload.update(overrides)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _load_synthetic(path: Path, digest: str) -> dict[str, object]:
    return inference._load_artifact(
        path,
        expected_sha256=digest,
        artifact_type="canonical_inference_checkpoint",
        model_role="unconditional_main",
        expected_parameter_count=1,
        selected_field="selected_step",
        selected_value=20_000,
        require_ema=True,
        expected_architecture={},
    )


def test_checkpoint_hash_schema_and_role_validation(tmp_path: Path) -> None:
    path, digest = _artifact(tmp_path / "valid.pt")
    assert _load_synthetic(path, digest)["model_role"] == "unconditional_main"
    with pytest.raises(ValueError, match="SHA256"):
        _load_synthetic(path, "f" * 64)

    wrong_role, wrong_role_digest = _artifact(
        tmp_path / "wrong-role.pt", model_role="conditional_main"
    )
    with pytest.raises(ValueError, match="model role"):
        _load_synthetic(wrong_role, wrong_role_digest)

    bad_schema, bad_schema_digest = _artifact(
        tmp_path / "bad-schema.pt", artifact_format_version=2
    )
    with pytest.raises(ValueError, match="schema"):
        _load_synthetic(bad_schema, bad_schema_digest)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"artifact_type": "wrong"}, "artifact type"),
        ({"selected_step": 19_999}, "selected_step"),
        ({"weight_variant": "RAW"}, "EMA"),
        ({"architecture": {"wrong": True}}, "architecture"),
        ({"expected_parameter_count": 2}, "parameter-count"),
        ({"model_state_dict": None}, "model_state_dict"),
        ({"source_checkpoint_sha256": "bad"}, "source SHA256"),
        ({"source_git_commit": "bad"}, "source Git commit"),
        ({"source_checkpoint_path": ""}, "source path"),
    ],
)
def test_checkpoint_metadata_validation(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    path, digest = _artifact(tmp_path / "invalid.pt", **overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        _load_synthetic(path, digest)


def test_checkpoint_rejects_missing_state_dict(tmp_path: Path) -> None:
    path, _ = _artifact(tmp_path / "missing-state.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["model_state_dict"]
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="schema"):
        _load_synthetic(path, digest)


def test_unconditional_loader_rejects_incompatible_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference,
        "_load_artifact",
        lambda *_args, **_kwargs: {"model_state_dict": {"not_a_model_weight": torch.zeros(1)}},
    )
    with pytest.raises(ValueError, match="state_dict is incompatible"):
        load_unconditional_bundle("cpu")


def test_real_canonical_checkpoints_load_on_cpu_with_frozen_parameter_counts() -> None:
    conditional = load_conditional_bundle("cpu")
    unconditional = load_unconditional_bundle("cpu")
    evaluator = load_evaluator_bundle("cpu")
    assert conditional.device == unconditional.device == evaluator.device == torch.device("cpu")
    assert sum(p.numel() for p in conditional.text_encoder.parameters()) + sum(
        p.numel() for p in conditional.model.parameters()
    ) == CONDITIONAL_PARAMETER_COUNT
    assert sum(p.numel() for p in unconditional.model.parameters()) == UNCONDITIONAL_PARAMETER_COUNT
    assert sum(p.numel() for p in evaluator.model.parameters()) == EVALUATOR_PARAMETER_COUNT
    assert not conditional.text_encoder.training
    assert not conditional.model.training
    assert not unconditional.model.training
    assert not evaluator.model.training


def test_unconditional_sampler_matches_frozen_loop_and_preserves_mode() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=3)
    model = ZeroNoiseModel().train()
    expected = _manual_frozen_unconditional_loop(schedule, model, seed=23)
    actual = unconditional_sample(
        schedule,
        model,
        generator=torch.Generator(device="cpu").manual_seed(23),
        clip_denoised=True,
    )
    assert torch.equal(actual, expected)
    assert model.timesteps == [2, 1, 0, 2, 1, 0]
    assert model.training
    assert not actual.requires_grad


def test_unconditional_sampler_with_explicit_x_t_preserves_rng_and_mode() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=3)
    model = ZeroNoiseModel().eval()
    x_t = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    reference_generator = torch.Generator(device="cpu").manual_seed(31)
    expected = x_t
    with torch.no_grad():
        for value in range(2, -1, -1):
            timesteps = torch.full((1,), value, dtype=torch.long)
            noise = torch.randn(expected.shape, generator=reference_generator)
            expected = schedule.p_sample_from_noise(
                expected,
                timesteps,
                model(expected, timesteps),
                noise=noise,
                clip_denoised=True,
            )["sample"]
    expected_next = torch.randn(1, generator=reference_generator)

    actual_generator = torch.Generator(device="cpu").manual_seed(31)
    actual = unconditional_sample(
        schedule,
        model,
        x_T=x_t,
        generator=actual_generator,
        clip_denoised=True,
    )
    actual_next = torch.randn(1, generator=actual_generator)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_next, expected_next)
    assert not model.training


def test_high_level_unconditional_generation_is_seed_deterministic() -> None:
    bundle = UnconditionalInferenceBundle(
        model=ZeroNoiseModel(),
        schedule=LinearDDPMSchedule(num_timesteps=2),
        device=torch.device("cpu"),
    )
    first = generate_unconditional(bundle, seed=7)
    repeated = generate_unconditional(bundle, seed=7)
    different = generate_unconditional(bundle, seed=8)
    assert first.image.mode == "RGB" and first.image.size == (32, 32)
    assert first.image.tobytes() == repeated.image.tobytes()
    assert first.image.tobytes() != different.image.tobytes()
    reference = unconditional_sample(
        bundle.schedule,
        bundle.model,
        generator=torch.Generator().manual_seed(7),
        clip_denoised=True,
    )
    assert isinstance(reference, torch.Tensor)
    assert torch.equal(first.final_tensor, reference[0])


def test_high_level_conditional_generation_wires_prompt_seed_and_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_sample(
        schedule: LinearDDPMSchedule,
        text_encoder: nn.Module,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        x_T: torch.Tensor,
        generator: torch.Generator,
        clip_denoised: bool,
        snapshot_timesteps: object,
    ) -> torch.Tensor:
        captured.update(
            schedule=schedule,
            text_encoder=text_encoder,
            model=model,
            input_ids=input_ids.clone(),
            attention_mask=attention_mask.clone(),
            x_T=x_T.clone(),
            seed=generator.initial_seed(),
            generator_state=generator.get_state().clone(),
            clip_denoised=clip_denoised,
            snapshot_timesteps=snapshot_timesteps,
        )
        return torch.zeros(1, 3, 32, 32)

    monkeypatch.setattr(inference, "conditional_sample", fake_sample)
    text_encoder, model = ZeroNoiseModel(), ZeroNoiseModel()
    bundle = ConditionalInferenceBundle(
        text_encoder=text_encoder,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        schedule=LinearDDPMSchedule(num_timesteps=2),
        tokenizer=inference._load_tokenizer(),
        device=torch.device("cpu"),
    )
    attributes = PromptAttributes(face_shape=5, glasses=6, hair_color=9, face_color=0)
    encoded = bundle.tokenizer.encode(attributes.caption)
    result = generate_conditional(bundle, seed=123, attributes=attributes)
    assert captured["schedule"] is bundle.schedule
    assert captured["text_encoder"] is text_encoder
    assert captured["model"] is model
    assert captured["seed"] == 123
    assert captured["clip_denoised"] is True
    assert captured["snapshot_timesteps"] is None
    assert torch.equal(captured["input_ids"], torch.tensor([encoded.token_ids]))
    assert torch.equal(captured["attention_mask"], torch.tensor([encoded.attention_mask]))
    assert torch.equal(captured["x_T"], result.initial_x_T)
    expected_generator = torch.Generator().manual_seed(123)
    torch.randn(1, 3, 32, 32, generator=expected_generator)
    assert torch.equal(captured["generator_state"], expected_generator.get_state())
    assert result.image.mode == "RGB" and result.image.size == (32, 32)
    assert result.image.getpixel((0, 0)) == (128, 128, 128)


def test_tensor_to_rgb_image_matches_canonical_conversion() -> None:
    sample = torch.zeros(3, 32, 32)
    sample[:, 0, 0] = -2.0
    sample[:, 0, 1] = 2.0
    image = tensor_to_rgb_image(sample)
    assert image.mode == "RGB" and image.size == (32, 32)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 0)) == (255, 255, 255)
    assert image.getpixel((2, 0)) == (128, 128, 128)


def test_unconditional_snapshots_are_observational_and_normalized() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=3)
    model = ZeroNoiseModel().eval()
    x_T = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(40))
    plain_generator = torch.Generator().manual_seed(41)
    observed_generator = torch.Generator().manual_seed(41)
    plain = unconditional_sample(
        schedule, model, x_T=x_T, generator=plain_generator, clip_denoised=True
    )
    observed = unconditional_sample(
        schedule,
        model,
        x_T=x_T,
        generator=observed_generator,
        clip_denoised=True,
        snapshot_timesteps=[0, 2, 2],
    )
    assert isinstance(plain, torch.Tensor)
    assert isinstance(observed, ObservedSample)
    assert torch.equal(observed.sample, plain)
    assert torch.equal(observed_generator.get_state(), plain_generator.get_state())
    assert [item.timestep for item in observed.snapshots] == [2, 0]
    final_before_mutation = observed.sample.clone()
    for item in observed.snapshots:
        assert item.predicted_x_start.device.type == "cpu"
        assert not item.predicted_x_start.requires_grad
        item.predicted_x_start.add_(100)
    assert torch.equal(observed.sample, final_before_mutation)

    empty = unconditional_sample(
        schedule,
        model,
        x_T=x_T,
        generator=torch.Generator().manual_seed(41),
        snapshot_timesteps=[],
    )
    assert isinstance(empty, torch.Tensor)
    assert torch.equal(empty, plain)


@pytest.mark.parametrize("timesteps", [[-1], [3], [1.5]])
def test_unconditional_snapshots_reject_invalid_timesteps(timesteps: list[object]) -> None:
    with pytest.raises((TypeError, ValueError), match="snapshot timesteps"):
        unconditional_sample(
            LinearDDPMSchedule(num_timesteps=3),
            ZeroNoiseModel(),
            generator=torch.Generator().manual_seed(1),
            snapshot_timesteps=timesteps,  # type: ignore[arg-type]
        )


def _paired_test_bundles() -> tuple[ConditionalInferenceBundle, UnconditionalInferenceBundle]:
    schedule_conditional = LinearDDPMSchedule()
    schedule_unconditional = LinearDDPMSchedule()
    return (
        ConditionalInferenceBundle(
            text_encoder=ZeroNoiseModel(),  # type: ignore[arg-type]
            model=ZeroNoiseModel(),  # type: ignore[arg-type]
            schedule=schedule_conditional,
            tokenizer=inference._load_tokenizer(),
            device=torch.device("cpu"),
            requested_device="auto",
        ),
        UnconditionalInferenceBundle(
            model=ZeroNoiseModel(),
            schedule=schedule_unconditional,
            device=torch.device("cpu"),
            requested_device="auto",
        ),
    )


def test_paired_generation_shares_x_t_and_clones_post_x_t_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, torch.Tensor]] = {}

    def conditional_stub(*_args: object, **kwargs: object) -> torch.Tensor:
        generator = kwargs["generator"]
        captured["conditional"] = {
            "x_T": kwargs["x_T"].clone(),
            "state": generator.get_state().clone(),
        }
        return torch.zeros(1, 3, 32, 32)

    def unconditional_stub(*_args: object, **kwargs: object) -> torch.Tensor:
        generator = kwargs["generator"]
        captured["unconditional"] = {
            "x_T": kwargs["x_T"].clone(),
            "state": generator.get_state().clone(),
        }
        return torch.ones(1, 3, 32, 32)

    monkeypatch.setattr(inference, "conditional_sample", conditional_stub)
    monkeypatch.setattr(inference, "unconditional_sample", unconditional_stub)
    conditional_bundle, unconditional_bundle = _paired_test_bundles()
    attributes = PromptAttributes(5, 6, 9, 0)
    result = generate_paired(
        conditional_bundle, unconditional_bundle, seed=77, attributes=attributes
    )

    root = torch.Generator().manual_seed(77)
    initial_seed_state = root.get_state().clone()
    expected_x_T = torch.randn(1, 3, 32, 32, generator=root)
    expected_post_x_T_state = root.get_state().clone()
    assert torch.equal(captured["conditional"]["x_T"], expected_x_T)
    assert torch.equal(captured["unconditional"]["x_T"], expected_x_T)
    assert torch.equal(captured["conditional"]["state"], expected_post_x_T_state)
    assert torch.equal(captured["unconditional"]["state"], expected_post_x_T_state)
    assert not torch.equal(captured["conditional"]["state"], initial_seed_state)
    assert result.pairing.comparison_label == "QUALITATIVE PAIRED COMPARISON"
    assert result.pairing.shared_initial_x_T
    assert result.pairing.equivalent_post_x_T_generator_state
    assert torch.equal(result.conditional.initial_x_T, result.unconditional.initial_x_T)


def test_provenance_v1_pixel_hash_and_no_file_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditional_bundle, _ = _paired_test_bundles()

    def conditional_stub(*_args: object, **_kwargs: object) -> torch.Tensor:
        sample = torch.zeros(1, 3, 32, 32)
        sample[:, :, 0, 0] = 1.0
        return sample

    def no_git(*_args: object, **_kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(inference, "conditional_sample", conditional_stub)
    monkeypatch.setattr(inference.subprocess, "run", no_git)
    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: pytest.fail("write"))
    monkeypatch.setattr(Path, "write_bytes", lambda *_args, **_kwargs: pytest.fail("write"))
    monkeypatch.setattr(Image.Image, "save", lambda *_args, **_kwargs: pytest.fail("write"))
    inference._software_metadata.cache_clear()
    attributes = PromptAttributes(5, 6, 9, 0)
    result = generate_conditional(
        conditional_bundle,
        seed=91,
        attributes=attributes,
        snapshot_timesteps=[],
    )
    payload = result.provenance.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["schema_version"] == 1
    assert payload["generation"]["model_variant"] == "conditional"
    assert payload["generation"]["canonical_prompt"] == attributes.caption
    assert payload["generation"]["attributes"]["face_shape"]["raw_id"] == 5
    assert payload["model"]["checkpoint_sha256"] == inference.CONDITIONAL_SHA256
    assert payload["model"]["checkpoint_identity"] == (
        "artifacts/checkpoints/conditional_main_ema_step20000.pt"
    )
    assert payload["diffusion"] == {
        "image_size": [32, 32],
        "timesteps": 1000,
        "schedule_identity": "linear_beta_schedule",
        "beta_start": pytest.approx(1e-4),
        "beta_end": pytest.approx(0.02),
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
        "sampler_identity": "ancestral_ddpm",
        "clip_denoised": True,
        "precision": "fp32",
    }
    assert payload["runtime"] == {"requested_device": "auto", "resolved_device": "cpu"}
    assert payload["software"]["git_commit"] is None
    assert payload["software"]["git_dirty"] is None
    assert payload["observability"] is None
    assert "created_at" not in encoded
    assert "username" not in encoded and "hostname" not in encoded
    assert "png_sha256" not in encoded
    assert "output_filename" not in encoded and "output_directory" not in encoded
    assert str(inference.ROOT) not in encoded
    assert result.diagnostics is None and "diagnostics" not in payload
    assert payload["generation"]["pixel_content_sha256"] == pixel_content_sha256(
        result.image
    )
    inference._software_metadata.cache_clear()


def _mock_git_inspection(
    monkeypatch: pytest.MonkeyPatch, *, porcelain: str
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        assert kwargs["cwd"] == inference.ROOT
        stdout = "a" * 40 + "\n" if command[1:3] == ["rev-parse", "HEAD"] else porcelain
        return inference.subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    return calls


def test_git_metadata_reports_clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_git_inspection(monkeypatch, porcelain="")
    assert inference._git_metadata() == ("a" * 40, False)
    assert calls == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    ]
    assert '"git_dirty": false' in json.dumps(
        {"git_commit": "a" * 40, "git_dirty": False}
    )


def test_git_metadata_reports_dirty_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_git_inspection(monkeypatch, porcelain=" M src/module.py\n")
    assert inference._git_metadata() == ("a" * 40, True)
    assert '"git_dirty": true' in json.dumps(
        {"git_commit": "a" * 40, "git_dirty": True}
    )


def test_git_metadata_counts_untracked_files_as_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_git_inspection(monkeypatch, porcelain="?? untracked-file.txt\n")
    assert inference._git_metadata() == ("a" * 40, True)


def test_git_metadata_returns_nulls_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    assert inference._git_metadata() == (None, None)
    assert '"git_dirty": null' in json.dumps(
        {"git_commit": None, "git_dirty": None}
    )


def test_git_metadata_returns_nulls_outside_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_a_repository(command: list[str], **_kwargs: object) -> object:
        raise inference.subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(inference.subprocess, "run", not_a_repository)
    assert inference._git_metadata() == (None, None)


def test_no_git_export_software_provenance_is_null_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_a_repository(command: list[str], **_kwargs: object) -> object:
        raise inference.subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(inference.subprocess, "run", not_a_repository)
    inference._software_metadata.cache_clear()
    software = inference._software_metadata()
    assert software["git_commit"] is None
    assert software["git_dirty"] is None
    assert set(software) == {"package_version", "git_commit", "git_dirty"}
    encoded = json.dumps(software, sort_keys=True)
    for forbidden in (
        str(inference.ROOT),
        "username",
        "hostname",
        "remote",
        "branch",
        "patch",
        "diff",
        "inference.py",
    ):
        assert forbidden not in encoded
    inference._software_metadata.cache_clear()


def test_unconditional_provenance_uses_null_prompt_and_snapshot_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle = _paired_test_bundles()
    requested_timestep = int(bundle.schedule.betas.shape[0] - 1)

    def unconditional_stub(*_args: object, **kwargs: object) -> ObservedSample:
        snapshot = inference.SamplingSnapshot(
            requested_timestep, torch.zeros(1, 3, 32, 32)
        )
        return ObservedSample(torch.zeros(1, 3, 32, 32), (snapshot,))

    monkeypatch.setattr(inference, "unconditional_sample", unconditional_stub)
    result = generate_unconditional(
        bundle, seed=5, snapshot_timesteps=[requested_timestep, requested_timestep]
    )
    payload = result.provenance.to_dict()
    assert payload["generation"]["model_variant"] == "unconditional"
    assert payload["generation"]["canonical_prompt"] is None
    assert payload["generation"]["attributes"] is None
    assert payload["model"]["checkpoint_sha256"] == inference.UNCONDITIONAL_SHA256
    assert payload["observability"] == {
        "requested_snapshot_timesteps": [requested_timestep],
        "captured_snapshot_timesteps": [requested_timestep],
    }


def test_pixel_content_hash_is_raw_rgb_bytes_and_changes_with_pixels() -> None:
    first = Image.new("RGB", (32, 32), (0, 0, 0))
    second = first.copy()
    second.putpixel((0, 0), (1, 0, 0))
    assert pixel_content_sha256(first) == hashlib.sha256(first.tobytes()).hexdigest()
    assert pixel_content_sha256(first) != pixel_content_sha256(second)


def test_evaluator_diagnostic_maps_labels_and_scores() -> None:
    requested = PromptAttributes(face_shape=0, glasses=1, hair_color=2, face_color=3)
    diagnostic = evaluator_diagnostic(
        EvaluatorInferenceBundle(FixedEvaluator(), torch.device("cpu")),
        Image.new("RGB", (32, 32), (127, 127, 127)),
        requested=requested,
    )
    assert diagnostic["score_label"] == "UNCALIBRATED EVALUATOR SCORE"
    assert diagnostic["attributes_matched"] == 4
    assert diagnostic["attributes_total"] == 4
    for attribute, raw_id in zip(
        ("face_shape", "glasses", "hair_color", "face_color"), range(4), strict=True
    ):
        head = diagnostic["attributes"][attribute]
        assert head["predicted_raw_id"] == raw_id
        assert head["requested_raw_id"] == raw_id
        assert head["match"] is True
        assert sum(row["score"] for row in head["uncalibrated_evaluator_scores"]) == pytest.approx(1.0)


def test_evaluator_diagnostic_requires_canonical_generated_image() -> None:
    bundle = EvaluatorInferenceBundle(FixedEvaluator(), torch.device("cpu"))
    with pytest.raises(ValueError, match="RGB 32x32"):
        evaluator_diagnostic(bundle, Image.new("L", (32, 32)))
