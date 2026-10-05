"""Pure and lightweight UI tests for the optional Streamlit application."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Self

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from tiny_avatar_diffusion import inference, streamlit_app
from tiny_avatar_diffusion.diffusion import SamplingSnapshot

PROMPT = (
    "avatar face_shape_wide glasses_thick_rectangular_glasses "
    "hair_color_silver face_color_deep_brown"
)


def _provenance(variant: str, seed: int, pixel_hash: str) -> inference.GenerationProvenance:
    return inference.GenerationProvenance(
        schema_version=1,
        generation={
            "model_variant": variant,
            "seed": seed,
            "canonical_prompt": PROMPT if variant == "conditional" else None,
            "attributes": None,
            "pixel_content_sha256": pixel_hash,
        },
        model={
            "checkpoint_sha256": f"{variant}-checkpoint",
            "selected_step": 20_000,
            "weights_variant": "EMA",
        },
        diffusion={
            "timesteps": 1000,
            "schedule_identity": "linear_beta_schedule",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "prediction_type": "epsilon",
            "variance_type": "fixed_small",
            "clip_denoised": True,
            "precision": "fp32",
        },
        runtime={"requested_device": "auto", "resolved_device": "cpu"},
        software={"git_commit": "a" * 40, "git_dirty": True},
        observability=None,
    )


def _result(
    variant: str,
    *,
    color: tuple[int, int, int],
    pixel_hash: str,
    initial: torch.Tensor | None = None,
    snapshots: bool = False,
) -> inference.GenerationResult:
    x_t = initial if initial is not None else torch.linspace(-2, 2, 3 * 32 * 32).reshape(1, 3, 32, 32)
    observed = (
        tuple(
            SamplingSnapshot(timestep, torch.full((1, 3, 32, 32), index / 5))
            for index, timestep in enumerate(streamlit_app.SNAPSHOT_TIMESTEPS)
        )
        if snapshots
        else ()
    )
    return inference.GenerationResult(
        image=Image.new("RGB", (32, 32), color),
        final_tensor=torch.zeros(3, 32, 32),
        provenance=_provenance(variant, 2026, pixel_hash),
        initial_x_T=x_t,
        snapshots=observed,
    )


def _pair(*, snapshots: bool = False) -> inference.PairedGenerationResult:
    initial = torch.linspace(-2, 2, 3 * 32 * 32).reshape(1, 3, 32, 32)
    return inference.PairedGenerationResult(
        conditional=_result(
            "conditional",
            color=(10, 20, 30),
            pixel_hash="a1b2c3d4" + "1" * 56,
            initial=initial,
            snapshots=snapshots,
        ),
        unconditional=_result(
            "unconditional",
            color=(40, 50, 60),
            pixel_hash="e5f6a7b8" + "2" * 56,
            initial=initial.clone(),
            snapshots=snapshots,
        ),
        pairing=inference.PairingMetadata(
            comparison_label="QUALITATIVE PAIRED COMPARISON",
            seed=2026,
            shared_initial_x_T=True,
            equivalent_post_x_T_generator_state=True,
        ),
    )


def test_module_imports_with_installed_streamlit() -> None:
    assert streamlit_app.st is not None
    assert callable(streamlit_app.main)


def test_frozen_ui_defaults_and_snapshot_contract() -> None:
    assert streamlit_app.DEFAULT_SEED == 2026
    assert streamlit_app.MAX_SEED == 2**63 - 1
    assert streamlit_app.DEFAULT_DEVICE == "Auto"
    assert streamlit_app.DEFAULT_DENOISING is True
    assert streamlit_app.evaluator_default("Conditional") is True
    assert streamlit_app.evaluator_default("Unconditional") is False
    assert streamlit_app.SNAPSHOT_TIMESTEPS == (999, 749, 499, 249, 0)
    assert streamlit_app.SNAPSHOT_LABELS == (
        "Initial x_T",
        "Predicted x₀ at t=999",
        "Predicted x₀ at t=749",
        "Predicted x₀ at t=499",
        "Predicted x₀ at t=249",
        "Predicted x₀ at t=0",
    )
    with pytest.raises(ValueError, match="generation mode"):
        streamlit_app.evaluator_default("Other")


@pytest.mark.parametrize("value", ["0", "2026", str(2**53), str(2**63 - 1)])
def test_seed_parser_accepts_full_backend_domain(value: str) -> None:
    assert streamlit_app.parse_seed(value) == int(value)


@pytest.mark.parametrize("value", ["", "abc", "1.5", "-1", str(2**63)])
def test_seed_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Seed"):
        streamlit_app.parse_seed(value)


def test_structured_and_strict_prompt_inputs_are_exclusive_sources() -> None:
    values = {
        "face_shape": "wide",
        "glasses": "thick_rectangular_glasses",
        "hair_color": "silver",
        "face_color": "deep_brown",
    }
    structured = streamlit_app._resolve_attributes(False, "ignored", values)
    prompted = streamlit_app._resolve_attributes(True, PROMPT, None)
    assert structured == prompted == inference.PromptAttributes(5, 6, 9, 0)
    with pytest.raises(ValueError):
        streamlit_app._resolve_attributes(True, "free-form prompt", None)
    assert "wide" in streamlit_app.canonical_choices("face_shape")


def test_filename_and_technical_labels_derive_from_provenance() -> None:
    result = _result(
        "conditional", color=(10, 20, 30), pixel_hash="a1b2c3d4" + "1" * 56
    )
    assert streamlit_app.result_filename(result) == "conditional_seed2026_a1b2c3d4.png"
    details = streamlit_app._technical_details(result)
    assert details["model_variant"] == "conditional"
    assert details["seed"] == 2026
    assert details["weight_variant"] == "EMA"
    assert details["pixel_content_sha256"] == "a1b2c3d4" + "1" * 56
    assert not any(isinstance(value, str) and str(streamlit_app.__file__) in value for value in details.values())


def test_png_download_retains_original_and_display_copy_is_non_mutating() -> None:
    result = _result(
        "conditional", color=(10, 20, 30), pixel_hash="a1b2c3d4" + "1" * 56
    )
    original = result.image.tobytes()
    png = streamlit_app.png_bytes(result.image)
    downloaded = Image.open(io.BytesIO(png))
    preview = streamlit_app.display_copy(result.image)
    assert downloaded.mode == "RGB" and downloaded.size == (32, 32)
    assert preview.size == (256, 256)
    assert preview.getpixel((0, 0)) == result.image.getpixel((0, 0))
    assert result.image.tobytes() == original


def test_generation_downloads_are_stable_separate_and_write_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result(
        "conditional", color=(10, 20, 30), pixel_hash="a1b2c3d4" + "1" * 56
    )
    provenance_before = copy.deepcopy(result.provenance.to_dict())
    monkeypatch.setattr(Path, "write_bytes", lambda *_args, **_kwargs: pytest.fail("write"))
    first = streamlit_app.prepare_generation_downloads(
        result, {"diagnostic": True}, created_at_utc="2026-01-01T00:00:00Z"
    )
    repeated = streamlit_app.prepare_generation_downloads(
        result, {"diagnostic": True}, created_at_utc="2026-01-01T00:00:00Z"
    )
    assert first == repeated
    payload = json.loads(first.metadata)
    assert payload["provenance"] == provenance_before == result.provenance.to_dict()
    assert payload["diagnostics"] == {"diagnostic": True}
    assert payload["serialization"]["png_sha256"] == hashlib.sha256(first.png).hexdigest()
    assert payload["serialization"]["created_at_utc"] == "2026-01-01T00:00:00Z"
    assert "created_at_utc" not in json.dumps(payload["provenance"])
    assert str(Path.cwd()) not in first.metadata.decode("utf-8")
    assert first.steps_zip is None


def test_generation_steps_zip_uses_backend_snapshots_and_exact_names() -> None:
    result = _result(
        "conditional",
        color=(10, 20, 30),
        pixel_hash="a1b2c3d4" + "1" * 56,
        snapshots=True,
    )
    initial_before = result.initial_x_T.clone()
    downloads = streamlit_app.prepare_generation_downloads(result, None)
    assert downloads.steps_zip is not None
    with zipfile.ZipFile(io.BytesIO(downloads.steps_zip)) as archive:
        assert archive.namelist() == list(streamlit_app.STEP_FILENAMES)
        initial_image = Image.open(io.BytesIO(archive.read(streamlit_app.STEP_FILENAMES[0])))
        predicted = Image.open(io.BytesIO(archive.read(streamlit_app.STEP_FILENAMES[2])))
    assert initial_image.mode == "RGB" and initial_image.size == (32, 32)
    assert predicted.getpixel((0, 0)) == (153, 153, 153)
    assert torch.equal(result.initial_x_T, initial_before)
    metadata = json.loads(downloads.metadata)
    assert "clipped" in metadata["serialization"]["steps"]["initial_noise_display"]


def test_pair_downloads_are_distinct_with_one_metadata_and_exact_zip_layout() -> None:
    pair = _pair(snapshots=True)
    provenance_before = (
        copy.deepcopy(pair.conditional.provenance.to_dict()),
        copy.deepcopy(pair.unconditional.provenance.to_dict()),
    )
    downloads = streamlit_app.prepare_pair_downloads(
        pair,
        {"branch": "conditional"},
        {"branch": "unconditional"},
        created_at_utc="2026-02-03T04:05:06Z",
    )
    assert downloads.conditional_png_filename == "compare_conditional_seed2026_a1b2c3d4.png"
    assert downloads.unconditional_png_filename == "compare_unconditional_seed2026_e5f6a7b8.png"
    metadata = json.loads(downloads.metadata)
    assert set(metadata) == {"pairing", "conditional", "unconditional", "serialization"}
    assert metadata["conditional"]["provenance"] == provenance_before[0]
    assert metadata["unconditional"]["provenance"] == provenance_before[1]
    assert "comparison_score" not in json.dumps(metadata)
    assert str(Path.cwd()) not in downloads.metadata.decode("utf-8")
    assert downloads.steps_zip is not None
    with zipfile.ZipFile(io.BytesIO(downloads.steps_zip)) as archive:
        expected = [f"conditional/{name}" for name in streamlit_app.STEP_FILENAMES] + [
            f"unconditional/{name}" for name in streamlit_app.STEP_FILENAMES
        ]
        assert archive.namelist() == expected
        assert archive.read("conditional/00_initial_noise.png") == archive.read(
            "unconditional/00_initial_noise.png"
        )


def test_new_result_can_receive_new_serialization_timestamp() -> None:
    result = _result(
        "unconditional", color=(40, 50, 60), pixel_hash="e5f6a7b8" + "2" * 56
    )
    first = streamlit_app.prepare_generation_downloads(
        result, None, created_at_utc="2026-01-01T00:00:00Z"
    )
    second = streamlit_app.prepare_generation_downloads(
        result, None, created_at_utc="2026-01-01T00:00:01Z"
    )
    assert first.png == second.png
    assert first.metadata != second.metadata


def test_ui_source_exposes_no_scientific_tuning_widgets() -> None:
    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")
    forbidden_widget_labels = (
        'number_input("T"',
        'slider("Beta',
        'selectbox("Checkpoint',
        'slider("Guidance',
        'number_input("Batch size',
        'number_input("Number of generated images',
        'multiselect("Snapshot',
    )
    assert not any(label in source for label in forbidden_widget_labels)
    assert "st.number_input" not in source
    assert 'key="generate_seed"' in source
    assert 'key="compare_seed"' in source


def test_installed_streamlit_shell_reaches_exact_three_tabs() -> None:
    app = AppTest.from_file(str(Path(streamlit_app.__file__))).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Generate", "Compare", "Frozen Results"]


def test_seed_text_is_validated_only_when_each_generation_button_is_submitted() -> None:
    app = AppTest.from_file(str(Path(streamlit_app.__file__))).run(timeout=20)
    app.text_input(key="generate_seed").set_value("").run(timeout=20)
    assert not app.error
    app.button[0].click().run(timeout=20)
    assert [error.value for error in app.error] == ["Seed is required."]

    app.text_input(key="compare_seed").set_value("1.5").run(timeout=20)
    assert not app.error
    app.button[1].click().run(timeout=20)
    assert [error.value for error in app.error] == [
        "Seed must be a base-10 integer."
    ]


class _Context:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def metric(self, label: str, value: Any) -> None:
        self.owner.events.append(("metric", label, value))

    def __init__(self, owner: _Recorder) -> None:
        self.owner = owner


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            self.events.append((name, *args, kwargs))
            if name == "columns":
                count = args[0] if isinstance(args[0], int) else len(args[0])
                return [_Context(self) for _ in range(count)]
            if name == "expander":
                return _Context(self)
            return None

        return record


def _frozen_payload() -> dict[str, Any]:
    return {
        "conditioning": {
            "id_maa": 0.51,
            "ood_maa": 0.42,
            "absolute_maa_gap": 0.09,
            "macro_psr": 0.21,
            "id_maa_bootstrap_95": {"lower": 0.49, "upper": 0.53},
            "ood_maa_bootstrap_95": {"lower": 0.40, "upper": 0.44},
        },
        "quality_by_split": {
            "conditional": {
                "id": {"kid": 0.01, "fid": 40.1},
                "ood": {"kid": 0.02, "fid": 50.2},
            },
            "unconditional": {
                "id": {"kid": 0.03, "fid": 60.3},
                "ood": {"kid": 0.04, "fid": 70.4},
            },
        },
        "lpips_diversity_global_prompt_macro": 0.05,
        "parameter_counts": {"conditional": 101, "unconditional": 202},
        "latency": {
            model: {
                batch: {"median_seconds_per_image": value}
                for batch, value in (("batch_1", 1.1), ("batch_16", 1.6))
            }
            for model in ("conditional", "unconditional")
        },
        "memory": {
            model: {
                batch: {"peak_allocated_mib": value}
                for batch, value in (("batch_1", 2.1), ("batch_16", 2.6))
            }
            for model in ("conditional", "unconditional")
        },
    }


def test_frozen_results_presentation_uses_loader_values_and_scientific_caveats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)
    monkeypatch.setattr(streamlit_app, "load_frozen_data", _frozen_payload)
    monkeypatch.setattr(
        streamlit_app,
        "load_conditional_resource",
        lambda _device: pytest.fail("model load"),
    )
    streamlit_app._frozen_results_tab()
    rendered = json.dumps(recorder.events, default=str, ensure_ascii=False)
    for section in (
        "Conditioning adherence",
        "ID vs compositional OOD",
        "Generative quality",
        "Diversity & engineering",
    ):
        assert section in rendered
    assert "QUALITY ≠ CONDITIONING" in rendered
    assert "0.49" in rendered and "0.44" in rendered
    assert "0.09" in rendered and "point estimate only" in rendered
    assert "No canonical bootstrap confidence interval" in rendered
    assert "Lower KID/FID is better under this evaluation protocol" in rendered
    assert "Higher LPIPS indicates greater perceptual variation under the evaluated protocol" in rendered
    assert "significance test" in rendered
    assert "gap confidence interval is available" not in rendered


def test_resource_loaders_delegate_and_keep_device_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference,
        "load_conditional_bundle",
        lambda device: calls.append(("conditional", device)),
    )
    monkeypatch.setattr(
        inference,
        "load_unconditional_bundle",
        lambda device: calls.append(("unconditional", device)),
    )
    monkeypatch.setattr(
        inference,
        "load_evaluator_bundle",
        lambda device: calls.append(("evaluator", device)),
    )
    streamlit_app.load_conditional_resource("cpu")
    streamlit_app.load_conditional_resource("cuda")
    streamlit_app.load_unconditional_resource("cpu")
    streamlit_app.load_evaluator_resource("cuda")
    assert calls == [
        ("conditional", "cpu"),
        ("conditional", "cuda"),
        ("unconditional", "cpu"),
        ("evaluator", "cuda"),
    ]
