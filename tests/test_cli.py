"""Focused tests for the Phase 3 command-line interface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from tiny_avatar_diffusion import cli, inference
from tiny_avatar_diffusion.diffusion import SamplingSnapshot

ATTR_ARGS = [
    "--face-shape",
    "wide",
    "--glasses",
    "thick_rectangular_glasses",
    "--hair-color",
    "silver",
    "--face-color",
    "deep_brown",
]
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
        model={"checkpoint_sha256": f"{variant}-checkpoint"},
        diffusion={"timesteps": 1000},
        runtime={"requested_device": "auto", "resolved_device": "cpu"},
        software={"git_commit": "a" * 40, "git_dirty": True},
        observability=None,
    )


def _result(
    variant: str,
    *,
    seed: int = 2026,
    color: tuple[int, int, int] = (10, 20, 30),
    pixel_hash: str | None = None,
    initial: torch.Tensor | None = None,
    with_snapshots: bool = False,
) -> inference.GenerationResult:
    image = Image.new("RGB", (32, 32), color)
    pixel_hash = pixel_hash or hashlib.sha256(image.tobytes()).hexdigest()
    x_t = initial if initial is not None else torch.linspace(-2, 2, 3 * 32 * 32).reshape(1, 3, 32, 32)
    snapshots = (
        tuple(
            SamplingSnapshot(timestep, torch.full((1, 3, 32, 32), index / 10))
            for index, timestep in enumerate(cli.CLI_SNAPSHOT_TIMESTEPS)
        )
        if with_snapshots
        else ()
    )
    return inference.GenerationResult(
        image=image,
        final_tensor=torch.zeros(3, 32, 32),
        provenance=_provenance(variant, seed, pixel_hash),
        initial_x_T=x_t,
        snapshots=snapshots,
    )


def _diagnostic(requested: inference.PromptAttributes | None) -> dict[str, Any]:
    values = {
        "face_shape": (5, "wide"),
        "glasses": (6, "thick_rectangular_glasses"),
        "hair_color": (9, "silver"),
        "face_color": (0, "deep_brown"),
    }
    attributes: dict[str, Any] = {}
    for name, (raw_id, label) in values.items():
        head: dict[str, Any] = {
            "predicted_raw_id": raw_id,
            "predicted_label": label,
            "uncalibrated_evaluator_scores": [{"raw_id": raw_id, "score": 0.75}],
        }
        if requested is not None:
            head.update(requested_label=label, requested_raw_id=raw_id, match=True)
        attributes[name] = head
    payload: dict[str, Any] = {"attributes": attributes}
    if requested is not None:
        payload.update(attributes_matched=4, attributes_total=4)
    return payload


def _fake_api(*, with_snapshots: bool = False) -> tuple[SimpleNamespace, dict[str, Any]]:
    calls: dict[str, Any] = {}
    shared_initial = torch.linspace(-2, 2, 3 * 32 * 32).reshape(1, 3, 32, 32)

    def load_conditional(device: str) -> object:
        calls.setdefault("conditional_devices", []).append(device)
        return object()

    def load_unconditional(device: str) -> object:
        calls.setdefault("unconditional_devices", []).append(device)
        return object()

    def generate_conditional(_bundle: object, **kwargs: Any) -> inference.GenerationResult:
        calls["conditional_kwargs"] = kwargs
        return _result(
            "conditional",
            seed=kwargs["seed"],
            pixel_hash="a1b2c3d4" + "1" * 56,
            with_snapshots=with_snapshots,
        )

    def generate_unconditional(_bundle: object, **kwargs: Any) -> inference.GenerationResult:
        calls["unconditional_kwargs"] = kwargs
        return _result(
            "unconditional",
            seed=kwargs["seed"],
            color=(40, 50, 60),
            pixel_hash="e5f6a7b8" + "2" * 56,
            with_snapshots=with_snapshots,
        )

    def generate_paired(_conditional: object, _unconditional: object, **kwargs: Any) -> Any:
        calls["paired_kwargs"] = kwargs
        return inference.PairedGenerationResult(
            conditional=_result(
                "conditional",
                seed=kwargs["seed"],
                pixel_hash="a1b2c3d4" + "1" * 56,
                initial=shared_initial,
                with_snapshots=with_snapshots,
            ),
            unconditional=_result(
                "unconditional",
                seed=kwargs["seed"],
                color=(40, 50, 60),
                pixel_hash="e5f6a7b8" + "2" * 56,
                initial=shared_initial.clone(),
                with_snapshots=with_snapshots,
            ),
            pairing=inference.PairingMetadata(
                comparison_label="QUALITATIVE PAIRED COMPARISON",
                seed=kwargs["seed"],
                shared_initial_x_T=True,
                equivalent_post_x_T_generator_state=True,
            ),
        )

    def evaluator_diagnostic(
        _bundle: object,
        _image: Image.Image,
        *,
        requested: inference.PromptAttributes | None,
    ) -> dict[str, Any]:
        calls.setdefault("evaluator_requests", []).append(requested)
        return _diagnostic(requested)

    api = SimpleNamespace(
        PromptAttributes=inference.PromptAttributes,
        validate_raw_caption=inference.validate_raw_caption,
        load_conditional_bundle=load_conditional,
        load_unconditional_bundle=load_unconditional,
        load_evaluator_bundle=lambda device: calls.setdefault("evaluator_devices", []).append(device),
        generate_conditional=generate_conditional,
        generate_unconditional=generate_unconditional,
        generate_paired=generate_paired,
        evaluator_diagnostic=evaluator_diagnostic,
        tensor_to_rgb_image=inference.tensor_to_rgb_image,
    )
    return api, calls


def _install_api(monkeypatch: pytest.MonkeyPatch, *, snapshots: bool = False) -> dict[str, Any]:
    api, calls = _fake_api(with_snapshots=snapshots)
    monkeypatch.setattr(cli, "_inference_api", lambda: api)
    return calls


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--help"], "generate"),
        (["generate", "conditional", "--help"], "structured attributes XOR --prompt"),
        (["generate", "unconditional", "--help"], "--evaluate"),
        (["compare", "--help"], "QUALITATIVE PAIRED COMPARISON"),
        (["results", "--help"], "frozen DR04/DR05 evidence"),
        (["demo", "--help"], "interactive demo"),
    ],
)
def test_help_surfaces(arguments: list[str], expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert expected in output
    if arguments == ["--help"]:
        assert "demo" in output


def test_structured_and_prompt_conditional_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_api(monkeypatch)
    first = tmp_path / "structured.png"
    assert cli.main(["generate", "conditional", *ATTR_ARGS, "--no-evaluate", "--output", str(first)]) == 0
    attributes = calls["conditional_kwargs"]["attributes"]
    assert attributes == inference.PromptAttributes(5, 6, 9, 0)
    second = tmp_path / "prompt.png"
    assert cli.main(
        ["generate", "conditional", "--prompt", PROMPT, "--no-evaluate", "--output", str(second)]
    ) == 0
    assert calls["conditional_kwargs"]["attributes"] == attributes


@pytest.mark.parametrize(
    "arguments",
    [
        ["generate", "conditional", "--prompt", PROMPT, *ATTR_ARGS],
        ["generate", "conditional", "--face-shape", "wide"],
        ["generate", "conditional", *ATTR_ARGS[:-1]],
        ["generate", "unconditional", "--prompt", PROMPT],
        ["generate", "unconditional", "--face-shape", "wide"],
        ["generate", "conditional", *ATTR_ARGS, "--seed", "-1"],
        ["generate", "conditional", *ATTR_ARGS, "--device", "metal"],
        ["generate", "conditional", "--face-shape", "unknown"],
    ],
)
def test_usage_errors_exit_two_without_traceback(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_operational_flags_and_unconditional_grammar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_api(monkeypatch, snapshots=True)
    output = tmp_path / "avatar.png"
    assert cli.main(
        [
            "generate",
            "unconditional",
            "--seed",
            "17",
            "--device",
            "cpu",
            "--metadata",
            "--save-steps",
            "--overwrite",
            "--output",
            str(output),
        ]
    ) == 0
    assert calls["unconditional_devices"] == ["cpu"]
    assert calls["unconditional_kwargs"] == {
        "seed": 17,
        "snapshot_timesteps": cli.CLI_SNAPSHOT_TIMESTEPS,
    }


def test_runtime_and_explicit_cuda_errors_exit_one_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api, _ = _fake_api()
    api.load_unconditional_bundle = lambda _device: (_ for _ in ()).throw(
        RuntimeError("CUDA was requested, but no CUDA device is available")
    )
    monkeypatch.setattr(cli, "_inference_api", lambda: api)
    status = cli.main(
        ["generate", "unconditional", "--device", "cuda", "--output", str(tmp_path / "x.png")]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "CUDA was requested" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "path",
    [
        cli.ROOT / "results" / "x.png",
        cli.ROOT / "data" / "processed" / ".." / "processed" / "x.png",
        cli.ROOT / "artifacts" / "x.png",
        cli.ROOT / "reports" / "x.png",
    ],
)
def test_protected_destinations_are_rejected_before_generation(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_api(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["generate", "unconditional", "--output", str(path)])
    assert exit_info.value.code == 2
    assert "unconditional_kwargs" not in calls


def test_invalid_extension_and_collision_are_usage_errors_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_api(monkeypatch)
    for output in (tmp_path / "x.jpg", tmp_path / "exists.png"):
        if output.suffix == ".png":
            output.write_bytes(b"old")
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["generate", "unconditional", "--output", str(output)])
        assert exit_info.value.code == 2
    assert "unconditional_kwargs" not in calls


def test_explicit_external_output_creates_parents_and_overwrite_is_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_api(monkeypatch)
    output = tmp_path / "nested" / "avatar.png"
    args = ["generate", "unconditional", "--output", str(output)]
    assert cli.main(args) == 0
    first = output.read_bytes()
    assert output.parent.is_dir()
    assert cli.main([*args, "--overwrite"]) == 0
    assert output.read_bytes() == first


def test_default_filename_and_no_companion_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_api(monkeypatch)
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIR", tmp_path)
    assert cli.main(["generate", "unconditional"]) == 0
    assert [path.name for path in tmp_path.iterdir()] == ["unconditional_seed2026_e5f6a7b8.png"]


def test_generate_metadata_separates_provenance_and_hashes_exact_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_api(monkeypatch)
    output = tmp_path / "avatar.png"
    assert cli.main(
        ["generate", "conditional", *ATTR_ARGS, "--metadata", "--output", str(output)]
    ) == 0
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(sidecar) == {"provenance", "diagnostics", "serialization"}
    assert sidecar["provenance"] == _provenance(
        "conditional", 2026, "a1b2c3d4" + "1" * 56
    ).to_dict()
    assert sidecar["provenance"]["software"]["git_dirty"] is True
    assert "created_at_utc" not in json.dumps(sidecar["provenance"])
    assert sidecar["serialization"]["png_filename"] == "avatar.png"
    assert sidecar["serialization"]["png_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar["serialization"]["created_at_utc"].endswith("Z")
    assert str(tmp_path) not in json.dumps(sidecar)


def test_snapshot_outputs_use_frozen_schedule_and_do_not_mutate_initial_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, calls = _fake_api(with_snapshots=True)
    original_generate = api.generate_conditional
    captured: dict[str, torch.Tensor] = {}

    def capture(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        result = original_generate(*args, **kwargs)
        captured["result"] = result.initial_x_T
        captured["before"] = result.initial_x_T.clone()
        return result

    api.generate_conditional = capture
    monkeypatch.setattr(cli, "_inference_api", lambda: api)
    output = tmp_path / "avatar.png"
    assert cli.main(
        [
            "generate",
            "conditional",
            *ATTR_ARGS,
            "--no-evaluate",
            "--save-steps",
            "--metadata",
            "--output",
            str(output),
        ]
    ) == 0
    assert calls["conditional_kwargs"]["snapshot_timesteps"] == (999, 749, 499, 249, 0)
    assert torch.equal(captured["result"], captured["before"])
    step_root = tmp_path / "avatar_steps"
    assert sorted(path.name for path in step_root.iterdir()) == sorted(cli._step_filenames())
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    steps = metadata["serialization"]["steps"]
    assert steps["captured_timesteps"] == [999, 749, 499, 249, 0]
    assert "clipped" in steps["initial_noise_display"]
    assert all(not item["filename"].startswith(("/", "C:")) for item in steps["files"])


def test_compare_outputs_pair_sidecar_steps_and_atomic_collision_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_api(monkeypatch, snapshots=True)
    args = ["compare", *ATTR_ARGS, "--no-evaluate", "--metadata", "--save-steps", "--output-dir", str(tmp_path)]
    assert cli.main(args) == 0
    conditional = tmp_path / "compare_conditional_seed2026_a1b2c3d4.png"
    unconditional = tmp_path / "compare_unconditional_seed2026_e5f6a7b8.png"
    sidecar = tmp_path / "compare_seed2026_a1b2c3d4_e5f6a7b8.json"
    assert conditional.is_file() and unconditional.is_file() and sidecar.is_file()
    assert len(list(tmp_path.glob("*.json"))) == 1
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(payload) == {"pairing", "conditional", "unconditional", "serialization"}
    assert "comparison" not in payload
    step_root = tmp_path / "compare_seed2026_a1b2c3d4_e5f6a7b8_steps"
    assert (step_root / "conditional" / "00_initial_noise.png").read_bytes() == (
        step_root / "unconditional" / "00_initial_noise.png"
    ).read_bytes()
    assert sorted(path.name for path in (step_root / "conditional").iterdir()) == sorted(
        cli._step_filenames()
    )
    assert calls["paired_kwargs"]["snapshot_timesteps"] == cli.CLI_SNAPSHOT_TIMESTEPS

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    collision = blocked / "compare_conditional_seed2026_a1b2c3d4.png"
    collision.write_bytes(b"old")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["compare", *ATTR_ARGS, "--no-evaluate", "--output-dir", str(blocked)])
    assert exit_info.value.code == 2
    assert collision.read_bytes() == b"old"
    assert not (blocked / "compare_unconditional_seed2026_e5f6a7b8.png").exists()


def test_evaluator_defaults_and_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _install_api(monkeypatch)
    assert cli.main(
        ["generate", "conditional", *ATTR_ARGS, "--output", str(tmp_path / "c.png")]
    ) == 0
    conditional_text = capsys.readouterr().out
    assert "LIVE SINGLE-SAMPLE DIAGNOSTIC" in conditional_text
    assert "Requested | Predicted | Uncalibrated evaluator score | Match" in conditional_text
    assert "Attributes matched: 4 / 4" in conditional_text
    assert "not probabilities of correctness" in conditional_text

    assert cli.main(
        ["generate", "unconditional", "--evaluate", "--output", str(tmp_path / "u.png")]
    ) == 0
    unconditional_text = capsys.readouterr().out
    assert "Attribute | Predicted | Uncalibrated evaluator score" in unconditional_text
    assert "Requested" not in unconditional_text
    assert "Attributes matched" not in unconditional_text

    assert cli.main(
        ["compare", *ATTR_ARGS, "--output-dir", str(tmp_path / "pair")]
    ) == 0
    compare_text = capsys.readouterr().out
    assert "QUALITATIVE PAIRED COMPARISON" in compare_text
    assert "CONDITIONAL BRANCH" in compare_text and "UNCONDITIONAL BRANCH" in compare_text
    assert compare_text.count("LIVE SINGLE-SAMPLE DIAGNOSTIC") == 2
    assert "Not a dataset-level metric." in compare_text
    assert len(calls["evaluator_requests"]) == 4
    assert calls["evaluator_requests"][-1] == calls["evaluator_requests"][-2]


def test_results_is_lazy_read_only_and_uses_loader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    values = {
        "conditioning": {"id_maa": 1, "ood_maa": 2, "absolute_maa_gap": 3, "macro_psr": 4},
        "quality_global": {
            "conditional": {"kid": 5, "fid": 6},
            "unconditional": {"kid": 7, "fid": 8},
        },
        "lpips_diversity_global_prompt_macro": 9,
        "parameter_counts": {"conditional": 10, "unconditional": 11},
        "latency": {"conditional": {"batch_1": {"median": 12}}, "unconditional": {}},
        "memory": {"conditional": {}, "unconditional": {"batch_1": {"peak": 13}}},
    }
    monkeypatch.setattr(cli, "_frozen_results_loader", lambda: lambda: values)
    monkeypatch.setattr(cli, "_inference_api", lambda: pytest.fail("inference loaded"))
    monkeypatch.setattr(Path, "write_bytes", lambda *_args, **_kwargs: pytest.fail("write"))
    assert cli.main(["results"]) == 0
    output = capsys.readouterr().out
    for text in ("CONDITIONING", "QUALITY & DIVERSITY", "ENGINEERING", "QUALITY != CONDITIONING"):
        assert text in output
    for value in range(1, 14):
        assert str(value) in output
    assert "frozen DR04/DR05 evidence" in output


def test_module_entry_and_exact_console_alias() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tiny_avatar_diffusion.cli", "--help"],
        cwd=cli.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Tiny Avatar Diffusion" in completed.stdout
    project = tomllib.loads((cli.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"tiny-avatar": "tiny_avatar_diffusion.cli:main"}
    assert project["project"]["optional-dependencies"]["demo"] == ["streamlit>=1.32,<2"]


def test_demo_missing_optional_dependency_is_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)
    assert cli.main(["demo"]) == 1
    captured = capsys.readouterr()
    assert "optional demo dependencies" in captured.err
    assert 'pip install -e ".[demo]"' in captured.err
    assert "Traceback" not in captured.err


def test_demo_launcher_uses_current_python_and_packaged_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], *, check: bool) -> Any:
        captured.update(command=command, check=check)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["demo"]) == 7
    assert captured == {
        "command": [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(cli.__file__).with_name("streamlit_app.py")),
        ],
        "check": False,
    }
