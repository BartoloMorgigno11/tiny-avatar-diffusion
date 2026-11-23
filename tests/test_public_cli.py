"""Focused contract tests for the four-model public CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tiny_avatar_diffusion import public_cli


ATTRIBUTES = [
    "--face-shape",
    "wide",
    "--glasses",
    "no_glasses",
    "--hair-color",
    "red",
    "--face-color",
    "pale_pink",
]


def test_public_cli_exposes_only_the_four_stable_model_names() -> None:
    parser = public_cli.build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model")
    assert model_action.choices == public_cli.PUBLIC_MODELS


def test_public_cli_requires_the_complete_conditional_grammar() -> None:
    with pytest.raises(SystemExit, match="conditional models require"):
        public_cli.main(["--model", "DDPM-C32", "--output", "avatar.png"])


def test_public_cli_rejects_attributes_for_an_unconditional_model() -> None:
    with pytest.raises(SystemExit, match="unconditional models do not accept"):
        public_cli.main(["--model", "DDPM-U64", *ATTRIBUTES, "--output", "avatar.png"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "DDPM-U32", "--seed", "-1", "--output", "avatar.png"],
        ["--model", "DDPM-U32", "--device", "metal", "--output", "avatar.png"],
    ],
)
def test_public_cli_rejects_invalid_seed_and_device(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        public_cli.main(arguments)
    assert error.value.code == 2


def test_public_cli_requires_a_png_output() -> None:
    with pytest.raises(SystemExit, match=".png"):
        public_cli.main(["--model", "DDPM-U32", "--output", "avatar.jpg"])


def test_public_cli_generates_only_after_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def generate(model: str, attributes: object, seed: int, device: str) -> Image.Image:
        calls.update(model=model, attributes=attributes, seed=seed, device=device)
        return Image.new("RGB", (32, 32), "white")

    saved: dict[str, object] = {}

    def save(_image: Image.Image, destination: Path, *, format: str) -> None:
        saved.update(destination=destination, format=format)

    monkeypatch.setattr(public_cli, "generate_public", generate)
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Image.Image, "save", save)
    output = Path("synthetic-output") / "avatar.png"

    assert public_cli.main(
        [
            "--model",
            "DDPM-C64",
            *ATTRIBUTES,
            "--seed",
            "2026",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    ) == 0
    assert saved["destination"] == output.resolve()
    assert saved["format"] == "PNG"
    assert calls["model"] == "DDPM-C64"
    assert calls["seed"] == 2026
    assert calls["device"] == "cpu"
    assert calls["attributes"] is not None


def test_public_cli_never_overwrites_without_explicit_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        public_cli.main(["--model", "DDPM-U32", "--output", "avatar.png"])
