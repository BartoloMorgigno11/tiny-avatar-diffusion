"""Professor-facing inference entry point for the four released DDPM models.

Technical training identifiers remain inside adapters and checkpoint provenance;
this interface accepts only the stable public model names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tiny_avatar_diffusion.final_streamlit_app import (
    caption_from_values,
    generate_public,
    model_by_id,
    public_control_labels,
)

PUBLIC_MODELS = ("DDPM-U32", "DDPM-C32", "DDPM-U64", "DDPM-C64")


def _seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed < 2**63:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**63)")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-avatar-public",
        description="Inference for the four released Tiny Avatar Diffusion models.",
    )
    parser.add_argument("--model", choices=PUBLIC_MODELS, required=True, help="Released model identifier.")
    parser.add_argument("--seed", type=_seed, default=2026, help="Initial-noise seed (default: 2026).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device.")
    parser.add_argument("--output", type=Path, required=True, help="New PNG destination.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing PNG.")
    for attribute, choices in public_control_labels().items():
        parser.add_argument(f"--{attribute.replace('_', '-')}", choices=choices)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.output.expanduser().resolve()
    if destination.suffix.lower() != ".png":
        raise SystemExit("--output must have the .png suffix")
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {destination}")
    entry = model_by_id(args.model)
    values = {key: getattr(args, key) for key in public_control_labels()}
    if entry["conditional"]:
        missing = [key for key, value in values.items() if value is None]
        if missing:
            raise SystemExit("conditional models require: " + ", ".join(f"--{key.replace('_', '-')}" for key in missing))
        attributes = caption_from_values(values)
    else:
        if any(value is not None for value in values.values()):
            raise SystemExit("unconditional models do not accept attribute options")
        attributes = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = generate_public(args.model, attributes, args.seed, args.device)
    image.save(destination, format="PNG")
    print(f"Generated {args.model}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
