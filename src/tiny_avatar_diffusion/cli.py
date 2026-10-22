"""Cross-platform command-line interface for Tiny Avatar Diffusion V1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "demo"
PROTECTED_REPOSITORY_PATHS = tuple(
    (ROOT / relative).resolve()
    for relative in ("results", "data/processed", "artifacts", "reports")
)
CLI_SNAPSHOT_TIMESTEPS = (999, 749, 499, 249, 0)


class CliUsageError(ValueError):
    """Represent a user input error that must exit with argparse code 2."""


def _inference_api() -> ModuleType:
    from tiny_avatar_diffusion import inference

    return inference


def _frozen_results_loader() -> Any:
    from tiny_avatar_diffusion.frozen_results import load_frozen_results

    return load_frozen_results


def _attribute_choices(attribute: str) -> tuple[str, ...]:
    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING

    return tuple(value.canonical_label for value in SEMANTIC_MAPPING[attribute].values())


def _seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed < 2**63:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**63)")
    return seed


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=_seed, default=2026, help="Generation seed (default: 2026).")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument("--metadata", action="store_true", help="Write JSON metadata.")
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save the initial-noise display and frozen predicted-x0 snapshots.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output files.")


def _add_conditional_description(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt",
        help="Strict canonical caption; mutually exclusive with structured attributes.",
    )
    parser.add_argument("--face-shape", choices=_attribute_choices("face_shape"))
    parser.add_argument("--glasses", choices=_attribute_choices("glasses"))
    parser.add_argument("--hair-color", choices=_attribute_choices("hair_color"))
    parser.add_argument("--face-color", choices=_attribute_choices("face_color"))


def _add_conditional_model_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("h7", "v1-ca8"),
        default="h7",
        help=(
            "Conditional checkpoint: h7 is the selected CA8+CA16 40k EMA release; "
            "v1-ca8 is the historical 20k CA8 baseline."
        ),
    )


def _add_v2_model_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("v2-a", "v2-b"),
        required=True,
        help="Native64 model architecture; an exact authenticated external checkpoint is required.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the authenticated V2 40k training checkpoint (not package-shipped).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-avatar",
        description=(
            "Tiny Avatar Diffusion\n\n"
            "From-scratch text-conditioned DDPM for 32x32 cartoon avatars."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, title="commands")

    generate = commands.add_parser("generate", help="Generate one avatar.")
    variants = generate.add_subparsers(dest="variant", required=True, title="variants")

    conditional = variants.add_parser(
        "conditional",
        help="Generate from structured attributes or one strict canonical prompt.",
        description="Generate a conditional avatar using structured attributes XOR --prompt.",
    )
    _add_conditional_description(conditional)
    _add_conditional_model_selection(conditional)
    _add_runtime_arguments(conditional)
    conditional.add_argument("--output", type=Path, help="Explicit .png destination.")
    conditional.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        default=True,
        help="Disable the default live evaluator diagnostic.",
    )
    conditional.set_defaults(handler=_run_generate_conditional)

    unconditional = variants.add_parser(
        "unconditional",
        help="Generate from the unconditional neural baseline.",
        description="Generate an unconditional avatar without prompt or attribute inputs.",
    )
    _add_runtime_arguments(unconditional)
    unconditional.add_argument("--output", type=Path, help="Explicit .png destination.")
    unconditional.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the optional live evaluator diagnostic.",
    )
    unconditional.set_defaults(handler=_run_generate_unconditional)

    native64 = variants.add_parser(
        "v2",
        help="Generate one native64 V2-A or V2-B avatar from an authenticated external checkpoint.",
        description="Native64 V2 generation; no resize is applied before or after sampling.",
    )
    _add_conditional_description(native64)
    _add_v2_model_selection(native64)
    native64.add_argument("--seed", type=_seed, default=2026, help="Generation seed (default: 2026).")
    native64.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    native64.add_argument("--metadata", action="store_true", help="Write JSON metadata.")
    native64.add_argument("--overwrite", action="store_true", help="Allow replacing output files.")
    native64.add_argument("--output", type=Path, help="Explicit .png destination.")
    native64.set_defaults(handler=_run_generate_v2)

    compare = commands.add_parser(
        "compare",
        help="Generate a qualitative paired conditional/unconditional comparison.",
        description=(
            "QUALITATIVE PAIRED COMPARISON using structured attributes XOR --prompt.\n"
            "This is not a scientific metric, conditioning score, or quality score."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_conditional_description(compare)
    _add_conditional_model_selection(compare)
    _add_runtime_arguments(compare)
    compare.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory."
    )
    compare.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        default=True,
        help="Disable live diagnostics for both branches.",
    )
    compare.set_defaults(handler=_run_compare)

    results = commands.add_parser(
        "results",
        help="Display frozen DR04/DR05 evidence.",
        description="Read-only viewer for frozen DR04/DR05 evidence.",
    )
    results.set_defaults(handler=_run_results)

    demo = commands.add_parser(
        "demo",
        help="Launch the optional Streamlit interactive demo.",
        description="Launch the Tiny Avatar Diffusion interactive demo.",
    )
    demo.set_defaults(handler=_run_demo)
    return parser


def _structured_attributes(args: argparse.Namespace, api: ModuleType) -> Any:
    names = ("face_shape", "glasses", "hair_color", "face_color")
    supplied = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.prompt is not None and supplied:
        raise CliUsageError("--prompt cannot be combined with structured attributes")
    if args.prompt is not None:
        try:
            attributes = api.validate_raw_caption(args.prompt)
        except (TypeError, ValueError) as error:
            raise CliUsageError(str(error)) from error
        return attributes, args.prompt
    if len(supplied) != len(names):
        raise CliUsageError(
            "provide --prompt or all four structured attributes: --face-shape, "
            "--glasses, --hair-color, and --face-color"
        )

    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING

    raw_values: dict[str, int] = {}
    for name in names:
        selected = getattr(args, name)
        raw_values[name] = next(
            raw_id
            for raw_id, value in SEMANTIC_MAPPING[name].items()
            if value.canonical_label == selected
        )
    attributes = api.PromptAttributes(**raw_values)
    return attributes, attributes.caption


def _resolved_output(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".png":
        raise CliUsageError("output path must use the .png suffix")
    resolved = _resolved_output(path)
    _reject_protected_path(resolved)
    return resolved


def _validate_output_directory(path: Path) -> Path:
    resolved = _resolved_output(path)
    _reject_protected_path(resolved)
    return resolved


def _reject_protected_path(path: Path) -> None:
    for protected in PROTECTED_REPOSITORY_PATHS:
        if path == protected or protected in path.parents:
            raise CliUsageError(
                "demo output cannot be written under results, data/processed, artifacts, or reports"
            )


def _step_filenames() -> tuple[str, ...]:
    return (
        "00_initial_noise.png",
        *(f"{index:02d}_pred_x0_t{timestep:04d}.png" for index, timestep in enumerate(CLI_SNAPSHOT_TIMESTEPS, 1)),
    )


def _generate_destinations(
    image_path: Path, *, metadata: bool, save_steps: bool
) -> tuple[list[Path], list[Path]]:
    files = [image_path]
    directories: list[Path] = []
    if metadata:
        files.append(image_path.with_suffix(".json"))
    if save_steps:
        step_root = image_path.with_name(f"{image_path.stem}_steps")
        directories.append(step_root)
        files.extend(step_root / name for name in _step_filenames())
    return files, directories


def _preflight(
    files: Sequence[Path], directories: Sequence[Path], *, overwrite: bool
) -> None:
    for path in (*files, *directories):
        _reject_protected_path(path)
    if overwrite:
        for directory in directories:
            if directory.exists() and not directory.is_dir():
                raise CliUsageError(f"step output path is not a directory: {directory}")
        return
    collisions = [path for path in (*files, *directories) if path.exists()]
    if collisions:
        raise CliUsageError(f"output already exists: {collisions[0]}")


def _png_bytes(image: Any) -> bytes:
    if image.mode != "RGB" or image.size != (32, 32):
        raise RuntimeError("CLI serialization requires an RGB 32x32 image")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _native64_png_bytes(image: Any) -> bytes:
    if image.mode != "RGB" or image.size != (64, 64):
        raise RuntimeError("V2 CLI serialization requires an RGB 64x64 image")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _step_payloads(result: Any, step_root: Path, api: ModuleType) -> list[tuple[Path, bytes]]:
    snapshots = {item.timestep: item for item in result.snapshots}
    if tuple(snapshots) != CLI_SNAPSHOT_TIMESTEPS:
        raise RuntimeError("backend did not return the frozen CLI snapshot schedule")
    payloads = [
        (step_root / "00_initial_noise.png", _png_bytes(api.tensor_to_rgb_image(result.initial_x_T[0])))
    ]
    for index, timestep in enumerate(CLI_SNAPSHOT_TIMESTEPS, 1):
        tensor = snapshots[timestep].predicted_x_start
        if tensor.shape == (1, 3, 32, 32):
            tensor = tensor[0]
        payloads.append(
            (
                step_root / f"{index:02d}_pred_x0_t{timestep:04d}.png",
                _png_bytes(api.tensor_to_rgb_image(tensor)),
            )
        )
    return payloads


def _step_serialization(
    payloads: Sequence[tuple[Path, bytes]], *, relative_to: Path, captured: Sequence[int]
) -> dict[str, Any]:
    return {
        "initial_noise_display": "x_T clipped to [-1, 1] for display only",
        "captured_timesteps": list(captured),
        "files": [
            {
                "filename": path.relative_to(relative_to).as_posix(),
                "png_sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in payloads
        ],
    }


def _write_payloads(payloads: Sequence[tuple[Path, bytes]]) -> None:
    for path, content in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _evaluate(api: ModuleType, result: Any, device: str, requested: Any | None) -> Any:
    evaluator = api.load_evaluator_bundle(device)
    return api.evaluator_diagnostic(evaluator, result.image, requested=requested)


def _diagnostic_score(head: dict[str, Any]) -> float:
    predicted = head["predicted_raw_id"]
    return next(row["score"] for row in head["uncalibrated_evaluator_scores"] if row["raw_id"] == predicted)


def _print_diagnostic(diagnostic: dict[str, Any], *, title: str, requested: bool) -> None:
    print(f"\n{title}")
    print("LIVE SINGLE-SAMPLE DIAGNOSTIC")
    if requested:
        print("Attribute | Requested | Predicted | Uncalibrated evaluator score | Match")
    else:
        print("Attribute | Predicted | Uncalibrated evaluator score")
    for attribute, head in diagnostic["attributes"].items():
        score = _diagnostic_score(head)
        if requested:
            print(
                f"{attribute} | {head['requested_label']} | {head['predicted_label']} | "
                f"{score:.6f} | {str(head['match']).lower()}"
            )
        else:
            print(f"{attribute} | {head['predicted_label']} | {score:.6f}")
    if requested:
        print(f"Attributes matched: {diagnostic['attributes_matched']} / 4")
    print("Scores are uncalibrated evaluator scores, not probabilities of correctness.")


def _summary(result: Any, image_path: Path, *, prompt: str | None) -> None:
    generation = result.provenance.generation
    print("\nGeneration completed")
    print(f"Image: {image_path}")
    print(f"Model: {generation['model_variant']}")
    if prompt is not None:
        print(f"Prompt: {prompt}")
    print(f"Seed: {generation['seed']}")
    print(f"Device: {result.provenance.runtime['resolved_device']}")
    print(f"Pixel SHA256: {generation['pixel_content_sha256']}")


def _run_generate_conditional(args: argparse.Namespace) -> int:
    api = _inference_api()
    attributes, prompt = _structured_attributes(args, api)
    return _run_generate(args, api=api, variant="conditional", attributes=attributes, prompt=prompt)


def _run_generate_unconditional(args: argparse.Namespace) -> int:
    return _run_generate(args, api=_inference_api(), variant="unconditional", attributes=None, prompt=None)


def _run_generate_v2(args: argparse.Namespace) -> int:
    api = _inference_api()
    attributes, prompt = _structured_attributes(args, api)
    explicit_output = _validate_output_path(args.output) if args.output is not None else None
    if explicit_output is not None:
        _preflight([explicit_output, *( [explicit_output.with_suffix(".json")] if args.metadata else [])], [], overwrite=args.overwrite)
    variant = args.model.replace("-", "_")
    bundle = api.load_v2_native64_bundle(variant, args.checkpoint, device=args.device)
    print("Sampling native64 with 1000 ancestral DDPM reverse steps...")
    image = api.generate_v2_native64(bundle, caption=prompt, seed=args.seed)
    pixel_prefix = hashlib.sha256(image.tobytes()).hexdigest()[:8]
    image_path = explicit_output or (DEFAULT_OUTPUT_DIR / f"{args.model}_native64_seed{args.seed}_{pixel_prefix}.png").resolve()
    files = [image_path, *( [image_path.with_suffix(".json")] if args.metadata else [])]
    _preflight(files, [], overwrite=args.overwrite)
    payloads: list[tuple[Path, bytes]] = [(image_path, _native64_png_bytes(image))]
    if args.metadata:
        payloads.append((image_path.with_suffix(".json"), _json_bytes({
            "model_variant": variant,
            "native_resolution": [64, 64],
            "resize_applied": False,
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "checkpoint_path": str(bundle.checkpoint_path),
            "seed": args.seed,
            "canonical_prompt": prompt,
            "attributes": {name: getattr(attributes, name) for name in ("face_shape", "glasses", "hair_color", "face_color")},
            "sampler": "ancestral_ddpm_1000",
            "png_sha256": hashlib.sha256(_native64_png_bytes(image)).hexdigest(),
        })))
    _write_payloads(payloads)
    print(f"\nGeneration completed\nImage: {image_path}\nModel: {args.model}\nNative resolution: 64x64\nSeed: {args.seed}")
    return 0


def _run_generate(
    args: argparse.Namespace,
    *,
    api: ModuleType,
    variant: str,
    attributes: Any | None,
    prompt: str | None,
) -> int:
    explicit_output = _validate_output_path(args.output) if args.output is not None else None
    if explicit_output is not None:
        files, directories = _generate_destinations(
            explicit_output, metadata=args.metadata, save_steps=args.save_steps
        )
        _preflight(files, directories, overwrite=args.overwrite)

    print("Sampling 1000 reverse steps...")
    snapshot_timesteps = CLI_SNAPSHOT_TIMESTEPS if args.save_steps else None
    if variant == "conditional":
        model_choice = getattr(args, "model", "h7")
        bundle = (
            api.load_h7_conditional_bundle(args.device)
            if model_choice == "h7"
            else api.load_conditional_bundle(args.device)
        )
        result = api.generate_conditional(
            bundle,
            seed=args.seed,
            attributes=attributes,
            snapshot_timesteps=snapshot_timesteps,
        )
    else:
        bundle = api.load_unconditional_bundle(args.device)
        result = api.generate_unconditional(
            bundle,
            seed=args.seed,
            snapshot_timesteps=snapshot_timesteps,
        )

    diagnostic = _evaluate(api, result, args.device, attributes) if args.evaluate else None
    pixel_prefix = result.provenance.generation["pixel_content_sha256"][:8]
    image_path = explicit_output or (
        DEFAULT_OUTPUT_DIR / f"{variant}_seed{args.seed}_{pixel_prefix}.png"
    ).resolve()
    files, directories = _generate_destinations(
        image_path, metadata=args.metadata, save_steps=args.save_steps
    )
    _preflight(files, directories, overwrite=args.overwrite)

    png = _png_bytes(result.image)
    payloads: list[tuple[Path, bytes]] = [(image_path, png)]
    step_payloads: list[tuple[Path, bytes]] = []
    if args.save_steps:
        step_root = image_path.with_name(f"{image_path.stem}_steps")
        step_payloads = _step_payloads(result, step_root, api)
        payloads.extend(step_payloads)
    if args.metadata:
        serialization: dict[str, Any] = {
            "created_at_utc": _utc_now(),
            "png_filename": image_path.name,
            "png_sha256": hashlib.sha256(png).hexdigest(),
        }
        if step_payloads:
            serialization["steps"] = _step_serialization(
                step_payloads,
                relative_to=image_path.parent,
                captured=[item.timestep for item in result.snapshots],
            )
        payloads.append(
            (
                image_path.with_suffix(".json"),
                _json_bytes(
                    {
                        "provenance": result.provenance.to_dict(),
                        "diagnostics": diagnostic,
                        "serialization": serialization,
                    }
                ),
            )
        )
    _write_payloads(payloads)
    _summary(result, image_path, prompt=prompt)
    if diagnostic is not None:
        _print_diagnostic(
            diagnostic,
            title=f"{variant.upper()} DIAGNOSTIC",
            requested=attributes is not None,
        )
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    api = _inference_api()
    attributes, prompt = _structured_attributes(args, api)
    output_dir = _validate_output_directory(args.output_dir)
    print("QUALITATIVE PAIRED COMPARISON")
    print("Sampling 1000 reverse steps per branch...")
    snapshot_timesteps = CLI_SNAPSHOT_TIMESTEPS if args.save_steps else None
    conditional_bundle = (
        api.load_h7_conditional_bundle(args.device)
        if args.model == "h7"
        else api.load_conditional_bundle(args.device)
    )
    unconditional_bundle = api.load_unconditional_bundle(args.device)
    pair = api.generate_paired(
        conditional_bundle,
        unconditional_bundle,
        seed=args.seed,
        attributes=attributes,
        snapshot_timesteps=snapshot_timesteps,
    )
    conditional_hash = pair.conditional.provenance.generation["pixel_content_sha256"][:8]
    unconditional_hash = pair.unconditional.provenance.generation["pixel_content_sha256"][:8]
    conditional_path = output_dir / (
        f"compare_conditional_seed{args.seed}_{conditional_hash}.png"
    )
    unconditional_path = output_dir / (
        f"compare_unconditional_seed{args.seed}_{unconditional_hash}.png"
    )
    metadata_path = output_dir / (
        f"compare_seed{args.seed}_{conditional_hash}_{unconditional_hash}.json"
    )
    step_root = output_dir / (
        f"compare_seed{args.seed}_{conditional_hash}_{unconditional_hash}_steps"
    )
    files = [conditional_path, unconditional_path]
    directories: list[Path] = []
    if args.metadata:
        files.append(metadata_path)
    if args.save_steps:
        directories.append(step_root)
        for branch in ("conditional", "unconditional"):
            files.extend(step_root / branch / name for name in _step_filenames())
    _preflight(files, directories, overwrite=args.overwrite)

    conditional_diagnostic = None
    unconditional_diagnostic = None
    if args.evaluate:
        evaluator = api.load_evaluator_bundle(args.device)
        conditional_diagnostic = api.evaluator_diagnostic(
            evaluator, pair.conditional.image, requested=attributes
        )
        unconditional_diagnostic = api.evaluator_diagnostic(
            evaluator, pair.unconditional.image, requested=attributes
        )

    conditional_png = _png_bytes(pair.conditional.image)
    unconditional_png = _png_bytes(pair.unconditional.image)
    payloads: list[tuple[Path, bytes]] = [
        (conditional_path, conditional_png),
        (unconditional_path, unconditional_png),
    ]
    conditional_steps: list[tuple[Path, bytes]] = []
    unconditional_steps: list[tuple[Path, bytes]] = []
    if args.save_steps:
        conditional_steps = _step_payloads(pair.conditional, step_root / "conditional", api)
        unconditional_steps = _step_payloads(
            pair.unconditional, step_root / "unconditional", api
        )
        if conditional_steps[0][1] != unconditional_steps[0][1]:
            raise RuntimeError("paired initial-noise display PNGs are not identical")
        payloads.extend(conditional_steps)
        payloads.extend(unconditional_steps)
    if args.metadata:
        created_at = _utc_now()
        conditional_serialization: dict[str, Any] = {
            "png_filename": conditional_path.name,
            "png_sha256": hashlib.sha256(conditional_png).hexdigest(),
        }
        unconditional_serialization: dict[str, Any] = {
            "png_filename": unconditional_path.name,
            "png_sha256": hashlib.sha256(unconditional_png).hexdigest(),
        }
        if args.save_steps:
            conditional_serialization["steps"] = _step_serialization(
                conditional_steps,
                relative_to=output_dir,
                captured=[item.timestep for item in pair.conditional.snapshots],
            )
            unconditional_serialization["steps"] = _step_serialization(
                unconditional_steps,
                relative_to=output_dir,
                captured=[item.timestep for item in pair.unconditional.snapshots],
            )
        payloads.append(
            (
                metadata_path,
                _json_bytes(
                    {
                        "pairing": asdict(pair.pairing),
                        "conditional": {
                            "provenance": pair.conditional.provenance.to_dict(),
                            "diagnostics": conditional_diagnostic,
                            "serialization": conditional_serialization,
                        },
                        "unconditional": {
                            "provenance": pair.unconditional.provenance.to_dict(),
                            "diagnostics": unconditional_diagnostic,
                            "serialization": unconditional_serialization,
                        },
                        "serialization": {"created_at_utc": created_at},
                    }
                ),
            )
        )
    _write_payloads(payloads)
    _summary(pair.conditional, conditional_path, prompt=prompt)
    _summary(pair.unconditional, unconditional_path, prompt=None)
    print("\nQUALITATIVE PAIRED COMPARISON")
    print("Not a dataset-level metric.")
    if conditional_diagnostic is not None and unconditional_diagnostic is not None:
        _print_diagnostic(
            conditional_diagnostic,
            title="CONDITIONAL BRANCH",
            requested=True,
        )
        _print_diagnostic(
            unconditional_diagnostic,
            title="UNCONDITIONAL BRANCH",
            requested=True,
        )
    return 0


def _run_results(_args: argparse.Namespace) -> int:
    values = _frozen_results_loader()()
    conditioning = values["conditioning"]
    print("CONDITIONING")
    print(f"ID MAA: {conditioning['id_maa']}")
    print(f"OOD MAA: {conditioning['ood_maa']}")
    print(f"Absolute ID->OOD gap: {conditioning['absolute_maa_gap']}")
    print(f"MacroPSR: {conditioning['macro_psr']}")

    print("\nQUALITY & DIVERSITY")
    for model in ("conditional", "unconditional"):
        quality = values["quality_global"][model]
        print(f"{model.title()} KID: {quality['kid']}")
        print(f"{model.title()} FID: {quality['fid']}")
    print(f"LPIPS diversity: {values['lpips_diversity_global_prompt_macro']}")

    print("\nENGINEERING")
    for model, count in values["parameter_counts"].items():
        print(f"{model.title()} parameters: {count}")
    for category in ("latency", "memory"):
        print(f"{category.title()}:")
        for model, batches in values[category].items():
            for batch, statistics in batches.items():
                rendered = ", ".join(f"{key}={value}" for key, value in statistics.items())
                print(f"  {model} {batch}: {rendered}")
    print("\nQUALITY != CONDITIONING")
    print("Values are read from frozen DR04/DR05 evidence.")
    return 0


def _run_demo(_args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(
            "ERROR: The interactive demo requires the optional demo dependencies.\n"
            "Install them with:\n"
            'pip install -e ".[demo]"',
            file=sys.stderr,
        )
        return 1
    app = Path(__file__).with_name("streamlit_app.py")
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the single public CLI implementation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CliUsageError as error:
        parser.error(str(error))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
