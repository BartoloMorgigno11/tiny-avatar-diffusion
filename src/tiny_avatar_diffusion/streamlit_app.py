"""Optional Streamlit interface for Tiny Avatar Diffusion V1."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from PIL import Image

try:
    import streamlit as st
except ModuleNotFoundError:  # Streamlit is an optional dependency.
    st = None  # type: ignore[assignment]

SNAPSHOT_TIMESTEPS = (999, 749, 499, 249, 0)
DEFAULT_SEED = 2026
MAX_SEED = 2**63 - 1
DEFAULT_DEVICE = "Auto"
DEFAULT_DENOISING = True
SNAPSHOT_LABELS = (
    "Initial x_T",
    "Predicted x₀ at t=999",
    "Predicted x₀ at t=749",
    "Predicted x₀ at t=499",
    "Predicted x₀ at t=249",
    "Predicted x₀ at t=0",
)
STEP_FILENAMES = (
    "00_initial_noise.png",
    "01_pred_x0_t0999.png",
    "02_pred_x0_t0749.png",
    "03_pred_x0_t0499.png",
    "04_pred_x0_t0249.png",
    "05_pred_x0_t0000.png",
)


@dataclass(frozen=True)
class GenerationDownloads:
    png_filename: str
    png: bytes
    metadata_filename: str
    metadata: bytes
    steps_filename: str | None
    steps_zip: bytes | None
    created_at_utc: str


@dataclass(frozen=True)
class PairDownloads:
    conditional_png_filename: str
    conditional_png: bytes
    unconditional_png_filename: str
    unconditional_png: bytes
    metadata_filename: str
    metadata: bytes
    steps_filename: str | None
    steps_zip: bytes | None
    created_at_utc: str


def utc_now() -> str:
    """Return a timezone-aware UTC serialization timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def png_bytes(image: Image.Image) -> bytes:
    """Serialize one canonical scientific image without writing a file."""
    if image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("download image must be RGB 32x32")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def display_copy(image: Image.Image, size: int = 256) -> Image.Image:
    """Create a display-only nearest-neighbor enlargement."""
    if image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("display image must be RGB 32x32")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def result_filename(result: Any) -> str:
    generation = result.provenance.generation
    return (
        f"{generation['model_variant']}_seed{generation['seed']}_"
        f"{generation['pixel_content_sha256'][:8]}.png"
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _snapshot_entries(result: Any) -> list[tuple[str, str, bytes]]:
    from tiny_avatar_diffusion.inference import tensor_to_rgb_image

    snapshots = {item.timestep: item for item in result.snapshots}
    if tuple(snapshots) != SNAPSHOT_TIMESTEPS:
        raise ValueError("backend did not return the frozen UI snapshot schedule")
    entries = [
        (
            STEP_FILENAMES[0],
            SNAPSHOT_LABELS[0],
            png_bytes(tensor_to_rgb_image(result.initial_x_T[0])),
        )
    ]
    for index, timestep in enumerate(SNAPSHOT_TIMESTEPS, 1):
        tensor = snapshots[timestep].predicted_x_start
        if tensor.shape == (1, 3, 32, 32):
            tensor = tensor[0]
        entries.append(
            (
                STEP_FILENAMES[index],
                SNAPSHOT_LABELS[index],
                png_bytes(tensor_to_rgb_image(tensor)),
            )
        )
    return entries


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for filename, content in entries:
            member = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            archive.writestr(member, content)
    return stream.getvalue()


def prepare_generation_downloads(
    result: Any,
    diagnostic: dict[str, Any] | None,
    *,
    created_at_utc: str | None = None,
) -> GenerationDownloads:
    """Prepare a stable in-memory download bundle for one result."""
    created_at = created_at_utc or utc_now()
    filename = result_filename(result)
    png = png_bytes(result.image)
    serialization: dict[str, Any] = {
        "created_at_utc": created_at,
        "png_filename": filename,
        "png_sha256": hashlib.sha256(png).hexdigest(),
    }
    steps_filename = None
    steps_zip = None
    if result.snapshots:
        entries = _snapshot_entries(result)
        steps_filename = f"{filename[:-4]}_steps.zip"
        steps_zip = _zip_bytes([(name, content) for name, _label, content in entries])
        serialization["steps"] = {
            "initial_noise_display": "x_T clipped to [-1, 1] for display only",
            "captured_timesteps": list(SNAPSHOT_TIMESTEPS),
            "zip_filename": steps_filename,
            "files": [
                {"filename": name, "png_sha256": hashlib.sha256(content).hexdigest()}
                for name, _label, content in entries
            ],
        }
    metadata = _json_bytes(
        {
            "provenance": result.provenance.to_dict(),
            "diagnostics": diagnostic,
            "serialization": serialization,
        }
    )
    return GenerationDownloads(
        png_filename=filename,
        png=png,
        metadata_filename=f"{filename[:-4]}.json",
        metadata=metadata,
        steps_filename=steps_filename,
        steps_zip=steps_zip,
        created_at_utc=created_at,
    )


def prepare_pair_downloads(
    pair: Any,
    conditional_diagnostic: dict[str, Any] | None,
    unconditional_diagnostic: dict[str, Any] | None,
    *,
    created_at_utc: str | None = None,
) -> PairDownloads:
    """Prepare stable paired images, metadata, and optional steps in memory."""
    created_at = created_at_utc or utc_now()
    conditional_name = f"compare_{result_filename(pair.conditional)}"
    unconditional_name = f"compare_{result_filename(pair.unconditional)}"
    conditional_png = png_bytes(pair.conditional.image)
    unconditional_png = png_bytes(pair.unconditional.image)
    seed = pair.pairing.seed
    conditional_prefix = pair.conditional.provenance.generation[
        "pixel_content_sha256"
    ][:8]
    unconditional_prefix = pair.unconditional.provenance.generation[
        "pixel_content_sha256"
    ][:8]
    base = f"compare_seed{seed}_{conditional_prefix}_{unconditional_prefix}"
    conditional_serialization: dict[str, Any] = {
        "png_filename": conditional_name,
        "png_sha256": hashlib.sha256(conditional_png).hexdigest(),
    }
    unconditional_serialization: dict[str, Any] = {
        "png_filename": unconditional_name,
        "png_sha256": hashlib.sha256(unconditional_png).hexdigest(),
    }
    steps_filename = None
    steps_zip = None
    if pair.conditional.snapshots or pair.unconditional.snapshots:
        conditional_steps = _snapshot_entries(pair.conditional)
        unconditional_steps = _snapshot_entries(pair.unconditional)
        if conditional_steps[0][2] != unconditional_steps[0][2]:
            raise ValueError("paired initial-noise display PNGs are not identical")
        zip_entries = [
            (f"conditional/{name}", content)
            for name, _label, content in conditional_steps
        ] + [
            (f"unconditional/{name}", content)
            for name, _label, content in unconditional_steps
        ]
        steps_filename = f"{base}_steps.zip"
        steps_zip = _zip_bytes(zip_entries)
        conditional_serialization["steps"] = {
            "captured_timesteps": list(SNAPSHOT_TIMESTEPS),
            "zip_prefix": "conditional/",
        }
        unconditional_serialization["steps"] = {
            "captured_timesteps": list(SNAPSHOT_TIMESTEPS),
            "zip_prefix": "unconditional/",
        }
    metadata = _json_bytes(
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
    )
    return PairDownloads(
        conditional_png_filename=conditional_name,
        conditional_png=conditional_png,
        unconditional_png_filename=unconditional_name,
        unconditional_png=unconditional_png,
        metadata_filename=f"{base}.json",
        metadata=metadata,
        steps_filename=steps_filename,
        steps_zip=steps_zip,
        created_at_utc=created_at,
    )


def canonical_choices(attribute: str) -> tuple[str, ...]:
    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING

    return tuple(value.canonical_label for value in SEMANTIC_MAPPING[attribute].values())


def evaluator_default(mode: str) -> bool:
    """Return the frozen evaluator default for a generation mode."""
    if mode not in {"Conditional", "Unconditional"}:
        raise ValueError("generation mode must be Conditional or Unconditional")
    return mode == "Conditional"


def parse_seed(value: str) -> int:
    """Parse the full frozen backend seed domain without a JavaScript number."""
    if not value:
        raise ValueError("Seed is required.")
    if not value.isascii() or not value.isdecimal():
        raise ValueError("Seed must be a base-10 integer.")
    seed = int(value, 10)
    if seed > MAX_SEED:
        raise ValueError(f"Seed must be in [0, {MAX_SEED}].")
    return seed


def structured_attributes(values: dict[str, str]) -> Any:
    from tiny_avatar_diffusion.inference import PromptAttributes
    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING

    raw_values = {
        attribute: next(
            raw_id
            for raw_id, semantic in SEMANTIC_MAPPING[attribute].items()
            if semantic.canonical_label == label
        )
        for attribute, label in values.items()
    }
    return PromptAttributes(**raw_values)


def load_conditional_resource(device: str) -> Any:
    from tiny_avatar_diffusion.inference import load_conditional_bundle

    return load_conditional_bundle(device)


def load_unconditional_resource(device: str) -> Any:
    from tiny_avatar_diffusion.inference import load_unconditional_bundle

    return load_unconditional_bundle(device)


def load_evaluator_resource(device: str) -> Any:
    from tiny_avatar_diffusion.inference import load_evaluator_bundle

    return load_evaluator_bundle(device)


def load_frozen_data() -> dict[str, Any]:
    from tiny_avatar_diffusion.frozen_results import load_frozen_results

    return load_frozen_results()


if st is not None:
    load_conditional_resource = st.cache_resource(load_conditional_resource)
    load_unconditional_resource = st.cache_resource(load_unconditional_resource)
    load_evaluator_resource = st.cache_resource(load_evaluator_resource)
    load_frozen_data = st.cache_data(load_frozen_data)


def _attribute_controls(prefix: str) -> dict[str, str]:
    labels = {
        "face_shape": "Face shape",
        "glasses": "Glasses",
        "hair_color": "Hair color",
        "face_color": "Face color",
    }
    return {
        attribute: st.selectbox(
            label,
            canonical_choices(attribute),
            key=f"{prefix}_{attribute}",
        )
        for attribute, label in labels.items()
    }


def _conditional_description(prefix: str) -> tuple[bool, str, dict[str, str] | None]:
    with st.expander("Advanced input", expanded=False):
        use_prompt = st.checkbox(
            "Use strict canonical prompt",
            value=False,
            key=f"{prefix}_use_prompt",
        )
    if use_prompt:
        prompt = st.text_input(
            "Strict canonical prompt",
            placeholder="avatar <face_shape> <glasses> <hair_color> <face_color>",
            key=f"{prefix}_prompt",
        )
        st.caption("Strict canonical prompt is the active conditional input source.")
        return True, prompt, None
    st.caption("Structured attributes are the active conditional input source.")
    return False, "", _attribute_controls(prefix)


def _resolve_attributes(
    use_prompt: bool, prompt: str, values: dict[str, str] | None
) -> Any:
    if use_prompt:
        from tiny_avatar_diffusion.inference import validate_raw_caption

        return validate_raw_caption(prompt)
    if values is None:
        raise ValueError("structured attributes are required")
    return structured_attributes(values)


def _diagnostic_rows(diagnostic: dict[str, Any], requested: bool) -> list[dict[str, Any]]:
    rows = []
    for attribute, head in diagnostic["attributes"].items():
        predicted_id = head["predicted_raw_id"]
        score = next(
            item["score"]
            for item in head["uncalibrated_evaluator_scores"]
            if item["raw_id"] == predicted_id
        )
        if requested:
            row = {
                "Attribute": attribute,
                "Requested": head["requested_label"],
                "Predicted": head["predicted_label"],
                "Uncalibrated evaluator score": score,
                "Match": head["match"],
            }
        else:
            row = {
                "Attribute": attribute,
                "Predicted": head["predicted_label"],
                "Uncalibrated evaluator score": score,
            }
        rows.append(row)
    return rows


def _show_diagnostic(
    diagnostic: dict[str, Any], *, requested: bool, comparison_note: bool = False
) -> None:
    st.markdown("**LIVE SINGLE-SAMPLE DIAGNOSTIC**")
    if comparison_note:
        st.caption(
            "Match against requested comparison attributes; these attributes were not "
            "supplied to the unconditional generator."
        )
    st.dataframe(_diagnostic_rows(diagnostic, requested), hide_index=True)
    if requested:
        st.write(f"Attributes matched: {diagnostic['attributes_matched']} / 4")
    st.caption(
        "Scores are uncalibrated evaluator scores, not probabilities of correctness."
    )


def _show_steps(
    result: Any, *, include_initial: bool = True, include_final: bool = True
) -> None:
    entries = _snapshot_entries(result)
    if include_initial:
        st.caption("Display-only clipped visualization of the actual Gaussian x_T.")
    else:
        entries = entries[1:]
    items = [(label, Image.open(io.BytesIO(content))) for _name, label, content in entries]
    if include_final:
        items.append(("Final image", result.image))
    columns = st.columns(len(items))
    for column, (label, image) in zip(columns, items, strict=True):
        column.image(display_copy(image, 128), caption=label)


def _technical_details(result: Any) -> dict[str, Any]:
    provenance = result.provenance.to_dict()
    generation = provenance["generation"]
    model = provenance["model"]
    diffusion = provenance["diffusion"]
    runtime = provenance["runtime"]
    software = provenance["software"]
    keys = {
        "model_variant": generation.get("model_variant"),
        "checkpoint_sha256": model.get("checkpoint_sha256"),
        "selected_step": model.get("selected_step"),
        "weight_variant": model.get("weights_variant"),
        "seed": generation.get("seed"),
        "requested_device": runtime.get("requested_device"),
        "resolved_device": runtime.get("resolved_device"),
        "pixel_content_sha256": generation.get("pixel_content_sha256"),
        "timesteps": diffusion.get("timesteps"),
        "schedule_identity": diffusion.get("schedule_identity"),
        "beta_start": diffusion.get("beta_start"),
        "beta_end": diffusion.get("beta_end"),
        "prediction_type": diffusion.get("prediction_type"),
        "variance_type": diffusion.get("variance_type"),
        "clip_denoised": diffusion.get("clip_denoised"),
        "precision": diffusion.get("precision"),
        "git_commit": software.get("git_commit"),
        "git_dirty": software.get("git_dirty"),
    }
    return {key: value for key, value in keys.items() if value is not None}


def _show_generation_result(state: dict[str, Any]) -> None:
    result = state["result"]
    downloads: GenerationDownloads = state["downloads"]
    generation = result.provenance.generation
    st.subheader("Last generated result")
    st.image(display_copy(result.image), caption="Display-only nearest-neighbor preview")
    st.write(f"Model variant: {generation['model_variant']}")
    if generation.get("canonical_prompt") is not None:
        st.write(f"Canonical prompt: {generation['canonical_prompt']}")
    st.write(f"Seed: {generation['seed']}")
    if state["diagnostic"] is not None:
        _show_diagnostic(
            state["diagnostic"],
            requested=generation["model_variant"] == "conditional",
        )
    if result.snapshots:
        with st.expander("View denoising process", expanded=False):
            _show_steps(result)
    with st.expander("Technical details", expanded=False):
        st.json(_technical_details(result))
    st.download_button(
        "Download PNG",
        downloads.png,
        file_name=downloads.png_filename,
        mime="image/png",
    )
    st.download_button(
        "Download metadata JSON",
        downloads.metadata,
        file_name=downloads.metadata_filename,
        mime="application/json",
    )
    if downloads.steps_zip is not None and downloads.steps_filename is not None:
        st.download_button(
            "Download denoising steps (.zip)",
            downloads.steps_zip,
            file_name=downloads.steps_filename,
            mime="application/zip",
        )


def _generate_tab() -> None:
    st.header("Generate")
    mode = st.radio("Generation mode", ("Conditional", "Unconditional"), horizontal=True)
    use_prompt = False
    prompt = ""
    values = None
    if mode == "Conditional":
        use_prompt, prompt, values = _conditional_description("generate")
    seed_text = st.text_input(
        "Seed",
        value=str(DEFAULT_SEED),
        key="generate_seed",
    )
    with st.expander("Runtime controls", expanded=False):
        device = st.selectbox(
            "Device",
            ("Auto", "CUDA", "CPU"),
            index=("Auto", "CUDA", "CPU").index(DEFAULT_DEVICE),
            key="generate_device",
        )
        evaluate = st.checkbox(
            "Evaluate generated image",
            value=evaluator_default(mode),
            key=f"generate_evaluate_{mode.lower()}",
        )
        show_steps = st.checkbox("Show denoising process", value=DEFAULT_DENOISING)
    if st.button("Generate avatar", type="primary"):
        try:
            from tiny_avatar_diffusion.inference import (
                evaluator_diagnostic,
                generate_conditional,
                generate_unconditional,
            )

            seed = parse_seed(seed_text)
            requested = (
                _resolve_attributes(use_prompt, prompt, values)
                if mode == "Conditional"
                else None
            )
            snapshots = SNAPSHOT_TIMESTEPS if show_steps else None
            resolved_device = device.lower()
            if mode == "Conditional":
                result = generate_conditional(
                    load_conditional_resource(resolved_device),
                    seed=seed,
                    attributes=requested,
                    snapshot_timesteps=snapshots,
                )
            else:
                result = generate_unconditional(
                    load_unconditional_resource(resolved_device),
                    seed=seed,
                    snapshot_timesteps=snapshots,
                )
            diagnostic = None
            if evaluate:
                diagnostic = evaluator_diagnostic(
                    load_evaluator_resource(resolved_device),
                    result.image,
                    requested=requested,
                )
            st.session_state["latest_generate"] = {
                "result": result,
                "diagnostic": diagnostic,
                "downloads": prepare_generation_downloads(result, diagnostic),
            }
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            st.error(str(error))
    state = st.session_state.get("latest_generate")
    if state is not None:
        _show_generation_result(state)


def _show_pair_result(state: dict[str, Any]) -> None:
    pair = state["pair"]
    downloads: PairDownloads = state["downloads"]
    st.subheader("Last paired result")
    st.info(
        "QUALITATIVE PAIRED COMPARISON\n\n"
        "Same initial x_T and equivalent reverse-noise RNG state.\n\n"
        "Not a dataset-level metric."
    )
    conditional_column, unconditional_column = st.columns(2)
    with conditional_column:
        st.markdown("**CONDITIONAL**")
        st.image(display_copy(pair.conditional.image))
        st.caption(
            f"Seed {pair.pairing.seed} · Pixel SHA256 "
            f"{pair.conditional.provenance.generation['pixel_content_sha256']}"
        )
        if state["conditional_diagnostic"] is not None:
            _show_diagnostic(state["conditional_diagnostic"], requested=True)
    with unconditional_column:
        st.markdown("**UNCONDITIONAL**")
        st.image(display_copy(pair.unconditional.image))
        st.caption(
            f"Seed {pair.pairing.seed} · Pixel SHA256 "
            f"{pair.unconditional.provenance.generation['pixel_content_sha256']}"
        )
        if state["unconditional_diagnostic"] is not None:
            _show_diagnostic(
                state["unconditional_diagnostic"],
                requested=True,
                comparison_note=True,
            )
    if pair.conditional.snapshots:
        with st.expander("View paired denoising process", expanded=False):
            st.markdown("**Shared initial noise**")
            shared_initial = _snapshot_entries(pair.conditional)[0][2]
            st.image(
                display_copy(Image.open(io.BytesIO(shared_initial)), 128),
                caption="Display-only clipped visualization of the shared actual Gaussian x_T.",
            )
            st.markdown("**Conditional**")
            _show_steps(pair.conditional, include_initial=False)
            st.markdown("**Unconditional**")
            _show_steps(pair.unconditional, include_initial=False)
    with st.expander("Technical details", expanded=False):
        st.markdown("**Conditional**")
        st.json(_technical_details(pair.conditional))
        st.markdown("**Unconditional**")
        st.json(_technical_details(pair.unconditional))
    st.download_button(
        "Download conditional PNG",
        downloads.conditional_png,
        file_name=downloads.conditional_png_filename,
        mime="image/png",
    )
    st.download_button(
        "Download unconditional PNG",
        downloads.unconditional_png,
        file_name=downloads.unconditional_png_filename,
        mime="image/png",
    )
    st.download_button(
        "Download comparison metadata JSON",
        downloads.metadata,
        file_name=downloads.metadata_filename,
        mime="application/json",
    )
    if downloads.steps_zip is not None and downloads.steps_filename is not None:
        st.download_button(
            "Download denoising steps (.zip)",
            downloads.steps_zip,
            file_name=downloads.steps_filename,
            mime="application/zip",
        )


def _compare_tab() -> None:
    st.header("Compare")
    use_prompt, prompt, values = _conditional_description("compare")
    seed_text = st.text_input(
        "Seed",
        value=str(DEFAULT_SEED),
        key="compare_seed",
    )
    with st.expander("Runtime controls", expanded=False):
        device = st.selectbox(
            "Device",
            ("Auto", "CUDA", "CPU"),
            index=("Auto", "CUDA", "CPU").index(DEFAULT_DEVICE),
            key="compare_device",
        )
        evaluate = st.checkbox("Evaluate both branches", value=True)
        show_steps = st.checkbox("Show paired denoising process", value=True)
    if st.button("Generate paired comparison", type="primary"):
        try:
            from tiny_avatar_diffusion.inference import (
                evaluator_diagnostic,
                generate_paired,
            )

            seed = parse_seed(seed_text)
            requested = _resolve_attributes(use_prompt, prompt, values)
            resolved_device = device.lower()
            pair = generate_paired(
                load_conditional_resource(resolved_device),
                load_unconditional_resource(resolved_device),
                seed=seed,
                attributes=requested,
                snapshot_timesteps=SNAPSHOT_TIMESTEPS if show_steps else None,
            )
            conditional_diagnostic = None
            unconditional_diagnostic = None
            if evaluate:
                evaluator = load_evaluator_resource(resolved_device)
                conditional_diagnostic = evaluator_diagnostic(
                    evaluator, pair.conditional.image, requested=requested
                )
                unconditional_diagnostic = evaluator_diagnostic(
                    evaluator, pair.unconditional.image, requested=requested
                )
            st.session_state["latest_compare"] = {
                "pair": pair,
                "conditional_diagnostic": conditional_diagnostic,
                "unconditional_diagnostic": unconditional_diagnostic,
                "downloads": prepare_pair_downloads(
                    pair, conditional_diagnostic, unconditional_diagnostic
                ),
            }
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            st.error(str(error))
    state = st.session_state.get("latest_compare")
    if state is not None:
        _show_pair_result(state)


def _frozen_results_tab() -> None:
    st.header("Frozen Results")
    try:
        results = load_frozen_data()
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Frozen evidence could not be loaded: {error}")
        return
    conditioning = results["conditioning"]
    st.warning("QUALITY ≠ CONDITIONING")
    st.write(
        "KID/FID describe distribution-level generative quality, while MAA and "
        "MacroPSR measure adherence to requested structured attributes. The conditional "
        "model has better generative-quality metrics than the unconditional baseline, "
        "while conditioning adherence remains limited, especially on unseen combinations."
    )

    st.subheader("Conditioning adherence")
    columns = st.columns(4)
    columns[0].metric("ID MAA", conditioning["id_maa"])
    columns[1].metric("OOD MAA", conditioning["ood_maa"])
    columns[2].metric("Absolute ID → OOD gap", conditioning["absolute_maa_gap"])
    columns[3].metric("MacroPSR", conditioning["macro_psr"])
    st.caption(
        "MAA — Mean Attribute Accuracy: average correctness across evaluated requested "
        "attributes. MacroPSR — Macro Prompt Success Rate: prompt-level compositional "
        "success under the frozen evaluation protocol."
    )

    st.subheader("ID vs compositional OOD")
    chart_data = [
        {
            "Split": "ID",
            "MAA": conditioning["id_maa"],
            "Lower": conditioning["id_maa_bootstrap_95"]["lower"],
            "Upper": conditioning["id_maa_bootstrap_95"]["upper"],
        },
        {
            "Split": "OOD",
            "MAA": conditioning["ood_maa"],
            "Lower": conditioning["ood_maa_bootstrap_95"]["lower"],
            "Upper": conditioning["ood_maa_bootstrap_95"]["upper"],
        },
    ]
    st.vega_lite_chart(
        chart_data,
        {
            "layer": [
                {
                    "mark": {"type": "bar", "width": 48},
                    "encoding": {
                        "x": {"field": "Split", "type": "nominal"},
                        "y": {"field": "MAA", "type": "quantitative"},
                    },
                },
                {
                    "mark": {"type": "rule", "strokeWidth": 3},
                    "encoding": {
                        "x": {"field": "Split", "type": "nominal"},
                        "y": {"field": "Lower", "type": "quantitative"},
                        "y2": {"field": "Upper"},
                    },
                },
            ]
        },
        use_container_width=True,
    )
    st.write(f"Absolute ID → OOD gap: {conditioning['absolute_maa_gap']} (point estimate only)")
    st.caption(
        "The displayed 95% bootstrap intervals quantify uncertainty separately for ID "
        "and OOD MAA. No canonical bootstrap confidence interval is available for the "
        "ID→OOD gap, so the gap is reported as a point estimate only. The intervals are "
        "not a significance test, p-value, or evidence of causality."
    )

    st.subheader("Generative quality")
    quality_rows = []
    for model, splits in results["quality_by_split"].items():
        for split, metrics in splits.items():
            quality_rows.append(
                {
                    "Model": model.title(),
                    "Split": split.upper(),
                    "KID": metrics["kid"],
                    "FID": metrics["fid"],
                }
            )
    st.dataframe(quality_rows, hide_index=True)
    st.caption("Lower KID/FID is better under this evaluation protocol.")
    st.caption(
        "These metrics assess distribution-level generative quality, not whether a "
        "requested attribute combination was followed."
    )

    st.subheader("Diversity & engineering")
    st.metric(
        "Prompt-macro LPIPS",
        results["lpips_diversity_global_prompt_macro"],
    )
    st.caption(
        "Higher LPIPS indicates greater perceptual variation under the evaluated protocol."
    )
    engineering_rows = []
    for model in ("conditional", "unconditional"):
        for batch in ("batch_1", "batch_16"):
            engineering_rows.append(
                {
                    "Model": model.title(),
                    "Batch": batch.removeprefix("batch_"),
                    "Parameters": results["parameter_counts"][model],
                    "Median seconds/image": results["latency"][model][batch][
                        "median_seconds_per_image"
                    ],
                    "Peak allocated MiB": results["memory"][model][batch][
                        "peak_allocated_mib"
                    ],
                }
            )
    st.dataframe(engineering_rows, hide_index=True)
    st.caption(
        "Latency and memory are measurements from the frozen evaluation environment and "
        "protocol, not universal hardware benchmarks."
    )
    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            "- **MAA:** mean correctness across requested attributes.\n"
            "- **MacroPSR:** prompt-level compositional success under the frozen protocol.\n"
            "- **KID/FID:** distribution-level distance metrics; lower is better here.\n"
            "- **LPIPS:** perceptual variation under the evaluated sampling protocol."
        )


def main() -> None:
    """Construct the three-tab optional interactive application."""
    if st is None:
        raise RuntimeError(
            "The interactive demo requires the optional demo dependencies. "
            'Install them with: pip install -e ".[demo]"'
        )
    st.set_page_config(page_title="Tiny Avatar Diffusion", layout="wide")
    st.title("Tiny Avatar Diffusion")
    st.caption("From-scratch text-conditioned DDPM for 32×32 cartoon avatars.")
    generate_tab, compare_tab, results_tab = st.tabs(
        ("Generate", "Compare", "Frozen Results")
    )
    with generate_tab:
        _generate_tab()
    with compare_tab:
        _compare_tab()
    with results_tab:
        _frozen_results_tab()


if __name__ == "__main__":
    main()
