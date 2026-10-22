"""Four-public-model Streamlit UI.  It deliberately reuses release samplers."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tiny_avatar_diffusion.inference import (
    PromptAttributes, UnconditionalInferenceBundle, generate_conditional,
    generate_unconditional, resolve_device, sha256_file, tensor_to_rgb_image,
)
from tiny_avatar_diffusion.release_paths import release_root
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule, unconditional_sample
from tiny_avatar_diffusion.unet import TinyUNet

ROOT = release_root()
MANIFEST_PATH = Path(__file__).with_name("public_model_manifest.json")

def public_models() -> tuple[dict[str, Any], ...]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    if payload.get("schema") != "tiny_avatar_diffusion.public_four_model_manifest.v1":
        raise ValueError("invalid public-model manifest")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("public manifest must define exactly four models")
    return tuple(models)

def model_by_id(model_id: str) -> dict[str, Any]:
    matches = [entry for entry in public_models() if entry["id"] == model_id]
    if len(matches) != 1: raise ValueError("unknown public model")
    return matches[0]

def checkpoint_status(entry: dict[str, Any]) -> tuple[bool, str]:
    if not entry.get("available"): return False, entry.get("unavailable_reason", "Not available.")
    path = Path(entry["checkpoint"])
    if not path.is_absolute(): path = ROOT / path
    if not path.is_file(): return False, "Authenticated checkpoint is not available locally."
    expected = entry.get("sha256")
    if expected and sha256_file(path) != expected: return False, "Checkpoint SHA-256 mismatch."
    return True, "Available"

def caption_from_values(values: dict[str, str]) -> PromptAttributes:
    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING
    raw = {}
    for attribute, label in values.items():
        found = [raw_id for raw_id, item in SEMANTIC_MAPPING[attribute].items() if item.canonical_label == label]
        if len(found) != 1: raise ValueError(f"unsupported {attribute}: {label}")
        raw[attribute] = found[0]
    return PromptAttributes(**raw)


def public_control_labels() -> dict[str, tuple[str, ...]]:
    """Expose only semantic, human-selectable values from the frozen mapping."""
    from tiny_avatar_diffusion.semantic_mapping import SEMANTIC_MAPPING
    return {
        attribute: tuple(
            value.canonical_label
            for value in SEMANTIC_MAPPING[attribute].values()
            if value.is_semantic
        )
        for attribute in ("face_shape", "glasses", "hair_color", "face_color")
    }

def load_v1_ema40(device: str) -> UnconditionalInferenceBundle:
    entry = model_by_id("DDPM-U32"); ok, message = checkpoint_status(entry)
    if not ok: raise RuntimeError(message)
    path = ROOT / entry["checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"artifact_format_version", "artifact_type", "model_role", "selected_step", "weight_variant", "source_checkpoint_sha256", "expected_parameter_count", "model_state_dict"}
    if (not isinstance(checkpoint, dict) or not required.issubset(checkpoint)
            or checkpoint["artifact_format_version"] != 1
            or checkpoint["artifact_type"] != "canonical_inference_checkpoint"
            or checkpoint["model_role"] != "v1_unconditional_40k"
            or checkpoint["selected_step"] != 40_000
            or checkpoint["weight_variant"] != "EMA"
            or checkpoint["source_checkpoint_sha256"] != entry["source_checkpoint_sha256"]
            or checkpoint["expected_parameter_count"] != entry["parameters"]):
        raise ValueError("DDPM-U32 EMA-40k package identity mismatch")
    model = TinyUNet(dropout=0.0); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != entry["parameters"]: raise ValueError("V1 parameter mismatch")
    resolved = resolve_device(device); model.to(resolved).eval(); schedule = LinearDDPMSchedule().to(resolved).eval()
    return UnconditionalInferenceBundle(model, schedule, resolved, device)

def generate_public(model_id: str, attributes: PromptAttributes | None, seed: int, device: str) -> Image.Image:
    entry = model_by_id(model_id); ok, message = checkpoint_status(entry)
    if not ok: raise RuntimeError(message)
    if entry["conditional"] and attributes is None: raise ValueError("This model requires all four structured attributes.")
    if not entry["conditional"] and attributes is not None: raise ValueError("Unconditional models do not accept a caption.")
    if model_id == "DDPM-U32": return generate_unconditional(load_v1_ema40(device), seed=seed).image
    if model_id == "DDPM-C32":
        from tiny_avatar_diffusion.inference import load_h7_conditional_bundle
        return generate_conditional(load_h7_conditional_bundle(device), seed=seed, attributes=attributes).image
    if model_id == "DDPM-C64":
        from tiny_avatar_diffusion.v2_release import generate_v2_native64
        from tiny_avatar_diffusion.v2a_ema60_release import load_v2a_conditional_ema60
        return generate_v2_native64(load_v2a_conditional_ema60(device, expected_sha256=entry["sha256"]), caption=attributes.caption, seed=seed)
    if model_id == "DDPM-U64":
        from tiny_avatar_diffusion.v2a_unconditional_ema60_release import generate_v2a_unconditional, load_v2a_unconditional_ema60
        return generate_v2a_unconditional(load_v2a_unconditional_ema60(device, expected_sha256=entry["sha256"]), seed=seed)
    raise RuntimeError("No authenticated adapter is registered for this model.")

def _png(image: Image.Image) -> bytes:
    stream = io.BytesIO(); image.save(stream, format="PNG"); return stream.getvalue()

def main() -> None:
    try: import streamlit as st
    except ModuleNotFoundError as error: raise RuntimeError('Install with: pip install -e ".[demo]"') from error
    st.set_page_config(page_title="Avatar Diffusion — Four Models", layout="centered")
    st.title("Avatar Diffusion")
    entries = public_models(); labels = {entry["display_name"]: entry for entry in entries}
    selected_label = st.selectbox("Public model", list(labels)); entry = labels[selected_label]
    ok, status = checkpoint_status(entry)
    st.caption(f"{entry['resolution']}×{entry['resolution']} · {'Conditional' if entry['conditional'] else 'Unconditional'} · {entry['parameters']:,} parameters")
    if not ok: st.warning(status)
    # Streamlit serializes number_input bounds through JavaScript Number.
    # Keep UI seeds within its exact-integer range; adapters remain stricter only
    # where needed and still accept this complete UI domain.
    seed = st.number_input("Seed", min_value=0, max_value=2**53 - 1, value=2026, step=1)
    attributes = None
    if entry["conditional"]:
        labels_by_attr = public_control_labels()
        values = {key: st.selectbox(key.replace("_", " ").title(), choices) for key, choices in labels_by_attr.items()}
        attributes = caption_from_values(values); st.caption(f"Training caption: `{attributes.caption}`")
    else: st.info("This unconditional model ignores captions and attributes.")
    if st.button("Generate", type="primary", disabled=not ok):
        try:
            with st.spinner("Running the authenticated 1,000-step DDPM sampler…"):
                image = generate_public(entry["id"], attributes, int(seed), "auto")
            st.image(image.resize((256, 256), Image.Resampling.NEAREST), caption=f"Native {image.width}×{image.height}")
            st.download_button("Download PNG", _png(image), file_name=f"{entry['id']}_seed{seed}.png", mime="image/png")
        except (OSError, RuntimeError, TypeError, ValueError) as error: st.error(str(error))
    with st.expander("Model information"): st.json({k: v for k, v in entry.items() if k not in {"checkpoint"}})
    st.caption("Fixed-seed determinism is implementation and hardware dependent. The constrained caption grammar is not open-vocabulary prompting.")

if __name__ == "__main__": main()
