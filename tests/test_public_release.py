"""CPU-only regression coverage for the four artifacts distributed publicly.

These checks deliberately cover only the stable public interface. Historical
campaign tests live elsewhere in the repository and can require non-distributed
datasets, checkpoints, or experimental outputs.
"""

from tiny_avatar_diffusion import final_streamlit_app as app
from tiny_avatar_diffusion.inference import load_h7_conditional_bundle
from tiny_avatar_diffusion.v2a_ema60_release import load_v2a_conditional_ema60
from tiny_avatar_diffusion.v2a_unconditional_ema60_release import load_v2a_unconditional_ema60


EXPECTED_MODELS = ("DDPM-U32", "DDPM-C32", "DDPM-U64", "DDPM-C64")


def test_public_manifest_has_the_four_stable_models() -> None:
    assert tuple(entry["id"] for entry in app.public_models()) == EXPECTED_MODELS


def test_all_distributed_checkpoints_match_their_public_manifest_hashes() -> None:
    for entry in app.public_models():
        available, message = app.checkpoint_status(entry)
        assert available, f"{entry['id']}: {message}"


def test_all_public_model_adapters_load_on_cpu() -> None:
    entries = {entry["id"]: entry for entry in app.public_models()}

    assert app.load_v1_ema40("cpu").device.type == "cpu"
    assert load_h7_conditional_bundle("cpu").device.type == "cpu"
    assert (
        load_v2a_unconditional_ema60(
            "cpu", expected_sha256=entries["DDPM-U64"]["sha256"]
        ).device.type
        == "cpu"
    )
    assert (
        load_v2a_conditional_ema60(
            "cpu", expected_sha256=entries["DDPM-C64"]["sha256"]
        ).device.type
        == "cpu"
    )
