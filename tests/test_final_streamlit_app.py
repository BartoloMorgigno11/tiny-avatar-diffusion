from PIL import Image

from tiny_avatar_diffusion import final_streamlit_app as app
from tiny_avatar_diffusion import v2a_unconditional_ema60_release as u64


def test_ui_seed_limit_is_javascript_safe():
    source = app.main.__code__.co_consts
    assert 2**53 - 1 in source

def test_public_manifest_has_exactly_four_expected_public_entries():
    models = app.public_models()
    assert [item["id"] for item in models] == ["DDPM-U32", "DDPM-C32", "DDPM-U64", "DDPM-C64"]
    assert app.model_by_id("DDPM-U64")["available"] is True
    assert app.model_by_id("DDPM-C64")["available"] is True


def test_v1_public_entry_uses_the_ema_package_not_its_full_state_source():
    entry = app.model_by_id("DDPM-U32")
    assert entry["checkpoint"] == "artifacts/checkpoints/ddpm-u32-ema-40k.pt"
    assert entry["sha256"] == "578aa9301d19910f323332f46062b3803b8f8ad5236d67e9e0cbfb11a5694974"
    assert entry["source_checkpoint_sha256"] == "b0dbc80f07e02cbed257ef8a4d6faf8c194651f10ed13cecbcdd591d1acb5d89"

def test_prompt_uses_authentic_canonical_grammar():
    attributes = app.caption_from_values({"face_shape":"wide", "glasses":"thick_rectangular_glasses", "hair_color":"silver", "face_color":"deep_brown"})
    assert attributes.caption == "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"

def test_public_controls_exclude_neutral_face_shape_token():
    labels = app.public_control_labels()
    assert "face_shape_1" not in labels["face_shape"]
    assert labels["face_shape"] == ("pointed", "oval", "long", "round", "wide", "rectangular")

def test_missing_checkpoint_is_not_silently_substituted():
    entry = dict(app.model_by_id("DDPM-U64")); entry["checkpoint"] = "artifacts/checkpoints/missing.pt"
    ok, message = app.checkpoint_status(entry)
    assert not ok and "not available locally" in message.lower()

def test_unconditional_rejects_caption_without_loading_or_sampling():
    attributes = app.caption_from_values({"face_shape":"wide", "glasses":"thick_rectangular_glasses", "hair_color":"silver", "face_color":"deep_brown"})
    try:
        app.generate_public("DDPM-U64", attributes, 2026, "cpu")
    except ValueError as error:
        assert "do not accept a caption" in str(error)
    else:
        assert False

def test_registered_v2a_conditional_ema60_package_is_available():
    ok, message = app.checkpoint_status(app.model_by_id("DDPM-C64"))
    assert ok, message

def test_registered_v2a_unconditional_ema60_package_is_available():
    ok, message = app.checkpoint_status(app.model_by_id("DDPM-U64"))
    assert ok, message

def test_all_public_checkpoint_paths_are_authenticated():
    for entry in app.public_models():
        ok, message = app.checkpoint_status(entry)
        assert ok, f"{entry['id']}: {message}"

def test_unconditional_adapter_loads_on_cpu_without_tokens_or_sampling():
    bundle = u64.load_v2a_unconditional_ema60("cpu", expected_sha256=app.model_by_id("DDPM-U64")["sha256"])
    assert bundle.device.type == "cpu"
    assert bundle.checkpoint_sha256 == app.model_by_id("DDPM-U64")["sha256"]
    assert bundle.model.__class__.__name__ == "V2AUnconditionalNative64UNet"

def test_v1_model_switch_uses_only_its_selected_adapter(monkeypatch):
    sentinel = object()
    observed = []
    monkeypatch.setattr(app, "load_v1_ema40", lambda device: sentinel)
    monkeypatch.setattr(app, "generate_unconditional", lambda bundle, seed: (observed.append((bundle, seed)) or type("Result", (), {"image": sentinel})()))
    assert app.generate_public("DDPM-U32", None, 1234, "cpu") is sentinel
    assert observed == [(sentinel, 1234)]

def test_png_download_payload_is_png():
    assert app._png(Image.new("RGB", (1, 1))).startswith(b"\x89PNG\r\n\x1a\n")
