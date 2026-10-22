"""Strict CPU identity coverage for the selected H7 release artifact."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiny_avatar_diffusion import inference
from tiny_avatar_diffusion.h7_release import (
    H7_DENOISER_PARAMETER_COUNT,
    H7_PARAMETER_COUNT,
    H7_SOURCE_SHA256,
    H7_TEXT_PARAMETER_COUNT,
    H7_VOCABULARY_SHA256,
)
from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

H7_ROOT = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_H7")
SOURCE_CHECKPOINT = H7_ROOT / "results/h7_ca16_40k_v1/h7_ca16_40k_v1_seed2026/terminal_40000.pt"
SOURCE_VOCABULARY = H7_ROOT / "data/processed/dr01_v1/text/10k/vocab.json"


@pytest.fixture(scope="module")
def source_checkpoint() -> dict[str, object]:
    if not SOURCE_CHECKPOINT.is_file():
        pytest.skip("authoritative local H7 source checkpoint is unavailable")
    assert inference.sha256_file(SOURCE_CHECKPOINT) == H7_SOURCE_SHA256
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    return checkpoint


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_h7_artifact_is_exact_ema_extraction(source_checkpoint: dict[str, object]) -> None:
    assert inference.sha256_file(inference.H7_CONDITIONAL_CHECKPOINT) == inference.H7_CONDITIONAL_SHA256
    artifact = torch.load(inference.H7_CONDITIONAL_CHECKPOINT, map_location="cpu", weights_only=True)
    assert artifact["source_checkpoint_sha256"] == H7_SOURCE_SHA256
    assert artifact["selected_step"] == 40_000
    assert artifact["weight_variant"] == "EMA"
    assert artifact["vocabulary_sha256"] == H7_VOCABULARY_SHA256
    source_ema = source_checkpoint["ema_state_dict"]
    assert isinstance(source_ema, dict)
    for component in ("text_encoder", "conditional_unet"):
        assert set(artifact["model_state_dict"][component]) == set(source_ema[component])
        for key, value in source_ema[component].items():
            assert torch.equal(artifact["model_state_dict"][component][key], value)


def test_h7_loader_tokenizer_and_forward_are_cpu_exact(source_checkpoint: dict[str, object]) -> None:
    bundle = inference.load_h7_conditional_bundle("cpu")
    assert bundle.device.type == "cpu"
    assert bundle.model.use_decoder16_cross_attention is True
    assert _count(bundle.model) == H7_DENOISER_PARAMETER_COUNT
    assert _count(bundle.text_encoder) == H7_TEXT_PARAMETER_COUNT
    assert _count(bundle.model) + _count(bundle.text_encoder) == H7_PARAMETER_COUNT
    encoded = bundle.tokenizer.encode(
        "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"
    )
    assert encoded.token_ids == [1, 4, 22, 34, 44, 6, 2]
    assert encoded.attention_mask == [1, 1, 1, 1, 1, 1, 1]

    source_ema = source_checkpoint["ema_state_dict"]
    assert isinstance(source_ema, dict)
    original_text = TextConditioningEncoder(TextConditioningConfig()).eval()
    original_model = ConditionalTinyUNet(dropout=0.0, use_decoder16_cross_attention=True).eval()
    original_text.load_state_dict(source_ema["text_encoder"], strict=True)
    original_model.load_state_dict(source_ema["conditional_unet"], strict=True)
    input_ids = torch.tensor([encoded.token_ids], dtype=torch.long)
    mask = torch.tensor([encoded.attention_mask], dtype=torch.long)
    x_t = torch.linspace(-1.0, 1.0, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    ca8_calls: list[torch.Size] = []
    ca16_calls: list[torch.Size] = []
    hooks = [
        bundle.model.image_text_cross_attention.register_forward_hook(
            lambda _m, args, _out: ca8_calls.append(args[0].shape)
        ),
        bundle.model.decoder16_image_text_cross_attention.register_forward_hook(
            lambda _m, args, _out: ca16_calls.append(args[0].shape)
        ),
    ]
    try:
        with torch.inference_mode():
            source_context = original_text(input_ids, mask)
            released_context = bundle.text_encoder(input_ids, mask)
            assert torch.equal(released_context, source_context)
            for timestep in (0, 499, 999):
                t = torch.tensor([timestep], dtype=torch.long)
                source_epsilon = original_model(x_t, t, source_context, mask)
                released_epsilon = bundle.model(x_t, t, released_context, mask)
                assert released_epsilon.shape == (1, 3, 32, 32)
                assert torch.isfinite(released_epsilon).all()
                assert torch.equal(released_epsilon, source_epsilon)
    finally:
        for hook in hooks:
            hook.remove()
    assert ca8_calls == [torch.Size((1, 128, 8, 8))] * 3
    assert ca16_calls == [torch.Size((1, 128, 16, 16))] * 3


def test_v1_conditional_loader_remains_available() -> None:
    bundle = inference.load_conditional_bundle("cpu")
    assert bundle.model_identity == "conditional_v1"
    assert bundle.model.use_decoder16_cross_attention is False
