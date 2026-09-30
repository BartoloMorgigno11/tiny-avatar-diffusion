import inspect

import pytest
import torch

from tiny_avatar_diffusion.text_conditioning import (
    ImageTextCrossAttention,
    TextConditioningEncoder,
)
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, TinyUNet


def _conditioning(batch_size: int = 1, length: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch_size, length, 64), torch.ones(batch_size, length, dtype=torch.long)


def test_unconditional_tiny_unet_api_behavior_and_parameter_count_are_unchanged() -> None:
    model = TinyUNet()

    assert tuple(inspect.signature(TinyUNet.forward).parameters) == (
        "self",
        "x_t",
        "timesteps",
    )
    output = model(torch.randn(1, 3, 32, 32), torch.tensor([500], dtype=torch.long))

    assert output.shape == (1, 3, 32, 32)
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_595_779


def test_conditional_unet_forward_shape_and_finite_values() -> None:
    model = ConditionalTinyUNet()
    context, context_mask = _conditioning(batch_size=2, length=4)

    output = model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([0, 999], dtype=torch.long),
        context,
        context_mask,
    )

    assert output.shape == (2, 3, 32, 32)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(("context", "context_mask", "message"), [(None, torch.ones(1, 3), "context is required"), (torch.randn(1, 3, 64), None, "context_mask is required")])
def test_conditional_unet_requires_context_and_mask(
    context: torch.Tensor | None, context_mask: torch.Tensor | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ConditionalTinyUNet()(
            torch.randn(1, 3, 32, 32), torch.tensor([1]), context, context_mask
        )


def test_exactly_one_cross_attention_is_executed_between_bottleneck_blocks() -> None:
    model = ConditionalTinyUNet()
    assert sum(isinstance(module, ImageTextCrossAttention) for module in model.modules()) == 1
    call_order: list[str] = []
    handles = [
        model.bottleneck_block1.register_forward_hook(lambda *_: call_order.append("block1")),
        model.image_text_cross_attention.register_forward_hook(lambda *_: call_order.append("cross_attention")),
        model.bottleneck_block2.register_forward_hook(lambda *_: call_order.append("block2")),
    ]
    try:
        context, context_mask = _conditioning()
        model(torch.randn(1, 3, 32, 32), torch.tensor([1]), context, context_mask)
    finally:
        for handle in handles:
            handle.remove()

    assert call_order == ["block1", "cross_attention", "block2"]


def test_zeroed_cross_attention_output_projection_recovers_unconditional_baseline() -> None:
    baseline = TinyUNet().eval()
    conditional = ConditionalTinyUNet().eval()
    conditional.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        conditional.image_text_cross_attention.output_projection.weight.zero_()
        conditional.image_text_cross_attention.output_projection.bias.zero_()
    x_t = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([500], dtype=torch.long)
    context, context_mask = _conditioning()

    with torch.no_grad():
        baseline_output = baseline(x_t, timesteps)
        conditional_output = conditional(x_t, timesteps, context, context_mask)

    assert torch.allclose(baseline_output, conditional_output, atol=1e-6, rtol=1e-5)


def test_conditional_unet_is_context_sensitive_and_padding_invariant() -> None:
    model = ConditionalTinyUNet().eval()
    x_t = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([500], dtype=torch.long)
    first_context = torch.randn(1, 3, 64)
    second_context = torch.randn(1, 3, 64)
    padded_context = torch.cat((first_context, torch.randn(1, 4, 64)), dim=1)

    with torch.no_grad():
        first = model(x_t, timesteps, first_context, torch.ones(1, 3, dtype=torch.long))
        repeated = model(x_t, timesteps, first_context, torch.ones(1, 3, dtype=torch.long))
        second = model(x_t, timesteps, second_context, torch.ones(1, 3, dtype=torch.long))
        padded = model(
            x_t,
            timesteps,
            padded_context,
            torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.long),
        )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)
    assert torch.allclose(first, padded, atol=1e-5, rtol=1e-4)


def test_end_to_end_gradients_reach_text_cross_attention_and_unet() -> None:
    text_encoder = TextConditioningEncoder()
    denoiser = ConditionalTinyUNet()
    input_ids = torch.tensor([[1, 4, 21, 29, 44, 12, 2]], dtype=torch.long)
    attention_mask = torch.ones(1, 7, dtype=torch.long)
    context = text_encoder(input_ids, attention_mask)
    prediction = denoiser(
        torch.randn(1, 3, 32, 32),
        torch.tensor([500], dtype=torch.long),
        context,
        attention_mask,
    )
    prediction.square().mean().backward()

    for parameter in (
        text_encoder.token_embedding.weight,
        text_encoder.blocks[0].self_attention.in_proj_weight,
        text_encoder.blocks[0].ffn[0].weight,
        denoiser.image_text_cross_attention.query_projection.weight,
        denoiser.image_text_cross_attention.key_projection.weight,
        denoiser.image_text_cross_attention.value_projection.weight,
        denoiser.image_text_cross_attention.output_projection.weight,
        denoiser.input_projection.weight,
        denoiser.bottleneck_block1.conv1.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_timestep_conditioning_remains_active() -> None:
    model = ConditionalTinyUNet().eval()
    x_t = torch.randn(1, 3, 32, 32)
    context, context_mask = _conditioning()
    with torch.no_grad():
        at_zero = model(x_t, torch.tensor([0]), context, context_mask)
        at_final = model(x_t, torch.tensor([999]), context, context_mask)

    assert not torch.equal(at_zero, at_final)


def test_conditional_and_complete_system_parameter_counts() -> None:
    baseline = TinyUNet()
    conditional = ConditionalTinyUNet()
    text_encoder = TextConditioningEncoder()
    cross_attention = conditional.image_text_cross_attention

    assert sum(parameter.numel() for parameter in baseline.parameters()) == 4_595_779
    assert sum(parameter.numel() for parameter in cross_attention.parameters()) == 25_152
    assert sum(parameter.numel() for parameter in conditional.parameters()) == 4_620_931
    assert sum(parameter.numel() for parameter in text_encoder.parameters()) == 102_976
    assert sum(parameter.numel() for parameter in conditional.parameters()) + sum(
        parameter.numel() for parameter in text_encoder.parameters()
    ) == 4_723_907


@pytest.mark.parametrize(
    ("context", "context_mask", "message"),
    [
        (torch.randn(1, 3, 32), torch.ones(1, 3), "final dimension"),
        (torch.randn(2, 3, 64), torch.ones(2, 3), "batch size"),
        (torch.randn(1, 3, 64), torch.ones(1, 2), "context_mask"),
        (torch.randn(1, 8, 64), torch.ones(1, 8), "length"),
        (torch.randn(1, 3, 64), torch.zeros(1, 3), "at least one valid"),
    ],
)
def test_conditional_unet_rejects_invalid_conditioning(
    context: torch.Tensor, context_mask: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ConditionalTinyUNet()(
            torch.randn(1, 3, 32, 32), torch.tensor([500]), context, context_mask
        )
