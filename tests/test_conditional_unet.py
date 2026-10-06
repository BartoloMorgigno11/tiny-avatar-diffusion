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


def _pre_r4_control_forward(
    model: ConditionalTinyUNet,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> torch.Tensor:
    """Independent transcription of the pre-R4 CA8-only forward path."""
    time_emb = model.time_embedding(timesteps)
    h = model.input_projection(x_t)

    h = model.encoder32_block1(h, time_emb)
    s32 = model.encoder32_block2(h, time_emb)
    h = model.downsample32(s32)

    h = model.encoder16_block1(h, time_emb)
    s16 = model.encoder16_block2(h, time_emb)
    h = model.downsample16(s16)

    h = model.encoder8_block1(h, time_emb)
    s8 = model.encoder8_block2(h, time_emb)

    h = model.bottleneck_block1(s8, time_emb)
    h = model.image_text_cross_attention(h, context, context_mask)
    h = model.bottleneck_block2(h, time_emb)

    h = torch.cat((h, s8), dim=1)
    h = model.decoder8_block1(h, time_emb)
    h = model.decoder8_block2(h, time_emb)
    h = model.upsample8(h)

    h = torch.cat((h, s16), dim=1)
    h = model.decoder16_block1(h, time_emb)
    h = model.decoder16_block2(h, time_emb)
    h = model.upsample16(h)

    h = torch.cat((h, s32), dim=1)
    h = model.decoder32_block1(h, time_emb)
    h = model.decoder32_block2(h, time_emb)
    return model.output_projection(model.output_activation(model.output_norm(h)))


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


def test_default_control_forward_exactly_matches_pre_r4_reference() -> None:
    torch.manual_seed(2026)
    model = ConditionalTinyUNet().eval()
    x_t = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0, 999], dtype=torch.long)
    context, context_mask = _conditioning(batch_size=2, length=7)

    with torch.no_grad():
        expected = _pre_r4_control_forward(model, x_t, timesteps, context, context_mask)
        actual = model(x_t, timesteps, context, context_mask)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_control_and_treatment_share_identical_initialization(seed: int) -> None:
    torch.manual_seed(seed)
    control = ConditionalTinyUNet(use_decoder16_cross_attention=False)
    torch.manual_seed(seed)
    treatment = ConditionalTinyUNet(use_decoder16_cross_attention=True)

    control_parameters = dict(control.named_parameters())
    treatment_parameters = dict(treatment.named_parameters())
    assert control_parameters.keys() <= treatment_parameters.keys()
    assert all(
        torch.equal(parameter, treatment_parameters[name])
        for name, parameter in control_parameters.items()
    )
    assert set(treatment_parameters) - set(control_parameters) == {
        f"decoder16_image_text_cross_attention.{name}"
        for name, _ in treatment.decoder16_image_text_cross_attention.named_parameters()
    }


def test_decoder16_flag_requires_an_actual_bool() -> None:
    for invalid in (1, 0, "true", None):
        with pytest.raises(TypeError, match="must be a bool"):
            ConditionalTinyUNet(use_decoder16_cross_attention=invalid)  # type: ignore[arg-type]


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


def test_treatment_attention_locations_call_counts_and_shared_context() -> None:
    model = ConditionalTinyUNet(use_decoder16_cross_attention=True)
    assert sum(isinstance(module, ImageTextCrossAttention) for module in model.modules()) == 2
    call_order: list[str] = []
    observed: dict[str, tuple[torch.Size, torch.Tensor, torch.Tensor]] = {}

    def attention_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            call_order.append(name)
            observed[name] = (inputs[0].shape, inputs[1], inputs[2])

        return hook

    handles = [
        model.bottleneck_block1.register_forward_hook(lambda *_: call_order.append("bottleneck_block1")),
        model.image_text_cross_attention.register_forward_hook(attention_hook("CA8")),
        model.bottleneck_block2.register_forward_hook(lambda *_: call_order.append("bottleneck_block2")),
        model.decoder16_block1.register_forward_hook(lambda *_: call_order.append("decoder16_block1")),
        model.decoder16_image_text_cross_attention.register_forward_hook(attention_hook("CA16")),
        model.decoder16_block2.register_forward_hook(lambda *_: call_order.append("decoder16_block2")),
    ]
    context, context_mask = _conditioning()
    try:
        model(torch.randn(1, 3, 32, 32), torch.tensor([500]), context, context_mask)
    finally:
        for handle in handles:
            handle.remove()

    assert call_order == [
        "bottleneck_block1", "CA8", "bottleneck_block2",
        "decoder16_block1", "CA16", "decoder16_block2",
    ]
    assert observed["CA8"][0] == (1, 128, 8, 8)
    assert observed["CA16"][0] == (1, 128, 16, 16)
    assert observed["CA8"][1] is context and observed["CA16"][1] is context
    assert observed["CA8"][2] is context_mask and observed["CA16"][2] is context_mask


@pytest.mark.parametrize("use_decoder16_cross_attention", [False, True])
def test_control_and_treatment_forward_backward_connectivity(
    use_decoder16_cross_attention: bool,
) -> None:
    torch.manual_seed(2026)
    model = ConditionalTinyUNet(
        use_decoder16_cross_attention=use_decoder16_cross_attention
    )
    context = torch.randn(1, 4, 64, requires_grad=True)
    context_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    output = model(
        torch.randn(1, 3, 32, 32),
        torch.tensor([500]),
        context,
        context_mask,
    )

    assert output.shape == (1, 3, 32, 32) and torch.isfinite(output).all()
    output.square().mean().backward()

    gradient_modules: list[torch.nn.Module] = [
        model.image_text_cross_attention,
        model.decoder16_block1,
        model.decoder16_block2,
    ]
    if use_decoder16_cross_attention:
        gradient_modules.append(model.decoder16_image_text_cross_attention)
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert torch.count_nonzero(context.grad).item() > 0
    for parameter in (
        parameter for module in gradient_modules for parameter in module.parameters()
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(
        model.image_text_cross_attention.query_projection.weight.grad
    ).item() > 0
    if use_decoder16_cross_attention:
        assert torch.count_nonzero(
            model.decoder16_image_text_cross_attention.query_projection.weight.grad
        ).item() > 0


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
    treatment = ConditionalTinyUNet(use_decoder16_cross_attention=True)
    text_encoder = TextConditioningEncoder()
    cross_attention = conditional.image_text_cross_attention
    ca16 = treatment.decoder16_image_text_cross_attention

    assert sum(parameter.numel() for parameter in baseline.parameters()) == 4_595_779
    assert sum(parameter.numel() for parameter in cross_attention.parameters()) == 25_152
    assert sum(parameter.numel() for parameter in conditional.parameters()) == 4_620_931
    assert sum(parameter.numel() for parameter in ca16.parameters()) == 25_152
    assert sum(parameter.numel() for parameter in treatment.parameters()) == 4_646_083
    assert sum(parameter.numel() for parameter in text_encoder.parameters()) == 102_976
    assert sum(parameter.numel() for parameter in conditional.parameters()) + sum(
        parameter.numel() for parameter in text_encoder.parameters()
    ) == 4_723_907
    assert sum(parameter.numel() for parameter in treatment.parameters()) + sum(
        parameter.numel() for parameter in text_encoder.parameters()
    ) == 4_749_059
    assert sum(parameter.numel() for parameter in treatment.parameters()) - sum(
        parameter.numel() for parameter in conditional.parameters()
    ) == 25_152


def test_control_state_dict_is_backward_compatible_and_treatment_adds_only_ca16() -> None:
    default_control = ConditionalTinyUNet()
    explicit_control = ConditionalTinyUNet(use_decoder16_cross_attention=False)
    treatment = ConditionalTinyUNet(use_decoder16_cross_attention=True)
    default_keys = set(default_control.state_dict())
    explicit_keys = set(explicit_control.state_dict())
    treatment_keys = set(treatment.state_dict())

    assert default_keys == explicit_keys
    assert not any(key.startswith("decoder16_image_text_cross_attention.") for key in default_keys)
    assert default_keys <= treatment_keys
    assert treatment_keys - default_keys
    assert all(
        key.startswith("decoder16_image_text_cross_attention.")
        for key in treatment_keys - default_keys
    )


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
