import pytest
import torch

from tiny_avatar_diffusion.text_conditioning import (
    ImageTextCrossAttention,
    TextConditioningConfig,
    TextConditioningEncoder,
    sinusoidal_positional_encoding,
)


def test_frozen_configuration_defaults_and_invalid_values() -> None:
    config = TextConditioningConfig()

    assert (config.vocab_size, config.pad_id, config.max_length) == (45, 0, 7)
    assert (config.d_model, config.num_layers, config.num_heads, config.ff_dim) == (
        64,
        2,
        4,
        256,
    )
    assert config.dropout == 0.1

    with pytest.raises(ValueError, match="divisible"):
        TextConditioningConfig(d_model=66)
    with pytest.raises(ValueError, match="num_heads"):
        TextConditioningConfig(num_heads=0)
    with pytest.raises(ValueError, match="pad_id"):
        TextConditioningConfig(pad_id=45)


@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        (
            torch.tensor([[1, 4, 21, 29, 44, 12, 2]], dtype=torch.long),
            torch.ones(1, 7, dtype=torch.long),
        ),
        (
            torch.tensor([[1, 4, 21, 2, 0, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.long),
        ),
    ],
)
def test_shape_contract_for_full_and_padded_sequences(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> None:
    output = TextConditioningEncoder()(input_ids, attention_mask)

    assert output.shape == (1, 7, 64)


def test_input_ids_must_be_torch_long() -> None:
    encoder = TextConditioningEncoder()
    input_ids = torch.tensor([[1, 4, 2]], dtype=torch.int32)
    mask = torch.ones(1, 3, dtype=torch.long)

    with pytest.raises(TypeError, match="torch.long"):
        encoder(input_ids, mask)


def test_sinusoidal_positions_are_deterministic_non_trainable_and_parameter_free() -> None:
    first = sinusoidal_positional_encoding(7, 64)
    second = sinusoidal_positional_encoding(7, 64)
    encoder = TextConditioningEncoder()

    assert first.shape == (7, 64)
    assert torch.equal(first, second)
    assert not encoder.positional_encoding.requires_grad
    assert "positional_encoding" not in dict(encoder.named_parameters())
    assert not any("position" in name for name, _ in encoder.named_parameters())


def test_self_attention_receives_no_causal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = TextConditioningEncoder()
    received: dict[str, object] = {}
    original_forward = encoder.blocks[0].self_attention.forward

    def tracked_forward(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        received["attn_mask"] = kwargs.get("attn_mask")
        received["is_causal"] = kwargs.get("is_causal", False)
        return original_forward(*args, **kwargs)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(encoder.blocks[0].self_attention, "forward", tracked_forward)
    encoder(torch.tensor([[1, 4, 2]], dtype=torch.long), torch.ones(1, 3))

    assert received == {"attn_mask": None, "is_causal": False}


def test_valid_tokens_are_invariant_to_masked_right_padding_in_eval_mode() -> None:
    encoder = TextConditioningEncoder().eval()
    unpadded_ids = torch.tensor([[1, 4, 21, 2]], dtype=torch.long)
    unpadded_mask = torch.ones(1, 4, dtype=torch.long)
    padded_ids = torch.tensor([[1, 4, 21, 2, 0, 0, 0]], dtype=torch.long)
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        unpadded = encoder(unpadded_ids, unpadded_mask)
        padded = encoder(padded_ids, padded_mask)

    assert torch.allclose(unpadded, padded[:, :4], atol=1e-6, rtol=1e-5)


def test_eval_mode_is_deterministic_and_token_sensitive() -> None:
    encoder = TextConditioningEncoder().eval()
    mask = torch.ones(1, 7, dtype=torch.long)
    first_ids = torch.tensor([[1, 4, 21, 29, 44, 12, 2]], dtype=torch.long)
    second_ids = torch.tensor([[1, 4, 18, 23, 42, 15, 2]], dtype=torch.long)

    with torch.no_grad():
        first = encoder(first_ids, mask)
        repeated = encoder(first_ids, mask)
        second = encoder(second_ids, mask)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)


def test_gradients_reach_used_embeddings_attention_and_ffn() -> None:
    encoder = TextConditioningEncoder()
    input_ids = torch.tensor([[1, 4, 21, 2]], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)

    encoder(input_ids, mask).square().mean().backward()

    embedding_grad = encoder.token_embedding.weight.grad
    assert embedding_grad is not None
    assert torch.isfinite(embedding_grad[[1, 4, 21, 2]]).all()
    assert torch.count_nonzero(embedding_grad[[1, 4, 21, 2]]).item() > 0
    for parameter in (
        encoder.blocks[0].self_attention.in_proj_weight,
        encoder.blocks[0].ffn[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_pad_embedding_uses_dr01_padding_index_and_receives_no_gradient() -> None:
    encoder = TextConditioningEncoder()
    assert encoder.token_embedding.padding_idx == 0
    assert torch.equal(encoder.token_embedding.weight[0], torch.zeros(64))

    encoder(
        torch.tensor([[1, 4, 2, 0]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    ).square().mean().backward()

    assert torch.equal(encoder.token_embedding.weight.grad[0], torch.zeros(64))


def test_trainable_parameter_count() -> None:
    encoder = TextConditioningEncoder()

    assert sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad) == 102_976


def test_cross_attention_construction_and_frozen_dimensions() -> None:
    block = ImageTextCrossAttention()

    assert (
        block.image_channels,
        block.text_dim,
        block.attention_dim,
        block.num_heads,
        block.head_dim,
        block.max_context_length,
    ) == (128, 64, 64, 4, 16, 7)
    with pytest.raises(ValueError, match="divisible"):
        ImageTextCrossAttention(attention_dim=66, num_heads=4)


@pytest.mark.parametrize(("batch_size", "length"), [(1, 1), (2, 7), (3, 4)])
def test_cross_attention_shape_contract(batch_size: int, length: int) -> None:
    block = ImageTextCrossAttention()
    image = torch.randn(batch_size, 128, 8, 8)
    context = torch.randn(batch_size, length, 64)
    mask = torch.ones(batch_size, length, dtype=torch.long)

    output = block(image, context, mask)

    assert output.shape == (batch_size, 128, 8, 8)


def test_cross_attention_parameter_count() -> None:
    block = ImageTextCrossAttention()

    assert sum(parameter.numel() for parameter in block.parameters() if parameter.requires_grad) == 25_152


def test_cross_attention_uses_a_genuine_visual_residual_path() -> None:
    block = ImageTextCrossAttention()
    image = torch.randn(1, 128, 8, 8)
    context = torch.randn(1, 3, 64)
    mask = torch.ones(1, 3, dtype=torch.long)
    with torch.no_grad():
        for projection in (
            block.query_projection,
            block.key_projection,
            block.value_projection,
            block.output_projection,
        ):
            projection.weight.zero_()
            projection.bias.zero_()

    assert torch.equal(block(image, context, mask), image)


def test_cross_attention_is_invariant_to_arbitrary_masked_pad_context() -> None:
    block = ImageTextCrossAttention()
    image = torch.randn(1, 128, 8, 8)
    valid_context = torch.randn(1, 3, 64)
    padded_context = torch.cat((valid_context, torch.randn(1, 4, 64)), dim=1)

    unpadded = block(image, valid_context, torch.ones(1, 3, dtype=torch.long))
    padded = block(
        image,
        padded_context,
        torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.long),
    )

    assert torch.allclose(unpadded, padded, atol=1e-6, rtol=1e-5)


def test_cross_attention_masked_positions_receive_zero_probability() -> None:
    block = ImageTextCrossAttention()
    _, weights = block(
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 4, 64),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        return_attention=True,
    )

    assert weights.shape == (1, 4, 64, 4)
    assert torch.allclose(weights[..., 2:], torch.zeros_like(weights[..., 2:]))


def test_cross_attention_rejects_all_masked_context() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        ImageTextCrossAttention()(
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 3, 64),
            torch.zeros(1, 3, dtype=torch.long),
        )


def test_cross_attention_has_gradient_flow_and_masked_context_has_no_effect() -> None:
    block = ImageTextCrossAttention()
    image = torch.randn(1, 128, 8, 8, requires_grad=True)
    context = torch.randn(1, 4, 64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    block(image, context, mask).square().mean().backward()

    assert image.grad is not None and torch.count_nonzero(image.grad).item() > 0
    assert context.grad is not None
    assert torch.count_nonzero(context.grad[:, :3]).item() > 0
    assert torch.equal(context.grad[:, 3], torch.zeros_like(context.grad[:, 3]))
    for parameter in (
        block.query_projection.weight,
        block.key_projection.weight,
        block.value_projection.weight,
        block.output_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_cross_attention_is_text_sensitive_and_deterministic() -> None:
    block = ImageTextCrossAttention()
    image = torch.randn(1, 128, 8, 8)
    first_context = torch.randn(1, 3, 64)
    second_context = torch.randn(1, 3, 64)
    mask = torch.ones(1, 3, dtype=torch.long)

    first = block(image, first_context, mask)
    repeated = block(image, first_context, mask)
    second = block(image, second_context, mask)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    ("image", "context", "mask", "message"),
    [
        (torch.randn(1, 64, 8, 8), torch.randn(1, 3, 64), torch.ones(1, 3), "shape"),
        (torch.randn(1, 128, 4, 4), torch.randn(1, 3, 64), torch.ones(1, 3), "shape"),
        (torch.randn(1, 128, 8, 8), torch.randn(1, 3, 32), torch.ones(1, 3), "final dimension"),
        (torch.randn(1, 128, 8, 8), torch.randn(2, 3, 64), torch.ones(2, 3), "batch size"),
        (torch.randn(1, 128, 8, 8), torch.randn(1, 3, 64), torch.ones(1, 2), "context_mask"),
        (torch.randn(1, 128, 8, 8), torch.randn(1, 8, 64), torch.ones(1, 8), "length"),
    ],
)
def test_cross_attention_rejects_invalid_shapes(
    image: torch.Tensor, context: torch.Tensor, mask: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageTextCrossAttention()(image, context, mask)
