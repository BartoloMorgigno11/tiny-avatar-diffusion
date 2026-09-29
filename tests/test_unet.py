import pytest
import torch
from torch import nn

from tiny_avatar_diffusion.unet import (
    Downsample,
    ResBlock,
    TimeEmbedding,
    TinyUNet,
    Upsample,
    sinusoidal_timestep_embedding,
)


@pytest.mark.parametrize("batch_size", (1, 4))
def test_sinusoidal_timestep_embedding_shape(batch_size: int) -> None:
    timesteps = torch.arange(batch_size, dtype=torch.long)

    embedding = sinusoidal_timestep_embedding(timesteps, embedding_dim=64)

    assert embedding.shape == (batch_size, 64)


def test_sinusoidal_timestep_embedding_matches_independent_formula() -> None:
    timesteps = torch.tensor([0, 1, 500, 999], dtype=torch.long)
    half = 32
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0))
        * (torch.arange(half, dtype=torch.float32) / half)
    )
    arguments = timesteps[:, None].to(torch.float32) * frequencies[None]
    expected = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=-1)

    embedding = sinusoidal_timestep_embedding(timesteps, embedding_dim=64)

    assert torch.allclose(embedding, expected)


def test_sinusoidal_timestep_embedding_has_expected_zero_timestep_structure() -> None:
    embedding = sinusoidal_timestep_embedding(
        torch.tensor([0], dtype=torch.long), embedding_dim=64
    )

    assert torch.equal(embedding[:, :32], torch.zeros(1, 32))
    assert torch.equal(embedding[:, 32:], torch.ones(1, 32))


def test_sinusoidal_timestep_embedding_is_deterministic_and_distinguishes_timesteps() -> None:
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)

    first = sinusoidal_timestep_embedding(timesteps, embedding_dim=64)
    second = sinusoidal_timestep_embedding(timesteps, embedding_dim=64)

    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first[1], first[2])
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("batch_size", (1, 4))
def test_time_embedding_shape(batch_size: int) -> None:
    module = TimeEmbedding()
    timesteps = torch.arange(batch_size, dtype=torch.long)

    output = module(timesteps)

    assert output.shape == (batch_size, 256)
    assert torch.isfinite(output).all()


def test_time_embedding_has_the_frozen_architecture_and_trainable_parameters() -> None:
    module = TimeEmbedding()
    layers = list(module.mlp)

    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 64
    assert layers[0].out_features == 256
    assert isinstance(layers[1], nn.SiLU)
    assert isinstance(layers[2], nn.Linear)
    assert layers[2].in_features == 256
    assert layers[2].out_features == 256
    assert sum(parameter.numel() for parameter in module.parameters()) == 82_432
    assert set(dict(module.named_parameters())) == {
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
    }
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_time_embedding_linear_layers_receive_finite_gradients() -> None:
    module = TimeEmbedding()
    output = module(torch.tensor([0, 500, 999], dtype=torch.long))

    output.square().mean().backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sinusoidal_timestep_embedding_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sinusoidal_timestep_embedding(torch.zeros(2, 1, dtype=torch.long), 64)
    with pytest.raises(ValueError, match="positive even"):
        sinusoidal_timestep_embedding(torch.zeros(2, dtype=torch.long), 63)


def test_res_block_uses_an_identity_shortcut_and_preserves_spatial_resolution() -> None:
    block = ResBlock(64, 64)
    x = torch.randn(2, 64, 7, 9)
    time_emb = torch.randn(2, 256)

    output = block(x, time_emb)

    assert isinstance(block.shortcut, nn.Identity)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_res_block_uses_a_projection_shortcut_for_channel_change() -> None:
    block = ResBlock(64, 128)
    x = torch.randn(2, 64, 7, 9)

    output = block(x, torch.randn(2, 256))

    assert isinstance(block.shortcut, nn.Conv2d)
    assert block.shortcut.kernel_size == (1, 1)
    assert block.shortcut.stride == (1, 1)
    assert output.shape == (2, 128, 7, 9)


@pytest.mark.parametrize(
    ("in_channels", "out_channels"),
    ((256, 128), (192, 64)),
)
def test_res_block_supports_frozen_decoder_channel_transitions(
    in_channels: int, out_channels: int
) -> None:
    block = ResBlock(in_channels, out_channels)
    x = torch.randn(2, in_channels, 5, 6)

    output = block(x, torch.randn(2, 256))

    assert output.shape == (2, out_channels, 5, 6)
    assert torch.isfinite(output).all()


def test_res_block_has_frozen_time_projection_and_group_norm_contract() -> None:
    block = ResBlock(64, 128, dropout=0.25)

    assert isinstance(block.time_projection, nn.Linear)
    assert block.time_projection.in_features == 256
    assert block.time_projection.out_features == 128
    assert block.norm1.num_groups == 32
    assert block.norm1.num_channels == 64
    assert block.norm2.num_groups == 32
    assert block.norm2.num_channels == 128
    assert block.dropout.p == 0.25


def test_res_block_applies_time_after_the_first_convolution() -> None:
    block = ResBlock(64, 64)
    block.eval()
    x = torch.randn(1, 64, 4, 5).expand(2, -1, -1, -1).clone()
    time_emb = torch.zeros(2, 256)
    time_emb[1, 0] = 1.0

    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()
        channel_indices = torch.arange(64)
        block.conv2.weight[channel_indices, channel_indices, 1, 1] = 1.0
        block.time_projection.weight.zero_()
        block.time_projection.bias.zero_()
        block.time_projection.weight[:, 0] = torch.linspace(-2.0, 2.0, 64)

    output = block(x, time_emb)
    first_convolution = block.conv1(block.activation1(block.norm1(x)))
    projected_time = block.time_projection(time_emb)[:, :, None, None]
    expected = block.conv2(
        block.dropout(block.activation2(block.norm2(first_convolution + projected_time)))
    ) + block.shortcut(x)

    assert torch.allclose(output, expected)
    assert not torch.equal(output[0], output[1])


def test_res_block_with_zero_dropout_is_deterministic_in_train_mode() -> None:
    block = ResBlock(64, 64, dropout=0.0)
    block.train()
    x = torch.randn(2, 64, 4, 5)
    time_emb = torch.randn(2, 256)

    first = block(x, time_emb)
    second = block(x, time_emb)

    assert torch.equal(first, second)


def test_res_block_preserves_finite_gradient_paths() -> None:
    block = ResBlock(64, 128)
    x = torch.randn(2, 64, 4, 5, requires_grad=True)
    time_emb = torch.randn(2, 256, requires_grad=True)

    output = block(x, time_emb)
    output.square().mean().backward()

    assert x.grad is not None
    assert time_emb.grad is not None
    assert block.conv1.weight.grad is not None
    assert block.conv2.weight.grad is not None
    assert block.time_projection.weight.grad is not None
    for gradient in (
        x.grad,
        time_emb.grad,
        block.conv1.weight.grad,
        block.conv2.weight.grad,
        block.time_projection.weight.grad,
    ):
        assert torch.isfinite(gradient).all()


def test_res_block_rejects_invalid_inputs_and_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ResBlock(64, 64, num_groups=0)
    with pytest.raises(ValueError, match="divisible"):
        ResBlock(48, 64)
    with pytest.raises(ValueError, match="dropout"):
        ResBlock(64, 64, dropout=1.0)

    block = ResBlock(64, 64)
    x = torch.randn(2, 64, 4, 5)
    time_emb = torch.randn(2, 256)
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        block(torch.randn(2, 64, 4), time_emb)
    with pytest.raises(ValueError, match="channel count"):
        block(torch.randn(2, 32, 4, 5), time_emb)
    with pytest.raises(ValueError, match=r"\[B, time_embedding_dim\]"):
        block(x, torch.randn(2, 1, 256))
    with pytest.raises(ValueError, match="batch size"):
        block(x, torch.randn(1, 256))
    with pytest.raises(ValueError, match="embedding dimension"):
        block(x, torch.randn(2, 128))


@pytest.mark.parametrize(
    ("batch_size", "channels", "spatial_size", "expected_spatial_size"),
    ((1, 64, 32, 16), (2, 128, 16, 8)),
)
def test_downsample_preserves_channels_and_halves_spatial_resolution(
    batch_size: int,
    channels: int,
    spatial_size: int,
    expected_spatial_size: int,
) -> None:
    module = Downsample(channels)
    x = torch.randn(batch_size, channels, spatial_size, spatial_size)

    output = module(x)

    assert output.shape == (
        batch_size,
        channels,
        expected_spatial_size,
        expected_spatial_size,
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("batch_size", "spatial_size"),
    ((1, 8), (2, 8), (2, 16)),
)
def test_upsample_preserves_channels_and_doubles_spatial_resolution(
    batch_size: int, spatial_size: int
) -> None:
    module = Upsample(128)
    x = torch.randn(batch_size, 128, spatial_size, spatial_size)

    output = module(x)

    assert output.shape == (
        batch_size,
        128,
        spatial_size * 2,
        spatial_size * 2,
    )
    assert torch.isfinite(output).all()


def test_downsample_has_the_frozen_convolution_contract() -> None:
    module = Downsample(64)

    assert set(dict(module.named_parameters())) == {"conv.weight", "conv.bias"}
    assert module.conv.in_channels == 64
    assert module.conv.out_channels == 64
    assert module.conv.kernel_size == (3, 3)
    assert module.conv.stride == (2, 2)
    assert module.conv.padding == (1, 1)


def test_upsample_uses_nearest_resize_before_the_frozen_convolution() -> None:
    module = Upsample(1)
    with torch.no_grad():
        module.conv.weight.zero_()
        module.conv.bias.zero_()
        module.conv.weight[0, 0, 0, 0] = 1.0
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    output = module(x)
    expected = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 2.0],
           [0.0, 1.0, 1.0, 2.0], [0.0, 3.0, 3.0, 4.0]]]]
    )

    assert output.shape == (1, 1, 4, 4)
    assert torch.equal(output, expected)
    assert set(dict(module.named_parameters())) == {"conv.weight", "conv.bias"}
    assert module.conv.in_channels == module.conv.out_channels == 1
    assert module.conv.kernel_size == (3, 3)
    assert module.conv.stride == (1, 1)
    assert module.conv.padding == (1, 1)
    assert not any(isinstance(layer, nn.ConvTranspose2d) for layer in module.modules())


@pytest.mark.parametrize(
    "module",
    (Downsample(64), Upsample(64)),
)
def test_sampling_modules_preserve_finite_gradient_paths(module: nn.Module) -> None:
    x = torch.randn(2, 64, 8, 8, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    convolution = module.conv  # type: ignore[attr-defined]
    assert convolution.weight.grad is not None
    assert torch.isfinite(convolution.weight.grad).all()


@pytest.mark.parametrize("module", (Downsample(64), Upsample(64)))
def test_sampling_modules_reject_invalid_inputs(module: nn.Module) -> None:
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        module(torch.randn(2, 64, 8))
    with pytest.raises(ValueError, match="channel count"):
        module(torch.randn(2, 32, 8, 8))


@pytest.mark.parametrize("batch_size", (1, 2))
def test_tiny_unet_predicts_noise_with_input_shape(batch_size: int) -> None:
    model = TinyUNet()
    x_t = torch.randn(batch_size, 3, 32, 32)
    timesteps = torch.arange(batch_size, dtype=torch.long)

    epsilon_hat = model(x_t, timesteps)

    assert epsilon_hat.shape == x_t.shape
    assert torch.isfinite(epsilon_hat).all()


def test_tiny_unet_matches_the_frozen_internal_shape_contract() -> None:
    model = TinyUNet()
    observed_outputs: dict[str, tuple[int, ...]] = {}
    observed_inputs: dict[str, tuple[int, ...]] = {}

    def record_output(name: str):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed_outputs[name] = tuple(output.shape)

        return hook

    def record_input(name: str):
        def hook(_: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            observed_inputs[name] = tuple(inputs[0].shape)

        return hook

    hooks = [
        model.input_projection.register_forward_hook(record_output("input")),
        model.encoder32_block2.register_forward_hook(record_output("s32")),
        model.downsample32.register_forward_hook(record_output("down32")),
        model.encoder16_block2.register_forward_hook(record_output("s16")),
        model.downsample16.register_forward_hook(record_output("down16")),
        model.encoder8_block2.register_forward_hook(record_output("s8")),
        model.bottleneck_block2.register_forward_hook(record_output("bottleneck")),
        model.decoder8_block1.register_forward_pre_hook(record_input("concat8")),
        model.decoder8_block2.register_forward_hook(record_output("decoder8")),
        model.upsample8.register_forward_hook(record_output("up8")),
        model.decoder16_block1.register_forward_pre_hook(record_input("concat16")),
        model.decoder16_block2.register_forward_hook(record_output("decoder16")),
        model.upsample16.register_forward_hook(record_output("up16")),
        model.decoder32_block1.register_forward_pre_hook(record_input("concat32")),
        model.decoder32_block2.register_forward_hook(record_output("decoder32")),
        model.output_projection.register_forward_hook(record_output("final")),
    ]
    try:
        output = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 999]))
    finally:
        for hook in hooks:
            hook.remove()

    assert observed_outputs == {
        "input": (2, 64, 32, 32),
        "s32": (2, 64, 32, 32),
        "down32": (2, 64, 16, 16),
        "s16": (2, 128, 16, 16),
        "down16": (2, 128, 8, 8),
        "s8": (2, 128, 8, 8),
        "bottleneck": (2, 128, 8, 8),
        "decoder8": (2, 128, 8, 8),
        "up8": (2, 128, 16, 16),
        "decoder16": (2, 128, 16, 16),
        "up16": (2, 128, 32, 32),
        "decoder32": (2, 64, 32, 32),
        "final": (2, 3, 32, 32),
    }
    assert observed_inputs == {
        "concat8": (2, 256, 8, 8),
        "concat16": (2, 256, 16, 16),
        "concat32": (2, 192, 32, 32),
    }
    assert output.shape == (2, 3, 32, 32)


def test_tiny_unet_time_embedding_changes_the_prediction() -> None:
    model = TinyUNet()
    model.eval()
    x_t = torch.randn(1, 3, 32, 32)

    first = model(x_t, torch.tensor([0]))
    second = model(x_t, torch.tensor([999]))

    assert not torch.equal(first, second)


def test_tiny_unet_has_the_frozen_output_head_without_final_activation() -> None:
    model = TinyUNet()

    assert isinstance(model.output_norm, nn.GroupNorm)
    assert model.output_norm.num_groups == 32
    assert model.output_norm.num_channels == 64
    assert isinstance(model.output_activation, nn.SiLU)
    assert isinstance(model.output_projection, nn.Conv2d)
    assert model.output_projection.in_channels == 64
    assert model.output_projection.out_channels == 3
    assert model.output_projection.kernel_size == (3, 3)
    assert model.output_projection.stride == (1, 1)
    assert model.output_projection.padding == (1, 1)


def test_tiny_unet_is_deterministic_with_zero_dropout_in_train_mode() -> None:
    model = TinyUNet(dropout=0.0)
    model.train()
    x_t = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([500])

    first = model(x_t, timesteps)
    second = model(x_t, timesteps)

    assert torch.equal(first, second)


def test_tiny_unet_preserves_finite_forward_and_backward_connectivity() -> None:
    model = TinyUNet()
    x_t = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([500])
    target_noise = torch.randn_like(x_t)

    epsilon_hat = model(x_t, timesteps)
    loss = torch.nn.functional.mse_loss(epsilon_hat, target_noise)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(epsilon_hat).all()
    for parameter in (
        model.time_embedding.mlp[0].weight,
        model.input_projection.weight,
        model.encoder32_block1.conv1.weight,
        model.bottleneck_block1.conv1.weight,
        model.decoder8_block1.conv1.weight,
        model.output_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_tiny_unet_long_skip_receives_gradient_from_the_decoder() -> None:
    model = TinyUNet()
    skip_gradient: list[torch.Tensor] = []

    def capture_concat_gradient(
        _: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        inputs[0].register_hook(lambda gradient: skip_gradient.append(gradient))

    hook = model.decoder32_block1.register_forward_pre_hook(capture_concat_gradient)
    try:
        output = model(torch.randn(1, 3, 32, 32), torch.tensor([500]))
        output.square().mean().backward()
    finally:
        hook.remove()

    assert len(skip_gradient) == 1
    assert torch.isfinite(skip_gradient[0]).all()
    assert torch.count_nonzero(skip_gradient[0][:, 128:]) > 0


def test_tiny_unet_produces_finite_values_at_representative_timesteps() -> None:
    model = TinyUNet()
    x_t = torch.randn(3, 3, 32, 32)

    output = model(x_t, torch.tensor([0, 500, 999]))

    assert torch.isfinite(output).all()


def test_tiny_unet_rejects_invalid_inputs() -> None:
    model = TinyUNet()
    x_t = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0, 1])

    with pytest.raises(ValueError, match=r"\[B, 3, H, W\]"):
        model(torch.randn(2, 3, 32), timesteps)
    with pytest.raises(ValueError, match="three channels"):
        model(torch.randn(2, 1, 32, 32), timesteps)
    with pytest.raises(ValueError, match="one-dimensional"):
        model(x_t, torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="batch size"):
        model(x_t, torch.tensor([0]))
    with pytest.raises(ValueError, match="divisible by four"):
        model(torch.randn(2, 3, 30, 32), timesteps)
