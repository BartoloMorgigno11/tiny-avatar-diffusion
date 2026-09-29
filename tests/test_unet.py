import pytest
import torch
from torch import nn

from tiny_avatar_diffusion.unet import TimeEmbedding, sinusoidal_timestep_embedding


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
