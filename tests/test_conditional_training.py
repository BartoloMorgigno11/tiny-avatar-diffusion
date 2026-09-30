import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.conditional_debug_tiny_overfit import (
    TEXT_TRAIN_MANIFEST,
    load_train_text_records,
    select_text_conditioning,
)
from tiny_avatar_diffusion.conditional_training import (
    conditional_diffusion_loss,
    conditional_training_step,
)
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, TinyUNet


def _components() -> tuple[TextConditioningEncoder, ConditionalTinyUNet, LinearDDPMSchedule]:
    return TextConditioningEncoder(), ConditionalTinyUNet(), LinearDDPMSchedule()


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 3, 32, 32),
        torch.tensor([[1, 4, 21, 29, 44, 12, 2]], dtype=torch.long),
        torch.ones(1, 7, dtype=torch.long),
        torch.tensor([500], dtype=torch.long),
        torch.randn(1, 3, 32, 32),
    )


def test_conditional_loss_shape_finiteness_and_exact_explicit_path() -> None:
    text_encoder, denoiser, schedule = _components()
    x_start, input_ids, mask, timesteps, noise = _batch()

    result = conditional_diffusion_loss(
        text_encoder, denoiser, schedule, x_start, input_ids, mask, timesteps=timesteps, noise=noise
    )

    assert result.predicted_noise.shape == result.noise.shape == x_start.shape
    assert result.noisy_images.shape == x_start.shape
    assert result.timesteps is timesteps
    assert result.noise is noise
    assert torch.equal(result.noisy_images, schedule.q_sample(x_start, timesteps, noise=noise))
    assert result.loss.ndim == 0 and torch.isfinite(result.loss)
    assert torch.allclose(result.loss, F.mse_loss(result.predicted_noise, noise))


def test_conditional_loss_has_end_to_end_gradients_and_pad_embedding_safety() -> None:
    text_encoder, denoiser, schedule = _components()
    x_start, input_ids, mask, timesteps, noise = _batch()
    result = conditional_diffusion_loss(
        text_encoder, denoiser, schedule, x_start, input_ids, mask, timesteps=timesteps, noise=noise
    )
    result.loss.backward()

    assert text_encoder.token_embedding.padding_idx == 0
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
        denoiser.output_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert torch.count_nonzero(text_encoder.token_embedding.weight.grad[[1, 4, 21, 29, 44, 12, 2]]).item() > 0
    assert torch.equal(text_encoder.token_embedding.weight.grad[0], torch.zeros(64))


def test_one_debug_optimizer_step_updates_text_and_conditional_denoiser() -> None:
    text_encoder, denoiser, schedule = _components()
    optimizer = torch.optim.AdamW(
        [*text_encoder.parameters(), *denoiser.parameters()], lr=1e-4, weight_decay=0.0
    )
    x_start, input_ids, mask, timesteps, noise = _batch()
    embedding_before = text_encoder.token_embedding.weight.detach().clone()
    cross_attention_before = denoiser.image_text_cross_attention.output_projection.weight.detach().clone()
    denoiser_before = denoiser.input_projection.weight.detach().clone()

    result = conditional_training_step(
        text_encoder, denoiser, schedule, optimizer, x_start, input_ids, mask, timesteps=timesteps, noise=noise
    )

    assert torch.isfinite(result.loss)
    assert not torch.equal(text_encoder.token_embedding.weight, embedding_before)
    assert not torch.equal(denoiser.image_text_cross_attention.output_projection.weight, cross_attention_before)
    assert not torch.equal(denoiser.input_projection.weight, denoiser_before)


def test_explicit_randomness_is_deterministic_in_eval_mode() -> None:
    text_encoder, denoiser, schedule = _components()
    text_encoder.eval()
    denoiser.eval()
    x_start, input_ids, mask, timesteps, noise = _batch()
    with torch.no_grad():
        first = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, input_ids, mask, timesteps=timesteps, noise=noise
        )
        second = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, input_ids, mask, timesteps=timesteps, noise=noise
        )

    assert torch.equal(first.predicted_noise, second.predicted_noise)
    assert torch.equal(first.loss, second.loss)


def test_different_valid_conditioning_can_change_conditional_prediction() -> None:
    text_encoder, denoiser, schedule = _components()
    text_encoder.eval()
    denoiser.eval()
    x_start, input_ids, mask, timesteps, noise = _batch()
    alternate_ids = torch.tensor([[1, 4, 18, 23, 42, 15, 2]], dtype=torch.long)
    with torch.no_grad():
        first = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, input_ids, mask, timesteps=timesteps, noise=noise
        )
        second = conditional_diffusion_loss(
            text_encoder, denoiser, schedule, x_start, alternate_ids, mask, timesteps=timesteps, noise=noise
        )

    assert not torch.equal(first.predicted_noise, second.predicted_noise)


def test_real_dr01_train_text_artifact_feeds_conditional_loss() -> None:
    records = load_train_text_records()
    record = records[0]
    input_ids, mask = select_text_conditioning([record])
    text_encoder, denoiser, schedule = _components()
    result = conditional_diffusion_loss(
        text_encoder,
        denoiser,
        schedule,
        torch.randn(1, 3, 32, 32),
        input_ids,
        mask,
        timesteps=torch.tensor([500]),
        noise=torch.randn(1, 3, 32, 32),
    )

    assert record["split_role"] == "TRAIN"
    assert input_ids.shape == mask.shape == (1, 7)
    assert 3 not in input_ids.tolist()[0]
    assert mask.tolist()[0] == [1] * 7
    assert torch.isfinite(result.loss)


def test_unconditional_dr02_epsilon_objective_path_remains_independent() -> None:
    model = TinyUNet()
    schedule = LinearDDPMSchedule()
    x_start = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([500])
    noise = torch.randn_like(x_start)
    predicted = model(schedule.q_sample(x_start, timesteps, noise=noise), timesteps)

    assert torch.isfinite(F.mse_loss(predicted, noise))
