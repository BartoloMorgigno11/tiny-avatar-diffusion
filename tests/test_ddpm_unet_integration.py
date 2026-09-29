import torch
from torch.nn import functional as F

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.unet import TinyUNet


def test_ddpm_noise_prediction_objective_integrates_schedule_and_unet() -> None:
    schedule = LinearDDPMSchedule()
    model = TinyUNet(dropout=0.0)
    x_start = torch.empty(4, 3, 32, 32).uniform_(-1.0, 1.0)
    timesteps = torch.tensor([0, 250, 500, 999], dtype=torch.long)
    noise = torch.randn_like(x_start)
    output_weight_before = model.output_projection.weight.detach().clone()

    x_t = schedule.q_sample(x_start, timesteps, noise=noise)
    predicted_noise = model(x_t, timesteps)
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()

    assert torch.all((-1.0 <= x_start) & (x_start <= 1.0))
    assert x_start.shape == noise.shape == x_t.shape == predicted_noise.shape
    assert x_start.shape == (4, 3, 32, 32)
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(predicted_noise).all()
    assert torch.isfinite(loss)
    assert torch.equal(model.output_projection.weight, output_weight_before)
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
    assert not dict(schedule.named_parameters())
    for buffer in schedule.buffers():
        assert not buffer.requires_grad
        assert buffer.grad is None


def test_explicit_noise_training_path_is_deterministic_and_timestep_sensitive() -> None:
    schedule = LinearDDPMSchedule()
    model = TinyUNet(dropout=0.0)
    model.train()
    x_start = torch.full((1, 3, 32, 32), 0.25)
    noise = torch.linspace(-1.0, 1.0, x_start.numel()).reshape_as(x_start)
    timestep = torch.tensor([500], dtype=torch.long)

    first_x_t = schedule.q_sample(x_start, timestep, noise=noise)
    first_prediction = model(first_x_t, timestep)
    first_loss = F.mse_loss(first_prediction, noise)
    second_x_t = schedule.q_sample(x_start, timestep, noise=noise)
    second_prediction = model(second_x_t, timestep)
    second_loss = F.mse_loss(second_prediction, noise)

    x_t_at_zero = schedule.q_sample(
        x_start, torch.tensor([0], dtype=torch.long), noise=noise
    )
    x_t_at_final = schedule.q_sample(
        x_start, torch.tensor([999], dtype=torch.long), noise=noise
    )
    prediction_at_zero = model(x_t_at_zero, torch.tensor([0], dtype=torch.long))
    prediction_at_final = model(
        x_t_at_final, torch.tensor([999], dtype=torch.long)
    )

    assert torch.equal(first_x_t, second_x_t)
    assert torch.equal(first_prediction, second_prediction)
    assert torch.equal(first_loss, second_loss)
    assert not torch.equal(x_t_at_zero, x_t_at_final)
    assert not torch.equal(prediction_at_zero, prediction_at_final)


def test_tiny_unet_is_directly_compatible_with_the_reverse_sampling_loop() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=4)
    model = TinyUNet(dropout=0.0)
    model.eval()
    x_T = torch.randn(1, 3, 32, 32)

    def noise_predictor(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return model(x_t, timesteps)

    direct_prediction = noise_predictor(x_T, torch.tensor([3], dtype=torch.long))
    final_sample = schedule.sample_loop_from_noise_predictor(x_T, noise_predictor)

    assert not model.training
    assert direct_prediction.shape == x_T.shape
    assert final_sample.shape == x_T.shape
    assert torch.isfinite(final_sample).all()
    assert not final_sample.requires_grad


def test_schedule_and_unet_reject_incompatible_timestep_batches() -> None:
    schedule = LinearDDPMSchedule()
    model = TinyUNet()
    x_start = torch.zeros(2, 3, 32, 32)
    noise = torch.zeros_like(x_start)
    invalid_timesteps = torch.tensor([0], dtype=torch.long)

    try:
        schedule.q_sample(x_start, invalid_timesteps, noise=noise)
    except ValueError as error:
        assert "one value per batch" in str(error)
    else:
        raise AssertionError("schedule accepted incompatible timestep batch")

    try:
        model(x_start, invalid_timesteps)
    except ValueError as error:
        assert "same batch size" in str(error)
    else:
        raise AssertionError("model accepted incompatible timestep batch")
