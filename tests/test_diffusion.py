import pytest
import torch

from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule

COEFFICIENT_NAMES = (
    "betas",
    "alphas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "sqrt_recip_alphas",
    "posterior_variance",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
)


def test_linear_ddpm_schedule_shapes_and_endpoints() -> None:
    schedule = LinearDDPMSchedule()

    for name in COEFFICIENT_NAMES:
        coefficient = getattr(schedule, name)
        assert coefficient.shape == (1_000,)
        assert not coefficient.requires_grad
        assert torch.isfinite(coefficient).all()

    assert torch.isclose(schedule.betas[0], torch.tensor(1e-4))
    assert torch.isclose(schedule.betas[-1], torch.tensor(0.02))
    assert torch.all((schedule.betas > 0.0) & (schedule.betas < 1.0))


def test_linear_ddpm_schedule_is_explicit_float32_and_linear() -> None:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        schedule = LinearDDPMSchedule()
    finally:
        torch.set_default_dtype(previous_dtype)

    expected_betas = torch.linspace(1e-4, 0.02, 1_000, dtype=torch.float32)
    assert schedule.betas.dtype == torch.float32
    assert torch.equal(schedule.betas, expected_betas)


def test_linear_ddpm_schedule_forward_coefficients() -> None:
    schedule = LinearDDPMSchedule()

    assert torch.allclose(schedule.alphas, 1.0 - schedule.betas)
    assert torch.all(schedule.alphas_cumprod[1:] < schedule.alphas_cumprod[:-1])
    assert schedule.alphas_cumprod_prev[0] == 1.0
    assert torch.allclose(
        schedule.alphas_cumprod_prev[1:], schedule.alphas_cumprod[:-1]
    )


def test_linear_ddpm_schedule_cumprod_and_square_root_coefficients() -> None:
    schedule = LinearDDPMSchedule()

    expected_cumprod = torch.cumprod(schedule.alphas, dim=0)
    assert torch.allclose(schedule.alphas_cumprod, expected_cumprod)
    assert torch.allclose(
        schedule.sqrt_alphas_cumprod, torch.sqrt(schedule.alphas_cumprod)
    )
    assert torch.allclose(
        schedule.sqrt_one_minus_alphas_cumprod,
        torch.sqrt(1.0 - schedule.alphas_cumprod),
    )
    assert torch.allclose(schedule.sqrt_recip_alphas, 1.0 / torch.sqrt(schedule.alphas))


def test_linear_ddpm_schedule_posterior_coefficients_match_definitions() -> None:
    schedule = LinearDDPMSchedule()
    denominator = 1.0 - schedule.alphas_cumprod
    expected_variance = (
        schedule.betas * (1.0 - schedule.alphas_cumprod_prev) / denominator
    )
    expected_coef1 = (
        schedule.betas * torch.sqrt(schedule.alphas_cumprod_prev) / denominator
    )
    expected_coef2 = (
        torch.sqrt(schedule.alphas)
        * (1.0 - schedule.alphas_cumprod_prev)
        / denominator
    )

    assert torch.all(schedule.posterior_variance >= 0.0)
    assert schedule.posterior_variance[0] == 0.0
    assert torch.allclose(schedule.posterior_variance, expected_variance)
    assert torch.allclose(schedule.posterior_mean_coef1, expected_coef1)
    assert torch.allclose(schedule.posterior_mean_coef2, expected_coef2)


def test_linear_ddpm_schedule_coefficients_are_buffers_and_follow_to() -> None:
    schedule = LinearDDPMSchedule()

    assert set(dict(schedule.named_buffers())) == set(COEFFICIENT_NAMES)
    assert not dict(schedule.named_parameters())

    schedule.to(dtype=torch.float64)
    for name in COEFFICIENT_NAMES:
        coefficient = getattr(schedule, name)
        assert coefficient.dtype == torch.float64
        assert not coefficient.requires_grad


@pytest.mark.parametrize("timestep", (0, 500, 999))
def test_q_sample_matches_the_forward_noising_formula(timestep: int) -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2)
    noise = torch.linspace(-1.0, 1.0, x_start.numel()).reshape_as(x_start)
    timesteps = torch.full((x_start.shape[0],), timestep, dtype=torch.long)

    result = schedule.q_sample(x_start, timesteps, noise)
    sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[timestep]
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alphas_cumprod[timestep]
    expected = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    assert result.shape == x_start.shape
    assert torch.allclose(result, expected)


def test_q_sample_uses_one_timestep_per_sample_with_channel_broadcasting() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.zeros(3, 2, 4, 5)
    noise = torch.ones_like(x_start)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)

    result = schedule.q_sample(x_start, timesteps, noise)
    expected = schedule.sqrt_one_minus_alphas_cumprod[timesteps].reshape(3, 1, 1, 1)

    assert torch.allclose(result, expected.expand_as(x_start))
    assert not torch.isclose(result[0, 0, 0, 0], result[1, 0, 0, 0])
    assert not torch.isclose(result[1, 0, 0, 0], result[2, 0, 0, 0])


def test_q_sample_generates_finite_noise_when_not_provided() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.zeros(2, 3, 4, 5)
    timesteps = torch.tensor([0, 999], dtype=torch.long)

    result = schedule.q_sample(x_start, timesteps)

    assert result.shape == x_start.shape
    assert torch.isfinite(result).all()


def test_q_sample_is_deterministic_with_explicit_noise() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.ones(2, 3, 4, 5)
    noise = torch.full_like(x_start, 0.25)
    timesteps = torch.tensor([1, 500], dtype=torch.long)

    first = schedule.q_sample(x_start, timesteps, noise)
    second = schedule.q_sample(x_start, timesteps, noise)

    assert torch.equal(first, second)


def test_q_sample_preserves_the_gradient_path_to_x_start() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.ones(2, 3, 4, 5, requires_grad=True)
    noise = torch.zeros_like(x_start)
    timesteps = torch.tensor([0, 500], dtype=torch.long)

    schedule.q_sample(x_start, timesteps, noise).square().mean().backward()

    assert x_start.grad is not None
    assert torch.isfinite(x_start.grad).all()


def test_q_sample_rejects_evidently_invalid_inputs() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.zeros(2, 3, 4, 5)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="one value per batch"):
        schedule.q_sample(x_start, torch.tensor([0], dtype=torch.long))
    with pytest.raises(ValueError, match="schedule range"):
        schedule.q_sample(x_start, torch.tensor([0, 1_000], dtype=torch.long))
    with pytest.raises(ValueError, match="same shape"):
        schedule.q_sample(x_start, timesteps, torch.zeros(2, 3, 4, 4))
