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
