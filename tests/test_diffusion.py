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


def _independent_schedule() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, 1_000, dtype=torch.float32)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


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


@pytest.mark.parametrize("timestep", (0, 500, 999))
def test_q_sample_round_trips_through_x_start_reconstruction(timestep: int) -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.linspace(-1.5, 1.5, 48).reshape(2, 3, 4, 2)
    noise = torch.linspace(1.0, -1.0, x_start.numel()).reshape_as(x_start)
    timesteps = torch.full((x_start.shape[0],), timestep, dtype=torch.long)

    x_t = schedule.q_sample(x_start, timesteps, noise)
    reconstructed = schedule.predict_x_start_from_noise(x_t, timesteps, noise)

    assert reconstructed.shape == x_start.shape
    assert torch.isfinite(reconstructed).all()
    assert torch.allclose(reconstructed, x_start, atol=1e-5, rtol=1e-5)


def test_predict_x_start_from_noise_matches_an_independent_formula() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.linspace(-2.0, 2.0, 36).reshape(3, 1, 3, 4)
    noise = torch.linspace(0.75, -0.75, x_t.numel()).reshape_as(x_t)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)
    _, _, alphas_cumprod = _independent_schedule()
    alpha_bar = alphas_cumprod[timesteps].reshape(3, 1, 1, 1)

    result = schedule.predict_x_start_from_noise(x_t, timesteps, noise)
    expected = (x_t - torch.sqrt(1.0 - alpha_bar) * noise) / torch.sqrt(alpha_bar)

    assert result.shape == x_t.shape
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected)


def test_q_posterior_mean_matches_an_independent_formula_for_heterogeneous_batch() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.linspace(-1.0, 1.0, 36).reshape(3, 1, 3, 4)
    x_t = torch.linspace(2.0, -2.0, x_start.numel()).reshape_as(x_start)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)
    betas, alphas, alphas_cumprod = _independent_schedule()
    alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]))
    selected = timesteps.reshape(3, 1, 1, 1)
    denominator = 1.0 - alphas_cumprod[selected]
    coef1 = betas[selected] * torch.sqrt(alphas_cumprod_prev[selected]) / denominator
    coef2 = torch.sqrt(alphas[selected]) * (1.0 - alphas_cumprod_prev[selected]) / denominator

    result = schedule.q_posterior_mean(x_start, x_t, timesteps)
    expected = coef1 * x_start + coef2 * x_t

    assert result.shape == x_start.shape
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected)


def test_predict_x_start_from_noise_does_not_clip() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((1, 1, 2, 2), 2.0)
    noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([999], dtype=torch.long)

    reconstructed = schedule.predict_x_start_from_noise(x_t, timesteps, noise)

    assert torch.all(reconstructed > 1.0)


def test_reverse_deterministic_operations_preserve_gradient_paths() -> None:
    schedule = LinearDDPMSchedule()
    x_start = torch.ones(2, 1, 3, 4, requires_grad=True)
    x_t = torch.full_like(x_start, 0.5, requires_grad=True)
    noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([0, 500], dtype=torch.long)

    reconstructed = schedule.predict_x_start_from_noise(x_t, timesteps, noise)
    posterior_mean = schedule.q_posterior_mean(x_start, x_t, timesteps)
    (reconstructed.square().mean() + posterior_mean.square().mean()).backward()

    assert x_start.grad is not None
    assert x_t.grad is not None
    assert torch.isfinite(x_start.grad).all()
    assert torch.isfinite(x_t.grad).all()


def test_reverse_deterministic_operations_reject_incompatible_shapes() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(2, 1, 3, 4)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same shape as x_t"):
        schedule.predict_x_start_from_noise(x_t, timesteps, torch.zeros(2, 1, 3, 3))
    with pytest.raises(ValueError, match="same shape"):
        schedule.q_posterior_mean(torch.zeros(2, 1, 3, 3), x_t, timesteps)
    with pytest.raises(ValueError, match="one value per batch"):
        schedule.q_posterior_mean(x_t, x_t, torch.tensor([0], dtype=torch.long))


@pytest.mark.parametrize("timestep", (0, 500, 999))
def test_p_mean_variance_from_noise_reuses_reverse_primitives(timestep: int) -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.linspace(-1.5, 1.5, 48).reshape(2, 3, 4, 2)
    predicted_noise = torch.linspace(0.75, -0.75, x_t.numel()).reshape_as(x_t)
    timesteps = torch.full((x_t.shape[0],), timestep, dtype=torch.long)

    result = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)
    expected_x_start = schedule.predict_x_start_from_noise(
        x_t, timesteps, predicted_noise
    )
    expected_model_mean = schedule.q_posterior_mean(expected_x_start, x_t, timesteps)
    expected_variance = schedule.posterior_variance[timesteps].reshape(2, 1, 1, 1)

    assert set(result) == {"model_mean", "posterior_variance", "predicted_x_start"}
    assert torch.allclose(result["predicted_x_start"], expected_x_start)
    assert torch.allclose(result["model_mean"], expected_model_mean)
    assert torch.allclose(result["posterior_variance"], expected_variance)
    assert result["model_mean"].shape == x_t.shape
    assert result["posterior_variance"].shape == (2, 1, 1, 1)
    assert torch.isfinite(result["model_mean"]).all()
    assert torch.isfinite(result["predicted_x_start"]).all()
    assert torch.isfinite(result["posterior_variance"]).all()


def test_p_mean_variance_from_noise_supports_heterogeneous_timesteps() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(3, 2, 4, 5)
    predicted_noise = torch.ones_like(x_t)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)

    result = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)

    assert result["predicted_x_start"].shape == x_t.shape
    assert result["model_mean"].shape == x_t.shape
    assert result["posterior_variance"].shape == (3, 1, 1, 1)
    assert result["posterior_variance"][0] == 0.0
    assert not torch.isclose(
        result["posterior_variance"][1, 0, 0, 0],
        result["posterior_variance"][2, 0, 0, 0],
    )


def test_p_mean_variance_from_noise_matches_an_independent_formula() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.linspace(-2.0, 2.0, 36).reshape(3, 1, 3, 4)
    predicted_noise = torch.linspace(-0.5, 0.5, x_t.numel()).reshape_as(x_t)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)
    betas, alphas, alphas_cumprod = _independent_schedule()
    alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]))
    selected = timesteps.reshape(3, 1, 1, 1)
    alpha_bar = alphas_cumprod[selected]
    denominator = 1.0 - alpha_bar
    predicted_x_start = (
        x_t - torch.sqrt(1.0 - alpha_bar) * predicted_noise
    ) / torch.sqrt(alpha_bar)
    coef1 = betas[selected] * torch.sqrt(alphas_cumprod_prev[selected]) / denominator
    coef2 = torch.sqrt(alphas[selected]) * (1.0 - alphas_cumprod_prev[selected]) / denominator
    expected_model_mean = coef1 * predicted_x_start + coef2 * x_t
    expected_variance = betas[selected] * (1.0 - alphas_cumprod_prev[selected]) / denominator

    result = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)

    assert torch.allclose(result["predicted_x_start"], predicted_x_start)
    assert torch.allclose(result["model_mean"], expected_model_mean)
    assert torch.allclose(result["posterior_variance"], expected_variance)


def test_p_mean_variance_from_noise_does_not_clip_predicted_x_start() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((1, 1, 2, 2), 2.0)
    predicted_noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([999], dtype=torch.long)

    result = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)

    assert torch.all(result["predicted_x_start"] > 1.0)


def test_p_mean_variance_from_noise_preserves_gradient_paths() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((2, 1, 3, 4), 0.5, requires_grad=True)
    predicted_noise = torch.full_like(x_t, 0.25, requires_grad=True)
    timesteps = torch.tensor([0, 500], dtype=torch.long)

    result = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)
    (result["model_mean"].square().mean() + result["predicted_x_start"].square().mean()).backward()

    assert x_t.grad is not None
    assert predicted_noise.grad is not None
    assert torch.isfinite(x_t.grad).all()
    assert torch.isfinite(predicted_noise.grad).all()


def test_p_mean_variance_from_noise_rejects_incompatible_inputs() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(2, 1, 3, 4)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same shape as x_t"):
        schedule.p_mean_variance_from_noise(
            x_t, timesteps, torch.zeros(2, 1, 3, 3)
        )
    with pytest.raises(ValueError, match="one value per batch"):
        schedule.p_mean_variance_from_noise(
            x_t, torch.tensor([0], dtype=torch.long), torch.zeros_like(x_t)
        )


@pytest.mark.parametrize("timestep", (500, 999))
def test_p_sample_from_noise_matches_the_nonfinal_step_formula(timestep: int) -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.linspace(-1.5, 1.5, 48).reshape(2, 3, 4, 2)
    predicted_noise = torch.linspace(-0.75, 0.75, x_t.numel()).reshape_as(x_t)
    noise = torch.linspace(1.0, -1.0, x_t.numel()).reshape_as(x_t)
    timesteps = torch.full((x_t.shape[0],), timestep, dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)
    bridge = schedule.p_mean_variance_from_noise(x_t, timesteps, predicted_noise)
    expected = bridge["model_mean"] + torch.sqrt(bridge["posterior_variance"]) * noise

    assert set(result) == {
        "sample",
        "predicted_x_start",
        "model_mean",
        "posterior_variance",
    }
    assert torch.allclose(result["sample"], expected)
    assert torch.allclose(result["predicted_x_start"], bridge["predicted_x_start"])
    assert torch.allclose(result["model_mean"], bridge["model_mean"])
    assert torch.allclose(result["posterior_variance"], bridge["posterior_variance"])


def test_p_sample_from_noise_omits_noise_at_the_final_step() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((2, 1, 3, 4), 0.5)
    predicted_noise = torch.full_like(x_t, -0.25)
    noise = torch.full_like(x_t, 1_000.0)
    timesteps = torch.zeros(x_t.shape[0], dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)

    assert result["posterior_variance"].eq(0.0).all()
    assert torch.equal(result["sample"], result["model_mean"])
    assert torch.isfinite(result["sample"]).all()


def test_p_sample_from_noise_masks_a_heterogeneous_batch() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(3, 2, 4, 5)
    predicted_noise = torch.ones_like(x_t)
    noise = torch.ones_like(x_t)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)
    expected_mask = (timesteps != 0).reshape(3, 1, 1, 1)
    expected = result["model_mean"] + (
        expected_mask * torch.sqrt(result["posterior_variance"]) * noise
    )

    assert result["sample"].shape == x_t.shape
    assert torch.allclose(result["sample"], expected)
    assert torch.equal(result["sample"][0], result["model_mean"][0])
    assert not torch.equal(result["sample"][1], result["model_mean"][1])
    assert not torch.equal(result["sample"][2], result["model_mean"][2])


def test_p_sample_from_noise_matches_an_independent_formula() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.linspace(-2.0, 2.0, 36).reshape(3, 1, 3, 4)
    predicted_noise = torch.linspace(-0.5, 0.5, x_t.numel()).reshape_as(x_t)
    noise = torch.linspace(0.25, -0.25, x_t.numel()).reshape_as(x_t)
    timesteps = torch.tensor([0, 500, 999], dtype=torch.long)
    betas, alphas, alphas_cumprod = _independent_schedule()
    alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]))
    selected = timesteps.reshape(3, 1, 1, 1)
    alpha_bar = alphas_cumprod[selected]
    denominator = 1.0 - alpha_bar
    predicted_x_start = (
        x_t - torch.sqrt(1.0 - alpha_bar) * predicted_noise
    ) / torch.sqrt(alpha_bar)
    coef1 = betas[selected] * torch.sqrt(alphas_cumprod_prev[selected]) / denominator
    coef2 = torch.sqrt(alphas[selected]) * (1.0 - alphas_cumprod_prev[selected]) / denominator
    model_mean = coef1 * predicted_x_start + coef2 * x_t
    posterior_variance = betas[selected] * (1.0 - alphas_cumprod_prev[selected]) / denominator
    expected_sample = model_mean + (
        (timesteps != 0).reshape(3, 1, 1, 1)
        * torch.sqrt(posterior_variance)
        * noise
    )

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)

    assert torch.allclose(result["predicted_x_start"], predicted_x_start)
    assert torch.allclose(result["model_mean"], model_mean)
    assert torch.allclose(result["posterior_variance"], posterior_variance)
    assert torch.allclose(result["sample"], expected_sample)


def test_p_sample_from_noise_generates_finite_noise_when_not_provided() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(2, 3, 4, 5)
    predicted_noise = torch.ones_like(x_t)
    timesteps = torch.tensor([0, 999], dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise)

    assert result["sample"].shape == x_t.shape
    assert torch.isfinite(result["sample"]).all()


def test_p_sample_from_noise_is_deterministic_with_explicit_noise() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.ones(2, 1, 3, 4)
    predicted_noise = torch.full_like(x_t, 0.5)
    noise = torch.full_like(x_t, -0.25)
    timesteps = torch.tensor([500, 999], dtype=torch.long)

    first = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)
    second = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)

    assert torch.equal(first["sample"], second["sample"])


def test_p_sample_from_noise_does_not_clip() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((1, 1, 2, 2), 2.0)
    predicted_noise = torch.zeros_like(x_t)
    noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([999], dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)

    assert torch.all(result["predicted_x_start"] > 1.0)
    assert torch.all(result["sample"] > 1.0)


def test_p_sample_from_noise_preserves_gradient_paths() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.full((2, 1, 3, 4), 0.5, requires_grad=True)
    predicted_noise = torch.full_like(x_t, 0.25, requires_grad=True)
    noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([0, 500], dtype=torch.long)

    result = schedule.p_sample_from_noise(x_t, timesteps, predicted_noise, noise)
    (result["sample"].square().mean() + result["predicted_x_start"].square().mean()).backward()

    assert x_t.grad is not None
    assert predicted_noise.grad is not None
    assert torch.isfinite(x_t.grad).all()
    assert torch.isfinite(predicted_noise.grad).all()


def test_p_sample_from_noise_rejects_incompatible_inputs() -> None:
    schedule = LinearDDPMSchedule()
    x_t = torch.zeros(2, 1, 3, 4)
    predicted_noise = torch.zeros_like(x_t)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same shape as x_t"):
        schedule.p_sample_from_noise(
            x_t, timesteps, predicted_noise, torch.zeros(2, 1, 3, 3)
        )
    with pytest.raises(ValueError, match="one value per batch"):
        schedule.p_sample_from_noise(
            x_t, torch.tensor([0], dtype=torch.long), predicted_noise
        )
    with pytest.raises(ValueError, match="schedule range"):
        schedule.p_sample_from_noise(
            x_t, torch.tensor([0, 1_000], dtype=torch.long), predicted_noise
        )


def test_sample_loop_runs_all_timesteps_and_propagates_each_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = LinearDDPMSchedule(num_timesteps=4)
    x_T = torch.zeros(2, 1, 2, 2)
    predictor_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    reverse_calls: list[torch.Tensor] = []

    def noise_predictor(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        predictor_calls.append((x_t.clone(), timesteps.clone()))
        return torch.zeros_like(x_t)

    def controlled_reverse_step(
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del predicted_noise
        reverse_calls.append(timesteps.clone())
        return {"sample": x_t + timesteps.reshape(-1, 1, 1, 1) + 1.0}

    monkeypatch.setattr(schedule, "p_sample_from_noise", controlled_reverse_step)
    final_sample = schedule.sample_loop_from_noise_predictor(x_T, noise_predictor)

    assert [call[1][0].item() for call in predictor_calls] == [3, 2, 1, 0]
    assert len(predictor_calls) == schedule.betas.shape[0]
    assert len(reverse_calls) == schedule.betas.shape[0]
    for _, timesteps in predictor_calls:
        assert timesteps.shape == (x_T.shape[0],)
        assert torch.equal(timesteps, torch.full_like(timesteps, timesteps[0]))
    assert [call[0].item() for call in reverse_calls] == [3, 2, 1, 0]
    assert torch.equal(predictor_calls[0][0], x_T)
    assert torch.equal(predictor_calls[1][0], torch.full_like(x_T, 4.0))
    assert torch.equal(predictor_calls[2][0], torch.full_like(x_T, 7.0))
    assert torch.equal(predictor_calls[3][0], torch.full_like(x_T, 9.0))
    assert torch.equal(final_sample, torch.full_like(x_T, 10.0))


def test_sample_loop_uses_no_grad_and_real_reverse_primitives() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=4)
    x_T = torch.ones(2, 1, 2, 2, requires_grad=True)
    predictor_grad_enabled: list[bool] = []

    def noise_predictor(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        predictor_grad_enabled.append(torch.is_grad_enabled())
        return x_t * 0.0

    final_sample = schedule.sample_loop_from_noise_predictor(x_T, noise_predictor)

    assert len(predictor_grad_enabled) == 4
    assert not any(predictor_grad_enabled)
    assert final_sample.shape == x_T.shape
    assert torch.isfinite(final_sample).all()
    assert not final_sample.requires_grad


@pytest.mark.parametrize(
    ("predictor", "error_type", "message"),
    (
        (lambda x_t, timesteps: torch.zeros(1), ValueError, "same shape"),
        (lambda x_t, timesteps: "not a tensor", TypeError, "torch.Tensor"),
    ),
)
def test_sample_loop_rejects_invalid_predictor_outputs(
    predictor: object,
    error_type: type[Exception],
    message: str,
) -> None:
    schedule = LinearDDPMSchedule(num_timesteps=4)
    x_T = torch.zeros(2, 1, 2, 2)

    with pytest.raises(error_type, match=message):
        schedule.sample_loop_from_noise_predictor(x_T, predictor)  # type: ignore[arg-type]


def test_sample_loop_rejects_invalid_initial_inputs() -> None:
    schedule = LinearDDPMSchedule(num_timesteps=4)
    predictor = lambda x_t, timesteps: torch.zeros_like(x_t)

    with pytest.raises(ValueError, match="batch dimension"):
        schedule.sample_loop_from_noise_predictor(torch.tensor(1.0), predictor)
    with pytest.raises(ValueError, match="non-empty"):
        schedule.sample_loop_from_noise_predictor(torch.empty(0, 1, 2, 2), predictor)
    with pytest.raises(TypeError, match="callable"):
        schedule.sample_loop_from_noise_predictor(torch.zeros(1, 1, 2, 2), None)  # type: ignore[arg-type]
