import json
from pathlib import Path

import pytest
import torch

from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ROOT = Path(__file__).resolve().parents[1]
ID_VAL = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_val.jsonl"
COMP_VAL = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "comp_val.jsonl"


def _components() -> tuple[TextConditioningEncoder, ConditionalTinyUNet, LinearDDPMSchedule]:
    return TextConditioningEncoder(), ConditionalTinyUNet(), LinearDDPMSchedule(num_timesteps=4)


def _real_prompt(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return (
        torch.tensor([record["token_ids"]], dtype=torch.long),
        torch.tensor([record["attention_mask"]], dtype=torch.long),
    )


def test_conditional_sampler_full_shape_real_reverse_loop_and_no_cfg() -> None:
    text_encoder, denoiser, schedule = _components()
    input_ids, mask = _real_prompt(ID_VAL)
    calls: list[int] = []
    handle = denoiser.register_forward_hook(lambda *_: calls.append(1))
    try:
        sample = conditional_sample(
            schedule,
            text_encoder,
            denoiser,
            input_ids,
            mask,
            generator=torch.Generator().manual_seed(2026),
        )
    finally:
        handle.remove()

    assert sample.shape == (1, 3, 32, 32)
    assert sample.dtype == torch.float32
    assert torch.isfinite(sample).all()
    assert len(calls) == 4


@pytest.mark.parametrize("clip_denoised", (True, False))
def test_conditional_sampler_propagates_clipping_policy_at_every_reverse_step(
    monkeypatch: pytest.MonkeyPatch, clip_denoised: bool
) -> None:
    text_encoder, denoiser, schedule = _components()
    input_ids, mask = _real_prompt(ID_VAL)
    received_flags: list[bool] = []
    original_reverse_step = schedule.p_sample_from_noise

    def tracked_reverse_step(
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        clip_denoised: bool = False,
    ) -> dict[str, torch.Tensor]:
        received_flags.append(clip_denoised)
        return original_reverse_step(
            x_t,
            timesteps,
            predicted_noise,
            noise=noise,
            clip_denoised=clip_denoised,
        )

    monkeypatch.setattr(schedule, "p_sample_from_noise", tracked_reverse_step)
    conditional_sample(
        schedule,
        text_encoder,
        denoiser,
        input_ids,
        mask,
        generator=torch.Generator().manual_seed(2026),
        clip_denoised=clip_denoised,
    )

    assert received_flags == [clip_denoised] * schedule.betas.shape[0]


def test_conditional_sampler_defaults_to_clipping_reconstructed_x0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_encoder, denoiser, schedule = _components()
    input_ids, mask = _real_prompt(ID_VAL)
    received_flags: list[bool] = []
    original_reverse_step = schedule.p_sample_from_noise

    def tracked_reverse_step(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        received_flags.append(bool(kwargs["clip_denoised"]))
        return original_reverse_step(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(schedule, "p_sample_from_noise", tracked_reverse_step)
    conditional_sample(
        schedule,
        text_encoder,
        denoiser,
        input_ids,
        mask,
        generator=torch.Generator().manual_seed(2027),
    )

    assert received_flags == [True] * schedule.betas.shape[0]


def test_conditional_sampling_is_deterministic_and_seed_sensitive() -> None:
    text_encoder, denoiser, schedule = _components()
    input_ids, mask = _real_prompt(ID_VAL)
    x_t = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        first = conditional_sample(
            schedule, text_encoder, denoiser, input_ids, mask, x_T=x_t,
            generator=torch.Generator().manual_seed(8)
        )
        repeated = conditional_sample(
            schedule, text_encoder, denoiser, input_ids, mask, x_T=x_t,
            generator=torch.Generator().manual_seed(8)
        )
        different_seed = conditional_sample(
            schedule, text_encoder, denoiser, input_ids, mask,
            generator=torch.Generator().manual_seed(9)
        )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_seed)


def test_conditional_sampling_is_sensitive_to_two_real_dr01_captions() -> None:
    text_encoder, denoiser, schedule = _components()
    id_ids, id_mask = _real_prompt(ID_VAL)
    comp_ids, comp_mask = _real_prompt(COMP_VAL)
    x_t = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(10))
    first = conditional_sample(
        schedule, text_encoder, denoiser, id_ids, id_mask, x_T=x_t,
        generator=torch.Generator().manual_seed(11)
    )
    second = conditional_sample(
        schedule, text_encoder, denoiser, comp_ids, comp_mask, x_T=x_t,
        generator=torch.Generator().manual_seed(11)
    )

    assert not torch.equal(first, second)


def test_text_encoder_runs_once_and_sampling_preserves_modes_gradients_and_parameters() -> None:
    text_encoder, denoiser, schedule = _components()
    text_encoder.train()
    denoiser.train()
    input_ids, mask = _real_prompt(ID_VAL)
    parameters_before = [parameter.detach().clone() for parameter in [*text_encoder.parameters(), *denoiser.parameters()]]
    calls: list[int] = []
    handle = text_encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        sample = conditional_sample(
            schedule, text_encoder, denoiser, input_ids, mask,
            generator=torch.Generator().manual_seed(12)
        )
    finally:
        handle.remove()

    assert len(calls) == 1
    assert not sample.requires_grad
    assert text_encoder.training and denoiser.training
    assert all(parameter.grad is None for parameter in [*text_encoder.parameters(), *denoiser.parameters()])
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip([*text_encoder.parameters(), *denoiser.parameters()], parameters_before, strict=True)
    )


@pytest.mark.parametrize(
    ("input_ids", "mask", "message"),
    [
        (None, torch.ones(1, 7), "input_ids"),
        (torch.ones(1, 7, dtype=torch.long), None, "attention_mask"),
        (torch.ones(1, 8, dtype=torch.long), torch.ones(1, 8), "sequence length"),
    ],
)
def test_conditional_sampling_requires_valid_conditioning(
    input_ids: torch.Tensor | None, mask: torch.Tensor | None, message: str
) -> None:
    text_encoder, denoiser, schedule = _components()
    with pytest.raises(ValueError, match=message):
        conditional_sample(schedule, text_encoder, denoiser, input_ids, mask)
