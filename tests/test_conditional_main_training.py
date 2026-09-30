from __future__ import annotations

import copy

import pytest
import torch

from tiny_avatar_diffusion.conditional_main_training import (
    CHECKPOINT_SCHEMA,
    ConditionalEMA,
    MainTrainingConfig,
    TrainingProgress,
    atomic_save_checkpoint,
    build_checkpoint,
    load_checkpoint,
    fixed_preview_manifest,
    _append_metric,
    tensor_audit,
    compare_continuation_audits,
)


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def test_ema_starts_as_exact_raw_copy_updates_both_components_and_never_requires_grad() -> None:
    text = torch.nn.Linear(3, 2)
    denoiser = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = ConditionalEMA(text, denoiser)
    assert _state_equal(ema.text_encoder, text.state_dict())
    assert _state_equal(ema.denoiser, denoiser.state_dict())
    assert all(not value.requires_grad for value in [*ema.text_encoder.values(), *ema.denoiser.values()])
    text.weight.data.add_(2.0)
    denoiser[0].weight.data.add_(4.0)
    before_text, before_denoiser = ema.text_encoder["weight"].clone(), ema.denoiser["0.weight"].clone()
    ema.update(text, denoiser, decay=0.5)
    assert torch.equal(ema.text_encoder["weight"], (before_text + text.weight) / 2)
    assert torch.equal(ema.denoiser["0.weight"], (before_denoiser + denoiser[0].weight) / 2)


def test_ema_preview_swap_restores_raw_weights_exactly() -> None:
    text, denoiser = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    ema = ConditionalEMA(text, denoiser)
    text.weight.data.add_(1.0); denoiser.bias.data.sub_(1.0)
    raw_text, raw_denoiser = copy.deepcopy(text.state_dict()), copy.deepcopy(denoiser.state_dict())
    with ema.applied_to(text, denoiser):
        assert _state_equal(text.state_dict(), ema.text_encoder)
        assert _state_equal(denoiser.state_dict(), ema.denoiser)
    assert _state_equal(text.state_dict(), raw_text)
    assert _state_equal(denoiser.state_dict(), raw_denoiser)


def test_checkpoint_round_trip_carries_raw_ema_optimizer_scaler_progress_and_rejects_config_mix(tmp_path) -> None:
    config = MainTrainingConfig(run_id="cpu-checkpoint")
    text, denoiser = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    ema = ConditionalEMA(text, denoiser)
    optimizer = torch.optim.AdamW([*text.parameters(), *denoiser.parameters()], lr=config.learning_rate)
    for parameter in [*text.parameters(), *denoiser.parameters()]: parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler, corruption = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    progress = TrainingProgress(step=7, images_seen=224, batch_offset=32, permutation=torch.tensor([1, 0]), train_losses=[1.0])
    checkpoint = build_checkpoint(config=config, progress=progress, text_encoder=text, denoiser=denoiser, ema=ema, optimizer=optimizer, scaler=scaler, sampler_generator=sampler, corruption_generator=corruption, validation_history=[{"step": 0}])
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    path = tmp_path / "latest.pt"; atomic_save_checkpoint(path, checkpoint)
    restored_text, restored_denoiser = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    restored_ema = ConditionalEMA(restored_text, restored_denoiser)
    restored_optimizer = torch.optim.AdamW([*restored_text.parameters(), *restored_denoiser.parameters()], lr=config.learning_rate)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_sampler, restored_corruption = torch.Generator(), torch.Generator()
    restored_progress, history = load_checkpoint(path, config=config, text_encoder=restored_text, denoiser=restored_denoiser, ema=restored_ema, optimizer=restored_optimizer, scaler=restored_scaler, sampler_generator=restored_sampler, corruption_generator=restored_corruption, map_location=torch.device("cpu"))
    assert restored_progress.step == 7 and history == [{"step": 0}]
    assert _state_equal(restored_text.state_dict(), text.state_dict())
    assert _state_equal(restored_ema.text_encoder, ema.text_encoder)
    with pytest.raises(ValueError, match="run ID"):
        load_checkpoint(path, config=MainTrainingConfig(run_id="other"), text_encoder=restored_text, denoiser=restored_denoiser, ema=restored_ema, optimizer=restored_optimizer, scaler=restored_scaler, sampler_generator=restored_sampler, corruption_generator=restored_corruption, map_location=torch.device("cpu"))


def test_fixed_preview_manifest_is_reproducible_and_validation_only() -> None:
    first, second = fixed_preview_manifest(), fixed_preview_manifest()
    assert first == second and len(first) == 4
    assert [entry["split"] for entry in first] == ["ID_VAL", "ID_VAL", "COMP_VAL", "COMP_VAL"]
    assert [entry["generation_seed"] for entry in first] == [2026, 2027, 2028, 2029]


def test_metric_export_is_appendable_jsonl(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    _append_metric(path, {"kind": "train", "step": 1})
    _append_metric(path, {"kind": "validation", "step": 1, "weight_set": "RAW"})
    assert [line.split('"kind": "')[1].split('"')[0] for line in path.read_text().splitlines()] == ["train", "validation"]


def test_compact_tensor_audit_and_comparator_detect_changes() -> None:
    tensor = torch.tensor([[1.0, 2.0]])
    assert tensor_audit(tensor) == tensor_audit(tensor.clone())
    assert tensor_audit(tensor) != tensor_audit(tensor + 1)
    record = {"indices": [1], "images": tensor_audit(tensor), "input_ids": [[1]], "attention_mask": [[1]], "timesteps": [3], "noise": tensor_audit(tensor)}
    assert all(compare_continuation_audits(record, dict(record)).values())
    changed = dict(record); changed["timesteps"] = [4]
    assert not compare_continuation_audits(record, changed)["timesteps"]
