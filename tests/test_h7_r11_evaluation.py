from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_avatar_diffusion import h7_r11_evaluation as adapter

CA16_ROOT = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")


def test_only_the_three_authorized_seeds_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only seeds"):
        adapter.verify_terminal_checkpoint(tmp_path / "x.pt", 9)


def test_frozen_plan_has_r11_roles_counts_and_seeds() -> None:
    plan = adapter.frozen_plan(CA16_ROOT)
    assert {key: len(value) for key, value in plan.items()} == {
        "ID_VAL": 256,
        "COMP_VAL": 237,
    }
    assert [row["generation_seed"] for row in plan["COMP_VAL"][:3]] == [
        2026,
        2027,
        2028,
    ]
    assert all(
        isinstance(row["token_ids"], list) and isinstance(row["attention_mask"], list)
        for rows in plan.values()
        for row in rows
    )
    assert all(
        "FINAL_OOD" not in row["split_role"] and "ID_TEST" not in row["split_role"]
        for rows in plan.values()
        for row in rows
    )


def test_family_macro_is_equal_weighted_not_record_weighted() -> None:
    plan = adapter.frozen_plan(CA16_ROOT)
    predictions = {
        role: [dict(row["ground_truth_attributes"]) for row in rows]
        for role, rows in plan.items()
    }
    metrics = adapter.aggregate_predictions(plan, predictions)
    assert metrics["id_maa"] == 1.0
    assert metrics["comp_family_macro_maa"] == 1.0
    assert metrics["comp_macro_psr"] == 1.0


def test_external_root_must_be_separate_worktree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        adapter.verify_ca16_assets(tmp_path)


def test_atomic_publication_recovers_no_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "out" / "preflight.json"
    adapter.atomic_json(target, {"status": "ok"})
    assert json.loads(target.read_text()) == {"status": "ok"}
    assert not list(target.parent.glob("*.tmp"))


def test_execute_refuses_cpu_before_loading_or_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.execute(
            ca16_assets_root=CA16_ROOT,
            terminal_checkpoint=Path("missing.pt"),
            seed=2026,
        )
