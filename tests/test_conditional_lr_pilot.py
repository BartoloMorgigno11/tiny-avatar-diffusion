import pytest

from tiny_avatar_diffusion.conditional_lr_pilot import (
    LEARNING_RATES,
    LRPilotConfig,
    select_learning_rate,
)


def _result(learning_rate: float, score: float, *, eligible: bool = True) -> dict[str, object]:
    return {
        "run_id": f"lr_{learning_rate:.0e}",
        "learning_rate": learning_rate,
        "eligible": eligible,
        "selection_score": score if eligible else None,
    }


def test_lr_pilot_configuration_is_frozen() -> None:
    config = LRPilotConfig()

    assert config.learning_rates == LEARNING_RATES == (5e-5, 1e-4, 2e-4)
    assert config.physical_batch_size == 32
    assert config.steps == 2_000
    with pytest.raises(ValueError, match="learning_rates"):
        LRPilotConfig(learning_rates=(1e-4,))
    with pytest.raises(ValueError, match="physical_batch_size"):
        LRPilotConfig(physical_batch_size=16)


def test_selection_uses_final_score_and_conservative_one_percent_tie_break() -> None:
    selection = select_learning_rate(
        [_result(5e-5, 0.402), _result(1e-4, 0.400), _result(2e-4, 0.407)]
    )

    assert selection["selected_learning_rate"] == 5e-5
    assert selection["strength"] == "WEAK"
    assert selection["tied_run_ids"] == ["lr_5e-05", "lr_1e-04"]


def test_selection_picks_clearly_better_candidate_and_reports_no_eligible_case() -> None:
    clear = select_learning_rate(
        [_result(5e-5, 0.421), _result(1e-4, 0.418), _result(2e-4, 0.390)]
    )
    none = select_learning_rate([_result(5e-5, 1.0, eligible=False)])

    assert clear["selected_learning_rate"] == 2e-4
    assert clear["strength"] == "CLEAR"
    assert none["selected_learning_rate"] is None
    assert none["strength"] == "INCONCLUSIVE"
