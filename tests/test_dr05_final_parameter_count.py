from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tiny_avatar_diffusion.dr05_parameters import (
    EXPECTED_COUNTS,
    OUTPUT_RELATIVE_PATH,
    parameter_counts,
    validate_expected_hash,
    write_parameter_summary,
)


def test_parameter_counts_match_module_iterator_and_respect_requires_grad() -> None:
    module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1, bias=False))
    module[1].weight.requires_grad_(False)
    counted = parameter_counts([module])
    assert counted["total_parameters"] == sum(parameter.numel() for parameter in module.parameters())
    assert counted["trainable_parameters"] == sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_parameter_counts_do_not_double_count_shared_references() -> None:
    shared = nn.Linear(2, 2)
    assert parameter_counts([shared, shared]) == {"total_parameters": 6, "trainable_parameters": 6}


def test_frozen_arithmetic_and_evaluator_separation() -> None:
    assert EXPECTED_COUNTS["unconditional_backbone"] + EXPECTED_COUNTS["cross_attention"] == EXPECTED_COUNTS["conditional_unet"]
    assert EXPECTED_COUNTS["conditional_unet"] + EXPECTED_COUNTS["text_encoder"] == EXPECTED_COUNTS["full_conditional"]
    assert EXPECTED_COUNTS["unconditional_backbone"] + EXPECTED_COUNTS["cross_attention"] + EXPECTED_COUNTS["text_encoder"] == EXPECTED_COUNTS["full_conditional"]
    assert EXPECTED_COUNTS["evaluator"] not in (EXPECTED_COUNTS["full_conditional"], EXPECTED_COUNTS["unconditional"])


def test_schema_writer_is_fail_closed_and_preserves_required_sections(tmp_path: Path) -> None:
    summary = {
        "conditional": {"full_system": {"total_parameters": 1, "trainable_parameters": 1}},
        "unconditional": {"total_parameters": 1, "trainable_parameters": 1},
        "evaluation_instrument": {"total_parameters": 1, "trainable_parameters": 1, "explicitly_excluded_from_generator_counts": True},
        "checkpoint_provenance": {"conditional_generator": {}},
        "anti_double_count": {"pass": True},
    }
    output = write_parameter_summary(tmp_path, summary)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / OUTPUT_RELATIVE_PATH
    assert {"conditional", "unconditional", "evaluation_instrument", "checkpoint_provenance", "anti_double_count"} <= parsed.keys()
    with pytest.raises(FileExistsError):
        write_parameter_summary(tmp_path, summary)


def test_hash_guard_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"canonical")
    digest = hashlib.sha256(b"canonical").hexdigest()
    assert validate_expected_hash(path, digest) == digest
    with pytest.raises(ValueError, match="unexpected SHA256"):
        validate_expected_hash(path, "0" * 64)
