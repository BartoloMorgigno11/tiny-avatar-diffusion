import json
from pathlib import Path

import pytest

from tiny_avatar_diffusion import h7_r13_pool as pool
from tiny_avatar_diffusion.ca16_evaluation import sha256_file

CA16 = Path(r"C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_CA16")


def test_external_r13_plan_has_frozen_geometry_and_roles():
    plan, identity = pool.prepare_plan(CA16)
    assert len(plan) == 804 and identity["protocol_sha256"] == pool.PROTOCOL_SHA256
    assert {x["split_role"] for x in plan} == {"ID_VAL", "COMP_VAL"}


def test_pool_validation_rejects_missing_or_foreign_pngs(tmp_path: Path) -> None:
    image = tmp_path / "images" / "id_val" / "0000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    plan = [
        {
            "sample_id": "quality_0000",
            "generation_seed": 2026,
            "png_relative_path": "images/id_val/0000.png",
        }
    ]
    identity, terminal = {"plan": "x"}, {"terminal": "x"}
    (tmp_path / "pool_state.json").write_text(
        json.dumps({"identity": identity, "terminal": terminal})
    )
    (tmp_path / "pool_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "quality_0000",
                "generation_seed": 2026,
                "png_relative_path": "images/id_val/0000.png",
                "png_sha256": sha256_file(image),
            }
        )
        + "\n"
    )
    assert (
        pool.validate_pool(tmp_path, plan, identity, terminal)[0]["sample_id"]
        == "quality_0000"
    )
    (tmp_path / "images" / "foreign.png").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="foreign"):
        pool.validate_pool(tmp_path, plan, identity, terminal)
