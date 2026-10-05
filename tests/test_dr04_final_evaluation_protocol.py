import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "dr04_final_eval_v1"


def load(name):
    return json.loads((OUT / name).read_text())


def test_frozen_main_panel_invariants():
    panel = load("final_prompt_panel_manifest.json")["prompt_records"]
    assert len(panel) == 252 == len({x["prompt_id"] for x in panel}) == len({x["caption"] for x in panel})
    ood = [x for x in panel if x["domain"] == "OOD"]
    controls = [x for x in panel if x["domain"] == "ID"]
    assert len(ood) == 100 and len(controls) == 152
    assert all(sum(x["pair_id"] == pair for x in ood) == 25 for pair in ("OOD1", "OOD2", "OOD3", "OOD4"))
    assert all(sum(x["source_pair_id"] == pair and x["control_type"] == typ for x in controls) == 19 for pair in ("OOD1", "OOD2", "OOD3", "OOD4") for typ in ("A_FIXED", "B_FIXED"))
    grouped = {x["prompt_id"]: [c for c in controls if c["matched_ood_prompt_id"] == x["prompt_id"]] for x in ood}
    assert sum(len(v) == 2 for v in grouped.values()) == 76
    assert sum(len(v) == 0 for v in grouped.values()) == 24
    assert all(len(v) in (0, 2) for v in grouped.values())
    assert {str(i): sum(x["nuisance_cost"] == i for x in controls) for i in range(3)} == {"0": 60, "1": 92, "2": 0}


def test_seed_and_minimal_edit_invariants():
    seeds = load("seed_manifest.json")
    pairs = load("minimal_edit_pair_manifest.json")["pair_records"]
    assert seeds["generation_seeds"] == [2026, 2027, 2028, 2029, 2030]
    assert seeds["expected_main_generation_count"] == 1260
    assert len(pairs) == 40
    assert len({c for p in pairs for c in (p["caption_a"], p["caption_b"])}) == 80
    for attr in ("face_shape", "glasses", "hair_color", "face_color"):
        assert sum(p["target_attribute"] == attr for p in pairs) == 10
    for pair in pairs:
        a, b = pair["attributes_a"], pair["attributes_b"]
        assert sum(a[k] != b[k] for k in a) == 1


def test_checkpoint_and_access_contract():
    provenance = load("final_eval_provenance.json")
    access = load("controlled_unseal_access_log.json")
    assert provenance["generator_checkpoint_sha256"] == "0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884"
    assert provenance["evaluator_checkpoint_sha256"] == "3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f"
    assert not access["ID_TEST_image_content_accessed"]
    assert not access["FINAL_OOD_image_content_accessed"]
    assert not access["generator_executed"] and not access["evaluator_executed"]


def test_closed_freeze_manifest_and_notebook_portability():
    freeze = load("dr04_freeze_manifest.json")
    notebook_names = {
        "01_data_and_protocol.ipynb",
        "02_model_and_diffusion.ipynb",
        "03_evaluation_and_results.ipynb",
        "04_interactive_generation.ipynb",
    }
    notebook_root = ROOT / "notebooks"
    assert freeze["status"] == "CLOSED"
    assert freeze["closure_audit_status"] == "PASS"
    assert not freeze["final_generation_executed"]
    assert {path.name for path in notebook_root.glob("*.ipynb")} == notebook_names
    for name in notebook_names:
        notebook = (notebook_root / name).read_text(encoding="utf-8")
        assert "C:\\Users\\" not in notebook
        assert "300 prompts" not in notebook
