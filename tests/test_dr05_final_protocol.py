from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dr05_materialize_final_evaluation_protocol.py"
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
ENVIRONMENT = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_environment.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in strings(nested)]
    return []


def load_materializer():
    spec = importlib.util.spec_from_file_location("dr05_final_protocol_materializer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_frozen_and_complete():
    protocol = load()
    assert protocol["protocol_identity"] == {
        "project": "Tiny Text-Conditioned Avatar Diffusion from Scratch",
        "protocol_name": "DR05-FINAL",
        "protocol_version": "v1",
        "status": "frozen",
    }
    assert protocol["upstream"]["dr04_freeze_manifest"]["sha256"] == "c849a50ae342803707c20090d8ed5cd1383ceac9854d2e6b035a2519ee6b4af6"
    assert protocol["checkpoints"]["conditional"]["canonical_path"] == "artifacts/checkpoints/conditional_main_ema_step20000.pt"
    assert protocol["checkpoints"]["conditional"]["sha256"] == "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"
    assert protocol["checkpoints"]["dr04_evaluator"]["canonical_path"] == "artifacts/checkpoints/dr04_evaluator_epoch41.pt"
    assert protocol["checkpoints"]["dr04_evaluator"]["sha256"] == "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"
    assert protocol["dependencies"] == {"installation_extra": "eval", "packages": {"clean-fid": "0.1.35", "lpips": "0.1.4"}}


def test_quality_diversity_and_efficiency_invariants():
    protocol = load()
    quality = protocol["quality"]
    assert quality["sample_budget"]["main_per_model"] == 10000
    assert quality["sample_budget"]["extended_maximum_per_model"] == 50000
    assert quality["quality_rng"]["canonical_generation_batch_size"] == 16
    assert quality["quality_rng"]["quality_generation_seed_base"] == 502026
    assert quality["operational_kid"]["kid_subset_seed"] == 2026
    diversity = protocol["diversity"]
    assert diversity["configuration"] == {"backbone": "AlexNet", "input_range": "[-1,1]", "input_resolution": "native 32x32", "lpips_version": "0.1", "metric": "LPIPS", "package": "lpips", "package_version": "0.1.4"}
    assert diversity["panel"]["prompts"] == 252
    assert diversity["panel"]["id_prompts"] == 152
    assert diversity["panel"]["ood_prompts"] == 100
    assert diversity["panel"]["ood_groups"] == {"OOD1": 25, "OOD2": 25, "OOD3": 25, "OOD4": 25}
    assert diversity["panel"]["generation_seeds"] == [2026, 2027, 2028, 2029, 2030]
    assert diversity["panel"]["total_generated_images"] == 1260
    assert diversity["panel"]["pairwise_lpips_evaluations"] == 2520
    assert diversity["uncertainty"]["bootstrap_resamples"] == 10000
    assert diversity["uncertainty"]["seed"] == 202605
    assert protocol["efficiency"]["parameter_counts"]["conditional"]["total"] == 4723907
    assert protocol["efficiency"]["parameter_counts"]["unconditional"] == 4595779
    assert protocol["efficiency"]["parameter_counts"]["dr04_evaluator"] == 293064
    assert protocol["efficiency"]["latency"]["warmup_runs"] == 3
    assert protocol["efficiency"]["latency"]["measured_runs"] == 10
    assert protocol["efficiency"]["memory"]["primary_metric"] == "peak CUDA allocated memory"


def test_pending_sealing_rules_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = sha256()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["optional-dependencies"]["eval"] == [
        "clean-fid==0.1.35",
        "lpips==0.1.4",
        "scipy==1.15.3",
    ]
    materializer = load_materializer()
    monkeypatch.setattr(materializer, "verify_dr04_closure_ancestry", lambda: None)
    output, digest = materializer.materialize(tmp_path / "dr05_final_protocol.json")
    assert output.read_bytes() == PROTOCOL.read_bytes()
    assert digest == first
    protocol = load()
    assert protocol["dependencies"] == {
        "installation_extra": "eval",
        "packages": {"clean-fid": "0.1.35", "lpips": "0.1.4"},
    }
    assert "scipy" not in protocol["dependencies"]["packages"]
    assert protocol["unconditional_baseline"]["required"] is True
    assert protocol["unconditional_baseline"]["checkpoint_status"] == "pending"
    assert all(binding["status"] == "pending" for binding in protocol["execution_bindings"].values())
    assert protocol["sealed_data_policy"]["id_test_image_content"] == "sealed"
    assert protocol["sealed_data_policy"]["final_ood_image_content"] == "sealed"
    assert "no metric substitution after results" in protocol["anti_post_hoc_rules"]["forbidden"]
    assert "no prompt cherry-picking" in protocol["anti_post_hoc_rules"]["forbidden"]
    assert "no seed cherry-picking" in protocol["anti_post_hoc_rules"]["forbidden"]
    assert not any(Path(value).is_absolute() for value in strings(protocol))


def _temporary_git_repository(path: Path, message: str) -> str:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "DR05 protocol test"], cwd=path, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "dr05-protocol-test@invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", message], cwd=path, check=True
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def test_dr04_closure_commit_must_be_an_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    descendant_root = tmp_path / "descendant"
    closure_commit = _temporary_git_repository(descendant_root, "closure")
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "later"],
        cwd=descendant_root,
        check=True,
    )
    descendant = load_materializer()
    monkeypatch.setattr(descendant, "ROOT", descendant_root)
    monkeypatch.setattr(descendant, "DR04_CLOSURE_COMMIT", closure_commit)
    descendant.verify_dr04_closure_ancestry()

    unrelated_root = tmp_path / "unrelated"
    _temporary_git_repository(unrelated_root, "unrelated")
    unrelated = load_materializer()
    monkeypatch.setattr(unrelated, "ROOT", unrelated_root)
    monkeypatch.setattr(unrelated, "DR04_CLOSURE_COMMIT", closure_commit)
    with pytest.raises(ValueError, match="not an ancestor"):
        unrelated.verify_dr04_closure_ancestry()


def test_environment_artifact_records_frozen_provenance():
    assert ENVIRONMENT.is_file()
    environment = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    assert environment["python"] == "3.12.4"
    assert environment["pytorch"] == "2.5.1+cu121"
    assert environment["torch_cuda"] == "12.1"
    assert environment["numpy"] == "2.4.6"
    assert environment["scipy"] == "1.15.3"
    assert environment["clean_fid"] == "0.1.35"
    assert environment["lpips"] == "0.1.4"
    assert environment["gpu"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert environment["driver"] == "595.79"
    assert environment["scientific_results_recomputed"] is False
    assert environment["dependencies"] == {
        "full_evaluation_environment": [
            "clean-fid==0.1.35",
            "lpips==0.1.4",
            "scipy==1.15.3",
        ],
        "metric_specific_protocol_dependencies": ["clean-fid==0.1.35", "lpips==0.1.4"],
    }
    assert "scipy" not in environment["dependencies"]["metric_specific_protocol_dependencies"]
