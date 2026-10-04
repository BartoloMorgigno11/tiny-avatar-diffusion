"""Deterministically materialize the frozen DR05 final-evaluation protocol."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
DR04_DIR = ROOT / "data" / "processed" / "dr04_final_eval_v1"
FREEZE_MANIFEST = DR04_DIR / "dr04_freeze_manifest.json"
PANEL_MANIFEST = DR04_DIR / "final_prompt_panel_manifest.json"
SEED_MANIFEST = DR04_DIR / "seed_manifest.json"
CONDITIONAL_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt"
EVALUATOR_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "dr04_evaluator_epoch41.pt"

DR04_CLOSURE_COMMIT = "ca47fd2f46e5926a02f9d4ed6193554acaf743a9"
DR04_FREEZE_SHA256 = "c849a50ae342803707c20090d8ed5cd1383ceac9854d2e6b035a2519ee6b4af6"
CONDITIONAL_SHA256 = "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"
EVALUATOR_SHA256 = "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc"
FULL_EVAL_ENVIRONMENT = ["clean-fid==0.1.35", "lpips==0.1.4", "scipy==1.15.3"]
METRIC_SPECIFIC_PROTOCOL_DEPENDENCIES = {"clean-fid": "0.1.35", "lpips": "0.1.4"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def serialize(protocol: dict[str, Any]) -> bytes:
    return (json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {observed!r}")


def verify_dr04_closure_ancestry() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", DR04_CLOSURE_COMMIT, "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("DR04 closure commit is not an ancestor of current HEAD")


def verify_inputs() -> None:
    verify_dr04_closure_ancestry()
    for path, expected in (
        (FREEZE_MANIFEST, DR04_FREEZE_SHA256),
        (CONDITIONAL_CHECKPOINT, CONDITIONAL_SHA256),
        (EVALUATOR_CHECKPOINT, EVALUATOR_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        require_equal(f"SHA256 for {relative(path)}", sha256_file(path), expected)
    if not PANEL_MANIFEST.is_file() or not SEED_MANIFEST.is_file():
        raise FileNotFoundError("required DR04 panel or seed manifest is missing")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    require_equal(
        "eval dependency pins",
        pyproject["project"]["optional-dependencies"]["eval"],
        FULL_EVAL_ENVIRONMENT,
    )
    panel = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    seeds = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    require_equal("panel prompt count", panel["prompt_count"], 252)
    require_equal("panel ID prompt count", panel["id_prompt_count"], 152)
    require_equal("panel OOD prompt count", panel["ood_prompt_count"], 100)
    require_equal("panel generation seed count", panel["generation_seed_count"], 5)
    require_equal("panel generated image count", panel["expected_main_generation_count"], 1260)
    records = panel["prompt_records"]
    require_equal("panel record count", len(records), 252)
    require_equal("ID record count", sum(record["domain"] == "ID" for record in records), 152)
    ood = [record for record in records if record["domain"] == "OOD"]
    require_equal("OOD record count", len(ood), 100)
    for group in ("OOD1", "OOD2", "OOD3", "OOD4"):
        require_equal(f"{group} record count", sum(record["pair_id"] == group for record in ood), 25)
    require_equal("generation seeds", seeds["generation_seeds"], [2026, 2027, 2028, 2029, 2030])


def build_protocol() -> dict[str, Any]:
    return {
        "anti_post_hoc_rules": {
            "forbidden": [
                "no metric substitution after results",
                "no prompt cherry-picking",
                "no seed cherry-picking",
                "no removal of visually poor generations",
                "no checkpoint replacement after final outputs are inspected",
                "no generator modification to improve final evaluation",
                "no artificial MAIN ID/OOD balancing",
                "no minimal-edit contamination of diversity",
                "no combined quality/conditioning/diversity scalar",
                "no broad hyperparameter sweep",
                "no result-driven promotion from 10k MAIN to 50k EXT",
            ]
        },
        "checkpoints": {
            "conditional": {
                "canonical_path": relative(CONDITIONAL_CHECKPOINT),
                "expected_parameter_count": 4723907,
                "selected_step": 20000,
                "sha256": CONDITIONAL_SHA256,
                "source_checkpoint_sha256": "0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884",
                "weight_variant": "EMA",
            },
            "dr04_evaluator": {
                "canonical_path": relative(EVALUATOR_CHECKPOINT),
                "expected_parameter_count": 293064,
                "selected_epoch": 41,
                "sha256": EVALUATOR_SHA256,
                "source_checkpoint_sha256": "3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f",
            },
        },
        "dependencies": {
            "installation_extra": "eval",
            "packages": METRIC_SPECIFIC_PROTOCOL_DEPENDENCIES,
        },
        "diversity": {
            "configuration": {
                "backbone": "AlexNet",
                "input_range": "[-1,1]",
                "input_resolution": "native 32x32",
                "lpips_version": "0.1",
                "metric": "LPIPS",
                "package": "lpips",
                "package_version": "0.1.4",
            },
            "formula": "D_i = 2/[S(S-1)] * sum_{a<b} LPIPS(x_ia, x_ib)",
            "interpretation_invariants": [
                "higher LPIPS must NEVER automatically be interpreted as better",
                "diversity must be interpreted jointly with conditioning and quality",
            ],
            "panel": {
                "excluded": ["DR04 minimal-edit panel"],
                "generation_seeds": [2026, 2027, 2028, 2029, 2030],
                "id_prompts": 152,
                "ood_groups": {"OOD1": 25, "OOD2": 25, "OOD3": 25, "OOD4": 25},
                "ood_prompts": 100,
                "pairwise_lpips_evaluations": 2520,
                "primary_reporting": ["D_ID", "D_OOD"],
                "prompt_source": "exact canonical DR04 main panel",
                "prompts": 252,
                "reporting_secondary": ["global 252-prompt diversity", "matched-control diagnostic"],
                "samples_per_prompt": 5,
                "total_generated_images": 1260,
                "unordered_pairs_per_prompt": 10,
            },
            "primary_aggregation": "macro mean over prompt-level D_i",
            "sanity_debug_only": ["L1"],
            "uncertainty": {
                "bootstrap_resamples": 10000,
                "confidence_interval": "95% prompt-level bootstrap",
                "descriptive_dispersion": "standard deviation of D_i across prompts",
                "point_estimate": "macro mean of prompt-level D_i",
                "resampling_unit": "one prompt-level D_i together with its five generations",
                "seed": 202605,
                "split_reporting": ["ID", "OOD", "OOD1", "OOD2", "OOD3", "OOD4"],
                "forbidden": ["treating 2520 LPIPS pairwise distances as independent observations"],
            },
        },
        "efficiency": {
            "latency": {
                "benchmark_prompt": "pending execution binding from ID_VAL",
                "conditional_unconditional_reporting": ["conditional latency", "unconditional latency", "absolute overhead", "percentage overhead"],
                "configuration": {"batch_size": 1, "precision": "FP32", "resolution": "32x32", "sampler": "ancestral DDPM", "timesteps": 1000},
                "exclude": ["model loading", "PNG serialization", "disk I/O", "evaluator inference"],
                "headline": "median seconds/image",
                "include": ["prompt/tokenization", "text encoding exactly once", "full reverse diffusion", "final GPU tensor"],
                "measured_runs": 10,
                "secondary": ["mean", "std", "min", "max"],
                "synchronization": "CUDA synchronization before and after timed region",
                "throughput_secondary": {"batch_size": 16, "timing_philosophy": "same rigorous timing philosophy"},
                "warmup_runs": 3,
            },
            "memory": {
                "configuration": {"batch_size": 1, "precision": "FP32", "resolution": "32x32", "sampler": "ancestral DDPM", "timesteps": 1000},
                "headline": "peak CUDA allocated memory during batch-1 FP32 sampling, including model resident on GPU",
                "primary_metric": "peak CUDA allocated memory",
                "secondary": ["peak CUDA reserved", "pre-sampling allocated", "incremental sampling memory", "batch-16 peak sampling memory", "training peak as contextual only"],
            },
            "parameter_counts": {
                "anti_double_count_validation_required": True,
                "conditional": {"cross_attention": 25152, "text_encoder": 102976, "total": 4723907, "unet_backbone": 4595779},
                "dr04_evaluator": 293064,
                "source": "actually loaded canonical checkpoint architecture",
                "unconditional": 4595779,
            },
        },
        "execution_bindings": {
            "exact_quality_generated_id_ood_counts": {"status": "pending", "reason": "instantiate later from canonical sealed manifests"},
            "exact_r_id_count": {"status": "pending", "reason": "sealed manifest binding"},
            "exact_r_ood_count": {"status": "pending", "reason": "sealed manifest binding"},
            "final_hardware_environment_identity": {"status": "pending", "reason": "execution-time provenance"},
            "latency_benchmark_id_val_prompt": {"status": "pending", "reason": "bind exact prompt ID/text before benchmark execution"},
            "timing_rng_namespace_value": {"status": "pending", "reason": "dedicated namespace distinct from final-generation seeds"},
            "unconditional_checkpoint_path_hash": {"status": "pending", "reason": "canonical checkpoint does not yet exist"},
        },
        "protocol_identity": {
            "project": "Tiny Text-Conditioned Avatar Diffusion from Scratch",
            "protocol_name": "DR05-FINAL",
            "protocol_version": "v1",
            "status": "frozen",
        },
        "quality": {
            "conditional_reporting": {"GLOBAL": "secondary", "ID": "primary", "OOD": "primary"},
            "conceptual_invariant": "quality != conditioning",
            "generated_image_serialization": {
                "canonical_generated_transform": "model tensor [-1,1] -> clamp -> map to [0,1] -> multiply by 255 -> nearest integer -> uint8 RGB PNG",
                "canonical_real_transform": "RGB -> deterministic resize 32x32 -> bilinear interpolation -> antialias true -> uint8 RGB PNG",
                "forbidden": ["crop", "augmentation", "JPEG", "manual resize to 299x299", "double normalization"],
                "format": "lossless PNG",
                "representation": "32x32 RGB uint8",
                "sample_manifest_fields": ["sample_id", "model", "checkpoint", "split", "prompt", "seed", "shape", "mode", "dtype", "png_sha256", "canonical_pixel_sha256", "status"],
            },
            "implementation": {"feature_extractor": "Inception-v3", "mode": "clean", "package": "clean-fid", "version": "0.1.35"},
            "metric": {"primary": "KID", "secondary": "FID"},
            "operational_kid": {
                "kid_subset_seed": 2026,
                "kernel_statistic": "polynomial KID",
                "num_subsets": 100,
                "record_later": ["m", "N_real", "N_generated"],
                "subset_sampling": "without replacement",
                "subset_size": "min(1000, N_real, N_generated)",
                "warning": "subset dispersion must NOT automatically be labeled 95% CI",
            },
            "prompt_distribution": {
                "conditional_id": "empirical ID_TEST prompt distribution",
                "conditional_ood": "empirical FINAL_OOD prompt distribution",
                "global_conditional_pool": "follows R_GLOBAL proportions",
                "if_n_exceeds_references": "deterministic repetition preserving empirical proportions",
                "main_artificial_50_50_balancing": "forbidden",
            },
            "quality_rng": {
                "batch_seed": "502026 + batch_index",
                "canonical_generation_batch_size": 16,
                "extended_50k_batches": "through batch 3125",
                "kid_rng_namespace": 2026,
                "main_10k_batches": "first 625 batches",
                "quality_generation_seed_base": 502026,
                "same_sequence_for": ["conditional", "unconditional"],
                "technical_retry": "repeat entire failed batch using same seed only",
            },
            "real_references": {"R_GLOBAL": "exact union R_ID ∪ R_OOD", "R_ID": "full sealed ID_TEST", "R_OOD": "full sealed FINAL_OOD", "forbidden": "no TRAIN/VAL substitution"},
            "sample_budget": {
                "conditional_unconditional_same_n": True,
                "extended_maximum_per_model": 50000,
                "main_per_model": 10000,
                "nested_prefixes": [5000, 10000, 25000, 50000],
                "promotion_rule": "50k continuation is structural and pre-registered only, never result-driven",
            },
            "split_allocation": {"conditional_pool": "ID/OOD proportional to |R_ID| : |R_OOD| for every relevant prefix", "exact_counts": "pending sealed-manifest instantiation", "within_split": "empirical prompt distribution"},
            "global_comparison_requirements": ["same R_GLOBAL", "same generated N", "same resolution", "same preprocessing", "same metric implementation"],
        },
        "sealed_data_policy": {
            "allowed_before_authorization": ["metadata already permitted by upstream protocol", "ID_VAL", "COMP_VAL", "synthetic fixtures"],
            "final_ood_image_content": "sealed",
            "forbidden_before_authorization": ["decode ID_TEST", "render ID_TEST", "evaluate ID_TEST", "decode FINAL_OOD", "render FINAL_OOD", "evaluate FINAL_OOD"],
            "id_test_image_content": "sealed",
            "materializer_data_access": "metadata/protocol files only; no final image access",
        },
        "unconditional_baseline": {
            "checkpoint_status": "pending",
            "fairness_requirements": ["same 32x32 resolution", "same applicable U-Net backbone", "same diffusion schedule", "same training-image budget", "same optimizer family/logic", "same successful-step budget", "same precision policy", "same final quality sample budget"],
            "required": True,
        },
        "upstream": {
            "dr04_closure_commit": DR04_CLOSURE_COMMIT,
            "dr04_freeze_manifest": {"path": relative(FREEZE_MANIFEST), "sha256": DR04_FREEZE_SHA256},
            "dr04_main_prompt_panel": {"path": relative(PANEL_MANIFEST)},
            "dr04_seed_manifest": {"path": relative(SEED_MANIFEST)},
        },
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    identity = protocol["protocol_identity"]
    require_equal("protocol name", identity["protocol_name"], "DR05-FINAL")
    require_equal("protocol version", identity["protocol_version"], "v1")
    require_equal("protocol status", identity["status"], "frozen")
    require_equal("quality main budget", protocol["quality"]["sample_budget"]["main_per_model"], 10000)
    require_equal("quality extended budget", protocol["quality"]["sample_budget"]["extended_maximum_per_model"], 50000)
    require_equal("quality batch size", protocol["quality"]["quality_rng"]["canonical_generation_batch_size"], 16)
    require_equal("KID seed", protocol["quality"]["operational_kid"]["kid_subset_seed"], 2026)
    require_equal("diversity panel prompts", protocol["diversity"]["panel"]["prompts"], 252)
    require_equal("diversity generated images", protocol["diversity"]["panel"]["total_generated_images"], 1260)
    require_equal("diversity pairs", protocol["diversity"]["panel"]["pairwise_lpips_evaluations"], 2520)
    require_equal("bootstrap seed", protocol["diversity"]["uncertainty"]["seed"], 202605)
    if any(Path(value).is_absolute() for value in _strings(protocol)):
        raise ValueError("protocol contains an absolute path")


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return []


def materialize(output: Path = OUTPUT) -> tuple[Path, str]:
    verify_inputs()
    protocol = build_protocol()
    validate_protocol(protocol)
    content = serialize(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    validate_protocol(parsed)
    if serialize(parsed) != content:
        raise ValueError("protocol serialization is not deterministic")
    return output, sha256_file(output)


if __name__ == "__main__":
    artifact, digest = materialize()
    print(json.dumps({"path": relative(artifact), "sha256": digest}, sort_keys=True))
