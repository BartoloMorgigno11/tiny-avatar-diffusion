"""Separately authorized entry point for the canonical DR05 diversity run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from tiny_avatar_diffusion.dr05_diversity import (DEFAULT_OUTPUT_ROOT, SOURCE_ROOT, aggregate_pairs, bootstrap_prompt_scopes, enumerate_pairs, lpips_alexnet_metric, load_jsonl, measure_pairs, reject_overlapping_output_root, source_artifact_hashes, validate_images, validate_population, write_results)

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
PROTOCOL_SHA256 = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
REQUIRED_ARTIFACTS = ("pair_records.jsonl", "prompt_records.jsonl", "diversity_summary.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _marker(name: str, **details: str) -> None:
    suffix = "".join(f" {key}={value}" for key, value in details.items())
    print(f"[DR05-DIVERSITY] {name}{suffix}", flush=True)


def _validate_protocol() -> str:
    observed = _sha256(PROTOCOL_PATH)
    if observed != PROTOCOL_SHA256:
        raise ValueError(f"frozen DR05 protocol SHA256 mismatch: {observed}")
    return observed


def _validate_written_artifacts(output_root: Path) -> None:
    if not output_root.is_dir():
        raise FileNotFoundError(f"canonical diversity output root is missing: {output_root}")
    for name in REQUIRED_ARTIFACTS:
        path = output_root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required diversity artifact is missing or empty: {path}")
        with path.open(encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                for line in handle:
                    if line.strip():
                        json.loads(line)
            else:
                json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the separately authorized DR05 LPIPS diversity evaluation.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    observed_protocol_sha256 = _sha256(PROTOCOL_PATH)
    stage = "START"
    _marker(
        stage,
        executable=sys.executable,
        script=str(Path(__file__).resolve()),
        cwd=str(Path.cwd()),
        script_sha256=_sha256(Path(__file__).resolve()),
        protocol=str(PROTOCOL_PATH),
        protocol_sha256=observed_protocol_sha256,
        output_root=str(args.output_root.resolve()),
    )
    try:
        protocol_sha256 = _validate_protocol()
        stage = "PROTOCOL_OK"
        _marker(stage, protocol_sha256=protocol_sha256)
        reject_overlapping_output_root(args.source_root, args.output_root)
        source_artifact_hashes(args.source_root)
        rows = load_jsonl(args.source_root / "manifests" / "generated_images.jsonl")
        groups = validate_population(rows)
        validate_images(rows, args.source_root)
        stage = "INPUT_OK"
        _marker(stage)
        pairs = enumerate_pairs(groups)
        metric = lpips_alexnet_metric(args.device)
        stage = "LPIPS_READY"
        _marker(stage)
        measured = measure_pairs(pairs, args.source_root, metric)
        stage = "PAIRS_COMPLETE"
        _marker(stage)
        prompts, aggregates = aggregate_pairs(measured)
        stage = "AGGREGATION_COMPLETE"
        _marker(stage)
        bootstrap = bootstrap_prompt_scopes(prompts)
        stage = "BOOTSTRAP_COMPLETE"
        _marker(stage)
        write_results(output_root=args.output_root, source_root=args.source_root, pair_rows=measured, prompt_rows=prompts, aggregates=aggregates, bootstrap=bootstrap)
        stage = "WRITE_COMPLETE"
        _marker(stage)
        _validate_written_artifacts(args.output_root)
        stage = "COMPLETE"
        _marker(stage)
    except Exception:
        _marker("FAILED", stage=stage)
        raise


if __name__ == "__main__":
    main()
