"""Official guarded entry point for CA16 R11 checkpoint selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_avatar_diffusion.ca16_r11_orchestration import execute_r11, validate_preflight


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-eval-sha", required=True)
    parser.add_argument("--expected-eval-tag")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="validate only; perform no scientific work",
    )
    mode.add_argument(
        "--execute", action="store_true", help="run the guarded scientific R11 workflow"
    )
    parser.add_argument("--confirm-r11-scientific-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight and args.confirm_r11_scientific_execution:
        parser.error("execution confirmation is invalid in preflight mode")
    if args.execute and not args.confirm_r11_scientific_execution:
        parser.error("--execute requires --confirm-r11-scientific-execution")
    common = {
        "output_root": args.output_dir.resolve(),
        "expected_eval_sha": args.expected_eval_sha,
        "expected_tag": args.expected_eval_tag,
    }
    result = validate_preflight(**common) if args.preflight else execute_r11(**common)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
