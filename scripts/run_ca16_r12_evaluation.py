"""Guarded R12 held-out evaluation for the six R11 V2 selected checkpoints."""

from __future__ import annotations

import argparse
import json

from tiny_avatar_diffusion.ca16_experiment import ROOT
from tiny_avatar_diffusion.ca16_r12_orchestration import (
    execute_r12,
    load_protocol,
    validate_preflight,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-eval-sha", required=True)
    parser.add_argument("--expected-eval-tag")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="hash-bind inputs only; never open sealed prompt records",
    )
    mode.add_argument(
        "--execute", action="store_true", help="run the frozen held-out evaluation"
    )
    parser.add_argument("--confirm-r12-heldout-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight and args.confirm_r12_heldout_evaluation:
        parser.error("execution confirmation is invalid in preflight mode")
    if args.execute and not args.confirm_r12_heldout_evaluation:
        parser.error("--execute requires --confirm-r12-heldout-evaluation")
    if args.execute and args.expected_eval_tag is None:
        parser.error("--execute requires --expected-eval-tag")
    protocol = load_protocol()
    output_root = ROOT / protocol["output_root"]
    result = (
        validate_preflight(
            output_root=output_root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
        if args.preflight
        else execute_r12(
            output_root=output_root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
