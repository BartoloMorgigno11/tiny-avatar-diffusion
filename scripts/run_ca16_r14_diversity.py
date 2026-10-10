"""Guarded R14 diversity entry point; preflight never samples or invokes LPIPS."""

from __future__ import annotations

import argparse
import json

from tiny_avatar_diffusion.ca16_r14_orchestration import (
    ROOT,
    execute_diversity,
    load_protocol,
    validate_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-eval-sha", required=True)
    parser.add_argument("--expected-eval-tag")
    parser.add_argument("--confirm-r14-diversity", action="store_true")
    args = parser.parse_args()
    root = ROOT / str(load_protocol()["output_root"]) / "diversity"
    if args.execute:
        if not args.expected_eval_tag or not args.confirm_r14_diversity:
            parser.error(
                "--execute requires an expected tag and --confirm-r14-diversity"
            )
        result = execute_diversity(
            output_root=root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
    else:
        result = validate_preflight(
            output_root=root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
