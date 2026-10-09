"""Guarded secondary FID evaluation over immutable completed R13 artifacts."""

from __future__ import annotations

import argparse
import json

from tiny_avatar_diffusion.ca16_experiment import ROOT
from tiny_avatar_diffusion.ca16_r13_fid_secondary import (
    execute_fid,
    load_protocol,
    validate_preflight,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-eval-sha", required=True)
    parser.add_argument("--expected-eval-tag")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-r13-fid-secondary", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight and args.confirm_r13_fid_secondary:
        parser.error("execution confirmation is invalid in preflight mode")
    if args.execute and not args.confirm_r13_fid_secondary:
        parser.error("--execute requires --confirm-r13-fid-secondary")
    if args.execute and args.expected_eval_tag is None:
        parser.error("--execute requires --expected-eval-tag")
    output_root = ROOT / str(load_protocol()["output_root"])
    result = (
        validate_preflight(
            output_root=output_root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
        if args.preflight
        else execute_fid(
            output_root=output_root,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
