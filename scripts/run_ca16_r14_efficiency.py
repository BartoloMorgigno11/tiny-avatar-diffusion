"""Guarded R14 efficiency entry point; preflight does not benchmark CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_avatar_diffusion.ca16_r14_orchestration import (
    ROOT,
    _run_efficiency_worker,
    efficiency_execution_order,
    execute_efficiency,
    load_protocol,
    resolve_inputs,
    validate_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-eval-sha", required=True)
    parser.add_argument("--expected-eval-tag")
    parser.add_argument("--confirm-r14-efficiency", action="store_true")
    parser.add_argument("--worker-ordinal", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = load_protocol()
    output = args.output_root or ROOT / str(protocol["output_root"]) / "efficiency"
    if args.worker_ordinal is not None:
        if args.output_root is None or args.preflight or args.execute:
            parser.error("worker mode requires only --worker-ordinal and --output-root")
        result = _run_efficiency_worker(
            output_root=output,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag or "",
            ordinal=args.worker_ordinal,
        )
    elif args.execute:
        if not args.expected_eval_tag or not args.confirm_r14_efficiency:
            parser.error(
                "--execute requires an expected tag and --confirm-r14-efficiency"
            )
        result = execute_efficiency(
            output_root=output,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
        )
    elif args.preflight:
        result = validate_preflight(
            output_root=output,
            expected_eval_sha=args.expected_eval_sha,
            expected_tag=args.expected_eval_tag,
            allow_existing=True,
        )
        _plan, checkpoints = resolve_inputs(protocol)
        result["efficiency_execution_order"] = efficiency_execution_order(checkpoints)
    else:
        parser.error("choose --preflight or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
