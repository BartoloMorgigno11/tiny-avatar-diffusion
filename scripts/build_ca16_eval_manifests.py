"""Materialize frozen R2C CA16 development evaluation manifests; never generates images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_avatar_diffusion.ca16_eval_manifests import ROOT, build


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed" / "ca16_eval_v1")
    args = parser.parse_args()
    result = build(args.output_root)
    print(json.dumps({"paths": {key: display_path(value) for key, value in result["paths"].items()}, "hashes": result["hashes"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
