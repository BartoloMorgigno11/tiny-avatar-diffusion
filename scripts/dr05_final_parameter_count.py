"""Produce the canonical DR05-D11 parameter-count artifact on CPU."""
from __future__ import annotations

import json
from pathlib import Path

from tiny_avatar_diffusion.dr05_parameters import build_parameter_summary, write_parameter_summary

ROOT = Path(__file__).resolve().parents[1]


def run() -> Path:
    return write_parameter_summary(ROOT, build_parameter_summary(ROOT))


if __name__ == "__main__":
    print(json.dumps({"path": run().relative_to(ROOT).as_posix()}, indent=2))
