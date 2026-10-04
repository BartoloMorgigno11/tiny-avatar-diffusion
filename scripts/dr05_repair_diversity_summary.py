"""Repair DR05 OOD-family summary fields from frozen prompt-level records."""
from __future__ import annotations

import argparse
from pathlib import Path

from tiny_avatar_diffusion.dr05_diversity import DEFAULT_OUTPUT_ROOT, repair_diversity_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair only DR05 OOD-family diversity summary fields.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    repair_diversity_summary(args.output_root)


if __name__ == "__main__":
    main()
