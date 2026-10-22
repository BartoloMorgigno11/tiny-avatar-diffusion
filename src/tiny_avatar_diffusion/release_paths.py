"""Locate release assets next to an extracted project, not in site-packages."""
from __future__ import annotations

from pathlib import Path


def release_root() -> Path:
    """Return the extracted release root when available.

    The wheel contains code and the public model manifest, while checkpoints and
    frozen assets intentionally remain beside the project.  Prefer the current
    directory (and its parents) so a non-editable installation works from an
    extracted ZIP.  Source-tree imports retain their normal repository fallback.
    """
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "artifacts" / "checkpoints").is_dir():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "artifacts" / "checkpoints").is_dir():
        return source_root
    return current
