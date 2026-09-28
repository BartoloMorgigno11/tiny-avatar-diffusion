"""Canonical deterministic caption generation for DR01."""

from __future__ import annotations

from collections.abc import Mapping

from tiny_avatar_diffusion.semantic_mapping import (
    CANONICAL_ATTRIBUTE_ORDER,
    semantic_value,
)

CAPTION_PREFIX = "avatar"
CAPTION_LEXICAL_LENGTH = 5


def caption_tokens(metadata: Mapping[str, object]) -> list[str]:
    """Generate the five canonical lexical tokens from raw metadata IDs."""
    tokens = [CAPTION_PREFIX]
    for attribute in CANONICAL_ATTRIBUTE_ORDER:
        raw_id = metadata.get(attribute)
        if type(raw_id) is not int:
            raise ValueError(f"Expected integer metadata for {attribute}, got {raw_id!r}")
        tokens.append(semantic_value(attribute, raw_id).canonical_model_token)
    if len(tokens) != CAPTION_LEXICAL_LENGTH:
        raise AssertionError("Canonical DR01 caption length changed")
    return tokens


def generate_caption(metadata: Mapping[str, object]) -> str:
    """Generate the canonical lowercase, single-space caption."""
    caption = " ".join(caption_tokens(metadata))
    if caption != caption.lower() or "  " in caption:
        raise AssertionError("Canonical caption normalization failed")
    return caption

