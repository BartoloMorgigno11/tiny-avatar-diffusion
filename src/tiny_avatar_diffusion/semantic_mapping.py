"""Frozen DR01 semantic labels and model tokens."""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_ATTRIBUTE_ORDER = ("face_shape", "glasses", "hair_color", "face_color")

OFFICIAL = "OFFICIAL"
PROJECT_DEFINED_VISUALLY_CLEAR = "PROJECT_DEFINED_VISUALLY_CLEAR"
PROJECT_DEFINED_AMBIGUOUS = "PROJECT_DEFINED_AMBIGUOUS"
NEUTRAL_TOKEN = "NEUTRAL_TOKEN"


@dataclass(frozen=True)
class SemanticValue:
    raw_id: int
    canonical_label: str
    canonical_model_token: str
    provenance_status: str

    @property
    def is_semantic(self) -> bool:
        return self.provenance_status != NEUTRAL_TOKEN


_LABELS = {
    "face_shape": (
        "pointed",
        "face_shape_1",
        "oval",
        "long",
        "round",
        "wide",
        "rectangular",
    ),
    "glasses": (
        "oversized_glasses",
        "aviator_sunglasses",
        "cat_eye_glasses",
        "rimless_glasses",
        "rectangular_glasses",
        "round_glasses",
        "thick_rectangular_glasses",
        "oval_glasses",
        "narrow_sunglasses",
        "round_sunglasses",
        "heart_glasses",
        "no_glasses",
    ),
    "hair_color": (
        "light_blond",
        "golden_blond",
        "orange",
        "red",
        "golden_brown",
        "medium_brown",
        "dark_brown",
        "black",
        "blue_gray",
        "silver",
    ),
    "face_color": (
        "deep_brown",
        "dark_brown",
        "medium_brown",
        "warm_brown",
        "golden_tan",
        "light_tan",
        "rose_tan",
        "peach_tan",
        "light_beige",
        "light_peach",
        "pale_pink",
    ),
}

_AMBIGUOUS_FACE_COLORS = frozenset({3, 4, 6, 7, 9})


def _status(attribute: str, raw_id: int) -> str:
    if attribute == "glasses" and raw_id == 11:
        return OFFICIAL
    if attribute == "face_shape" and raw_id == 1:
        return NEUTRAL_TOKEN
    if attribute == "face_color" and raw_id in _AMBIGUOUS_FACE_COLORS:
        return PROJECT_DEFINED_AMBIGUOUS
    return PROJECT_DEFINED_VISUALLY_CLEAR


def _model_token(attribute: str, raw_id: int, label: str) -> str:
    if attribute == "face_shape" and raw_id == 1:
        return "face_shape_1"
    return f"{attribute}_{label}"


SEMANTIC_MAPPING = {
    attribute: {
        raw_id: SemanticValue(
            raw_id=raw_id,
            canonical_label=label,
            canonical_model_token=_model_token(attribute, raw_id, label),
            provenance_status=_status(attribute, raw_id),
        )
        for raw_id, label in enumerate(labels)
    }
    for attribute, labels in _LABELS.items()
}


def semantic_value(attribute: str, raw_id: int) -> SemanticValue:
    """Return the frozen mapping value, raising for unsupported IDs."""
    try:
        return SEMANTIC_MAPPING[attribute][raw_id]
    except KeyError as error:
        raise ValueError(f"Unsupported DR01 value: {attribute}={raw_id}") from error

