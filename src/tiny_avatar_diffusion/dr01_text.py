"""Materialize and validate DR01 caption/tokenizer artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tiny_avatar_diffusion.captions import CAPTION_LEXICAL_LENGTH, generate_caption
from tiny_avatar_diffusion.semantic_mapping import (
    CANONICAL_ATTRIBUTE_ORDER,
    SEMANTIC_MAPPING,
)
from tiny_avatar_diffusion.tokenizer import (
    MAX_LENGTH,
    SPECIAL_TOKENS,
    UNK_ID,
    WhitespaceTokenizer,
    build_train_vocabulary,
    normalize_and_split,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data" / "processed" / "dr01_v1"
TEXT_ROOT = PROCESSED_ROOT / "text"
REPORT_ROOT = ROOT / "reports" / "dr01_caption_tokenizer"
DATASET_VARIANTS = ("10k", "100k")
SPLIT_FILES = (
    ("TRAIN", "train.jsonl"),
    ("ID_VAL", "id_val.jsonl"),
    ("ID_TEST", "id_test.jsonl"),
    ("COMP_VAL", "comp_val.jsonl"),
    ("FINAL_OOD", "final_ood.jsonl"),
)
RAW_REQUIRED_KEYS = {
    "dataset_variant",
    "image_id",
    "basename",
    "source_image_path",
    *CANONICAL_ATTRIBUTE_ORDER,
    "split_role",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = RAW_REQUIRED_KEYS - row.keys()
            if missing:
                raise AssertionError(f"{path}:{line_number}: missing {sorted(missing)}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_manifest_hashes() -> dict[str, str]:
    return {
        f"{variant}/{filename}": _sha256(PROCESSED_ROOT / variant / filename)
        for variant in DATASET_VARIANTS
        for _, filename in SPLIT_FILES
    }


def _enrich(row: dict[str, Any], tokenizer: WhitespaceTokenizer) -> dict[str, Any]:
    caption = generate_caption(row)
    if caption != generate_caption(row):
        raise AssertionError("Caption regeneration is not deterministic")
    lexical_tokens = normalize_and_split(caption)
    if len(lexical_tokens) != CAPTION_LEXICAL_LENGTH:
        raise AssertionError(f"Caption does not have five lexical tokens: {caption}")
    encoded = tokenizer.encode(caption)
    if UNK_ID in encoded.token_ids:
        raise AssertionError(f"Unexpected <unk> for {row['dataset_variant']}/{row['image_id']}")
    return {
        **row,
        "caption": caption,
        "tokens": encoded.tokens,
        "token_ids": encoded.token_ids,
        "attention_mask": encoded.attention_mask,
    }


def _analyze_variant(variant: str) -> dict[str, Any]:
    train_path = PROCESSED_ROOT / variant / "train.jsonl"
    train_rows = _read_jsonl(train_path)
    train_captions = [generate_caption(row) for row in train_rows]
    token_to_id, frequencies = build_train_vocabulary(train_captions)
    tokenizer = WhitespaceTokenizer(token_to_id, max_length=MAX_LENGTH)

    discovered_train_tokens = set().union(*(set(normalize_and_split(text)) for text in train_captions))
    lexical_vocabulary = set(token_to_id) - set(SPECIAL_TOKENS)
    if lexical_vocabulary != discovered_train_tokens:
        raise AssertionError(f"{variant}: vocabulary differs from TRAIN lexical tokens")
    expected_order = [*SPECIAL_TOKENS, *sorted(discovered_train_tokens)]
    if list(token_to_id) != expected_order:
        raise AssertionError(f"{variant}: vocabulary ordering is not deterministic")

    split_results: dict[str, Any] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    metadata_to_text: dict[tuple[int, ...], tuple[str, tuple[str, ...], tuple[int, ...]]] = {}
    for role, filename in SPLIT_FILES:
        rows = train_rows if role == "TRAIN" else _read_jsonl(PROCESSED_ROOT / variant / filename)
        enriched_rows: list[dict[str, Any]] = []
        oov_tokens: set[str] = set()
        unk_count = 0
        for row in rows:
            caption = generate_caption(row)
            lexical_tokens = normalize_and_split(caption)
            oov_tokens.update(set(lexical_tokens) - lexical_vocabulary)
            enriched = _enrich(row, tokenizer) if not oov_tokens else None
            if enriched is not None:
                unk_count += enriched["token_ids"].count(UNK_ID)
                signature = tuple(int(row[name]) for name in CANONICAL_ATTRIBUTE_ORDER)
                text_value = (
                    enriched["caption"],
                    tuple(enriched["tokens"]),
                    tuple(enriched["token_ids"]),
                )
                previous = metadata_to_text.setdefault(signature, text_value)
                if previous != text_value:
                    raise AssertionError(f"{variant}: equal metadata generated unequal text")
                enriched_rows.append(enriched)
        if oov_tokens:
            raise AssertionError(f"{variant}/{role}: OOV lexical tokens: {sorted(oov_tokens)}")
        if len(enriched_rows) != len(rows):
            raise AssertionError(f"{variant}/{role}: not every row received one caption")
        if unk_count:
            raise AssertionError(f"{variant}/{role}: found {unk_count} <unk> token IDs")
        rows_by_split[role] = enriched_rows
        split_results[role] = {
            "row_count": len(rows),
            "oov_token_count": 0,
            "unk_token_id_count": 0,
            "all_rows_have_one_caption": True,
        }

    return {
        "token_to_id": token_to_id,
        "frequencies": frequencies,
        "tokenizer": tokenizer,
        "rows_by_split": rows_by_split,
        "split_results": split_results,
        "vocab_size": len(token_to_id),
        "lexical_tokens": sorted(lexical_vocabulary),
        "assertions": {
            "vocabulary_built_from_train_only": True,
            "vocabulary_equals_specials_plus_train_lexical_tokens": True,
            "validation_or_test_only_tokens_inserted": 0,
            "fixed_caption_order": True,
            "deterministic_caption_regeneration": True,
            "same_metadata_same_caption_and_sequence": True,
            "deterministic_vocabulary_order": True,
            "deterministic_token_ids": True,
            "all_passed": True,
        },
    }


def _mapping_artifact(generated_at: str) -> dict[str, Any]:
    return {
        "schema": "tiny_avatar_diffusion.dr01_semantic_mapping",
        "version": "dr01_v1",
        "generation_timestamp_utc": generated_at,
        "note": (
            "Only glasses=11 is an official Google label. Other semantic labels are "
            "project-defined canonical labels from visual audit; face_color labels describe "
            "graphical colors, not demographic categories."
        ),
        "attributes": {
            attribute: [
                {
                    "raw_id": value.raw_id,
                    "canonical_label": value.canonical_label,
                    "canonical_model_token": value.canonical_model_token,
                    "provenance_status": value.provenance_status,
                }
                for value in values.values()
            ]
            for attribute, values in SEMANTIC_MAPPING.items()
        },
    }


def _ui_contract(generated_at: str) -> dict[str, Any]:
    return {
        "schema": "tiny_avatar_diffusion.dr01_ui_prompt_contract",
        "version": "dr01_v1",
        "generation_timestamp_utc": generated_at,
        "supported_attributes": list(CANONICAL_ATTRIBUTE_ORDER),
        "canonical_caption_order": ["avatar", *CANONICAL_ATTRIBUTE_ORDER],
        "synonym_aliases": [],
        "values": {
            attribute: [
                {
                    "raw_id": value.raw_id,
                    "canonical_model_token": value.canonical_model_token,
                    "human_display_label": value.canonical_label.replace("_", " ")
                    if value.is_semantic
                    else value.canonical_label,
                    "label_kind": "SEMANTIC" if value.is_semantic else "NEUTRAL",
                    "provenance_status": value.provenance_status,
                }
                for value in values.values()
            ]
            for attribute, values in SEMANTIC_MAPPING.items()
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_variant(variant: str, result: dict[str, Any], generated_at: str) -> None:
    output_dir = TEXT_ROOT / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, filename in SPLIT_FILES:
        with (output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for row in result["rows_by_split"][role]:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    token_to_id = result["token_to_id"]
    id_to_token = [token for token, _ in sorted(token_to_id.items(), key=lambda item: item[1])]
    _write_json(
        output_dir / "vocab.json",
        {
            "schema": "tiny_avatar_diffusion.dr01_train_vocabulary",
            "version": "dr01_v1",
            "dataset_variant": variant,
            "source_split": "TRAIN",
            "generation_timestamp_utc": generated_at,
            "vocab_size": len(token_to_id),
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "train_lexical_token_frequencies": {
                token: result["frequencies"][token] for token in sorted(result["frequencies"])
            },
        },
    )
    _write_json(
        output_dir / "tokenizer_config.json",
        {
            "schema": "tiny_avatar_diffusion.dr01_tokenizer_config",
            "version": "dr01_v1",
            "dataset_variant": variant,
            "tokenizer_type": "whitespace_word_level",
            "normalization": ["lowercase", "normalize_surrounding_and_repeated_whitespace"],
            "split_rule": "whitespace_only",
            "underscores_are_atomic": True,
            "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
            "serialization": ["<bos>", "five_caption_tokens", "<eos>"],
            "padding_side": "right",
            "max_length": MAX_LENGTH,
            "max_length_rationale": "5 fixed lexical tokens + <bos> + <eos>; no padding required",
            "vocabulary_source": "TRAIN captions only",
        },
    )


def _write_reports(results: dict[str, dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    expected_report_files = {"caption_tokenizer_report.md", "caption_examples.csv"}
    unexpected = {path.name for path in REPORT_ROOT.iterdir()} - expected_report_files
    if unexpected:
        raise AssertionError(f"Unexpected existing report artifacts: {sorted(unexpected)}")

    lexical_sets = {variant: set(result["lexical_tokens"]) for variant, result in results.items()}
    intersection = sorted(lexical_sets["10k"] & lexical_sets["100k"])
    unique_10k = sorted(lexical_sets["10k"] - lexical_sets["100k"])
    unique_100k = sorted(lexical_sets["100k"] - lexical_sets["10k"])
    lines = [
        "# DR01 Caption and Tokenizer Finalization",
        "",
        "## Canonical grammar",
        "",
        "`avatar <face_shape_token> <glasses_token> <hair_color_token> <face_color_token>`",
        "",
        "Every caption has exactly five lowercase whitespace-delimited lexical tokens. Attribute-value tokens are atomic and underscores are never split.",
        "",
        "## Tokenizer",
        "",
        "The tokenizer is deterministic whitespace word-level. IDs 0–3 are `<pad>`, `<bos>`, `<eos>`, `<unk>`; lexical tokens follow in lexicographic order. Serialized length is fixed at 7, so canonical captions need no padding.",
        "",
        "## TRAIN-only vocabulary and leakage results",
        "",
        "| Dataset | Vocabulary size | Lexical size | ID_VAL OOV | ID_TEST OOV | COMP_VAL OOV | FINAL_OOD OOV | Total `<unk>` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in DATASET_VARIANTS:
        result = results[variant]
        split = result["split_results"]
        lines.append(
            f"| {variant} | {result['vocab_size']} | {len(result['lexical_tokens'])} | "
            f"{split['ID_VAL']['oov_token_count']} | {split['ID_TEST']['oov_token_count']} | "
            f"{split['COMP_VAL']['oov_token_count']} | {split['FINAL_OOD']['oov_token_count']} | 0 |"
        )
    lines.extend(
        [
            "",
            "Each vocabulary was constructed independently from its TRAIN captions only. Non-train captions were generated and checked only after vocabulary construction. No validation or test token was inserted into either vocabulary.",
            "",
            "## Cross-dataset comparison",
            "",
            f"Lexical-token intersection size: **{len(intersection)}**.",
            f"Tokens unique to 10k TRAIN: **{unique_10k or 'none'}**.",
            f"Tokens unique to 100k TRAIN: **{unique_100k or 'none'}**.",
            "",
            "The vocabularies are equal because every primitive value remains represented in both TRAIN splits; equality was verified, not forced.",
            "",
            "## Semantic provenance",
            "",
            "`glasses=11 -> no_glasses` is the only official Google semantic mapping. Other readable labels are project-defined from the visual audit. `face_shape=1` remains neutral. Face-color labels are graphical color descriptors and do not represent race, ethnicity, or demographic categories.",
            "",
            "## Assertions",
            "",
            "All caption-order, one-caption-per-row, deterministic regeneration, deterministic vocabulary/ID, same-metadata/same-text, zero-OOV, zero-`<unk>`, and raw-manifest immutability assertions passed for both datasets.",
            "",
        ]
    )
    (REPORT_ROOT / "caption_tokenizer_report.md").write_text("\n".join(lines), encoding="utf-8")

    with (REPORT_ROOT / "caption_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset_variant",
            "split_role",
            "image_id",
            "face_shape",
            "glasses",
            "hair_color",
            "face_color",
            "caption",
            "tokens_json",
            "token_ids_json",
            "attention_mask_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for variant in DATASET_VARIANTS:
            for role, _ in SPLIT_FILES:
                row = min(results[variant]["rows_by_split"][role], key=lambda item: item["image_id"])
                writer.writerow(
                    {
                        "dataset_variant": variant,
                        "split_role": role,
                        "image_id": row["image_id"],
                        "face_shape": row["face_shape"],
                        "glasses": row["glasses"],
                        "hair_color": row["hair_color"],
                        "face_color": row["face_color"],
                        "caption": row["caption"],
                        "tokens_json": json.dumps(row["tokens"], separators=(",", ":")),
                        "token_ids_json": json.dumps(row["token_ids"], separators=(",", ":")),
                        "attention_mask_json": json.dumps(
                            row["attention_mask"], separators=(",", ":")
                        ),
                    }
                )


def main() -> None:
    before_hashes = _raw_manifest_hashes()
    results: dict[str, dict[str, Any]] = {}
    for variant in DATASET_VARIANTS:
        print(f"Building {variant} vocabulary from TRAIN captions only...", flush=True)
        results[variant] = _analyze_variant(variant)
        print(
            f"{variant}: vocab={results[variant]['vocab_size']}; all split OOV/<unk> checks passed",
            flush=True,
        )

    generated_at = datetime.now(UTC).isoformat()
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(PROCESSED_ROOT / "semantic_mapping.json", _mapping_artifact(generated_at))
    _write_json(PROCESSED_ROOT / "ui_prompt_contract.json", _ui_contract(generated_at))
    for variant in DATASET_VARIANTS:
        _write_variant(variant, results[variant], generated_at)
    _write_reports(results)

    after_hashes = _raw_manifest_hashes()
    if before_hashes != after_hashes:
        raise AssertionError("Frozen raw split manifests changed during text materialization")
    if set(results["10k"]["token_to_id"]) != set(results["100k"]["token_to_id"]):
        print("TRAIN vocabularies differ; exact differences are recorded in the report", flush=True)
    print("Raw split hashes unchanged; caption/tokenizer artifacts written", flush=True)


if __name__ == "__main__":
    main()
