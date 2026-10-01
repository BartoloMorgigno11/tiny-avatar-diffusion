"""Build deterministic, metadata-only DR04 evaluator splits from Cartoon Set 100k."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tiny_avatar_diffusion.dr04_support_audit import (
    MetadataRow,
    PROCESSED_ROOT,
    SELECTED_ATTRIBUTES,
    load_100k_manifests,
    matching_frozen_ood,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "dr04_evaluator_v1"
SPLIT_SEED = 2026
OOD_NAMES = ("OOD1", "OOD2", "OOD3", "OOD4")
NAMESPACES = {
    "ood_ceiling": "dr04-evaluator/ood-ceiling",
    "control_quota": "dr04-evaluator/control-quota-remainder",
    "control_assignment": "dr04-evaluator/control-candidate-assignment",
    "eval_val_exclusive": "dr04-evaluator/eval-val-exclusive-ood",
    "eval_val_natural": "dr04-evaluator/eval-val-natural-remainder",
}


def canonical_hash(seed: int, namespace: str, image_id: str, **fields: object) -> str:
    """SHA-256 of ``seed=<n>|namespace=<n>|image_id=<id>|key=value...``.

    Extra fields are sorted by key.  Synthetic IDs (for quota and slot ranking)
    are explicit canonical identifiers, never iteration indices.
    """
    parts = [f"seed={seed}", f"namespace={namespace}", f"image_id={image_id}"]
    parts.extend(f"{key}={fields[key]}" for key in sorted(fields))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _exclusive_ood(row: MetadataRow, definitions: list[dict[str, Any]]) -> str | None:
    matches = matching_frozen_ood(row, definitions)
    return matches[0] if len(matches) == 1 else None


def _globally_ordinary(row: MetadataRow, definitions: list[dict[str, Any]]) -> bool:
    return not matching_frozen_ood(row, definitions)


def _ranked_ids(rows: list[MetadataRow], seed: int, namespace: str, **fields: object) -> list[str]:
    return [
        row.image_id
        for row in sorted(rows, key=lambda row: (canonical_hash(seed, namespace, row.image_id, **fields), row.image_id))
    ]


def _quota_by_replacement(
    replacement_values: list[int], source_ood: str, direction: str, seed: int
) -> dict[int, int]:
    if not replacement_values:
        raise ValueError(f"{source_ood}/{direction}: no eligible replacement values")
    base, remainder = divmod(50, len(replacement_values))
    ranked = sorted(
        replacement_values,
        key=lambda value: canonical_hash(
            seed, NAMESPACES["control_quota"], f"quota:{source_ood}:{direction}:{value}",
            source_ood=source_ood, direction=direction, replacement_value=value,
        ),
    )
    return {value: base + int(value in ranked[:remainder]) for value in replacement_values}


def _allocate_unique_slots(slots: dict[str, list[str]], seed: int) -> dict[str, str]:
    """Deterministic augmenting-path matching for exact, globally unique slots."""
    candidates = {
        slot: sorted(
            ids,
            key=lambda image_id: (
                canonical_hash(seed, NAMESPACES["control_assignment"], image_id, slot=slot), image_id,
            ),
        )
        for slot, ids in slots.items()
    }
    slot_order = sorted(
        slots,
        key=lambda slot: (canonical_hash(seed, NAMESPACES["control_assignment"], f"slot:{slot}", slot=slot), slot),
    )
    image_to_slot: dict[str, str] = {}
    slot_to_image: dict[str, str] = {}

    def assign(slot: str, visited_images: set[str]) -> bool:
        for image_id in candidates[slot]:
            if image_id in visited_images:
                continue
            visited_images.add(image_id)
            occupied = image_to_slot.get(image_id)
            if occupied is None or assign(occupied, visited_images):
                image_to_slot[image_id] = slot
                slot_to_image[slot] = image_id
                return True
        return False

    for slot in slot_order:
        if not assign(slot, set()):
            raise ValueError(f"Infeasible globally unique matched-control allocation at slot {slot}")
    if len(slot_to_image) != len(slots):
        raise AssertionError("Matched-control allocation did not fill every exact quota slot")
    return slot_to_image


def build_assignments(rows: list[MetadataRow], definitions: list[dict[str, Any]], seed: int = SPLIT_SEED) -> list[dict[str, Any]]:
    """Build all four evaluator splits; input order is deliberately irrelevant."""
    if len(rows) != 100_000 or len({row.image_id for row in rows}) != len(rows):
        raise ValueError("Expected exactly 100000 unique 100k metadata rows")
    definition_by_name = {definition["name"]: definition for definition in definitions}
    if tuple(definition_by_name) != OOD_NAMES:
        raise ValueError("Expected the four canonical frozen FINAL_OOD definitions")
    by_id = {row.image_id: row for row in rows}
    assignments: dict[str, dict[str, Any]] = {}

    def add(image_id: str, split: str, **fields: Any) -> None:
        if image_id in assignments:
            raise AssertionError(f"Duplicate assignment for {image_id}")
        row = by_id[image_id]
        assignments[image_id] = {
            "image_id": image_id, "split": split,
            **{attribute: row.values[attribute] for attribute in SELECTED_ATTRIBUTES},
            "exclusive_ood_id": _exclusive_ood(row, definitions) or "none",
            "source_ood": fields.get("source_ood", "none"),
            "control_direction": fields.get("control_direction", "none"),
            "replacement_value": fields.get("replacement_value", ""),
            "selection_stratum": fields.get("selection_stratum", "natural"),
        }

    # 1. Exclusive OOD ceiling.
    for name in OOD_NAMES:
        pool = [row for row in rows if _exclusive_ood(row, definitions) == name]
        selected = _ranked_ids(pool, seed, NAMESPACES["ood_ceiling"], source_ood=name)[:100]
        if len(selected) != 100:
            raise ValueError(f"{name}: insufficient exclusive support for OOD ceiling")
        for image_id in selected:
            add(image_id, "REAL_GENERATOR_OOD_CEILING", source_ood=name, selection_stratum="exclusive_ood_ceiling")

    # 2. Globally ordinary, quota-balanced matched ID controls.
    slots: dict[str, list[str]] = {}
    slot_metadata: dict[str, tuple[str, str, int]] = {}
    for name in OOD_NAMES:
        definition = definition_by_name[name]
        a, a_value = definition["attribute_a"], definition["value_a"]
        b, b_value = definition["attribute_b"], definition["value_b"]
        for direction, changed, fixed, fixed_value, original in (
            ("A_FIXED", b, a, a_value, b_value),
            ("B_FIXED", a, b, b_value, a_value),
        ):
            pools: dict[int, list[MetadataRow]] = {}
            for row in rows:
                if _globally_ordinary(row, definitions) and row.values[fixed] == fixed_value and row.values[changed] != original:
                    pools.setdefault(row.values[changed], []).append(row)
            quotas = _quota_by_replacement(sorted(pools), name, direction, seed)
            for replacement, quota in quotas.items():
                if len(pools[replacement]) < quota:
                    raise ValueError(f"{name}/{direction}/{replacement}: {len(pools[replacement])} candidates for quota {quota}")
                for replica in range(quota):
                    slot = f"{name}|{direction}|replacement={replacement}|replica={replica}"
                    slots[slot] = [row.image_id for row in pools[replacement]]
                    slot_metadata[slot] = (name, direction, replacement)
    slot_to_image = _allocate_unique_slots(slots, seed)
    for slot, image_id in slot_to_image.items():
        source_ood, direction, replacement = slot_metadata[slot]
        add(image_id, "REAL_ID_CEILING", source_ood=source_ood, control_direction=direction,
            replacement_value=replacement, selection_stratum="globally_ordinary_matched_control")

    # 3. Validation: exclusive quota first, then natural remainder.
    for name in OOD_NAMES:
        pool = [row for row in rows if row.image_id not in assignments and _exclusive_ood(row, definitions) == name]
        selected = _ranked_ids(pool, seed, NAMESPACES["eval_val_exclusive"], source_ood=name)[:50]
        if len(selected) != 50:
            raise ValueError(f"{name}: insufficient remaining exclusive support for EVAL_VAL")
        for image_id in selected:
            add(image_id, "EVAL_VAL", source_ood=name, selection_stratum="exclusive_ood_validation_quota")
    remaining = [row for row in rows if row.image_id not in assignments]
    natural = _ranked_ids(remaining, seed, NAMESPACES["eval_val_natural"])[:4800]
    if len(natural) != 4800:
        raise ValueError("Insufficient remaining pool for EVAL_VAL natural remainder")
    for image_id in natural:
        add(image_id, "EVAL_VAL", selection_stratum="natural_validation_remainder")

    for row in rows:
        if row.image_id not in assignments:
            add(row.image_id, "EVAL_TRAIN")
    return [assignments[image_id] for image_id in sorted(assignments)]


def summarize(assignments: list[dict[str, Any]], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in assignments)
    id_rows = [row for row in assignments if row["split"] == "REAL_ID_CEILING"]
    id_counts = Counter((row["source_ood"], row["control_direction"], str(row["replacement_value"])) for row in id_rows)
    multi = Counter()
    for row in assignments:
        metadata = MetadataRow(row["image_id"], {attribute: row[attribute] for attribute in SELECTED_ATTRIBUTES})
        if len(matching_frozen_ood(metadata, definitions)) > 1:
            multi[row["split"]] += 1
    return {
        "split_counts": dict(sorted(split_counts.items())),
        "id_ceiling_by_source_direction_replacement": [
            {"source_ood": source, "direction": direction, "replacement_value": value, "count": count}
            for (source, direction, value), count in sorted(id_counts.items())
        ],
        "ood_ceiling_by_source": dict(sorted(Counter(row["source_ood"] for row in assignments if row["split"] == "REAL_GENERATOR_OOD_CEILING").items())),
        "eval_val_exclusive_by_source": dict(sorted(Counter(row["source_ood"] for row in assignments if row["selection_stratum"] == "exclusive_ood_validation_quota").items())),
        "multi_ood_by_split": {split: multi[split] for split in ("REAL_GENERATOR_OOD_CEILING", "REAL_ID_CEILING", "EVAL_VAL", "EVAL_TRAIN")},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(assignments: list[dict[str, Any]], definitions: list[dict[str, Any]], output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "tiny_avatar_diffusion.dr04_evaluator_split_config", "version": "dr04_evaluator_v1",
        "seed": seed, "selected_attributes": list(SELECTED_ATTRIBUTES), "frozen_final_ood_definitions": definitions,
        "hash_serialization": "seed=<seed>|namespace=<namespace>|image_id=<canonical_id>|<sorted key=value fields>",
        "namespaces": NAMESPACES,
        "target_counts": {"REAL_GENERATOR_OOD_CEILING": 400, "REAL_ID_CEILING": 400, "EVAL_VAL": 5000, "EVAL_TRAIN": 94200},
    }
    config_path = output_dir / "split_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "split_manifest.csv"
    fields = list(assignments[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignments)
    summary = {
        "schema": "tiny_avatar_diffusion.dr04_evaluator_split_summary", "version": "dr04_evaluator_v1",
        "manifest_sha256": _sha256(manifest_path), "config_sha256": _sha256(config_path),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(), "config_path": config_path.relative_to(ROOT).as_posix(), **summarize(assignments, definitions),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DR04 Evaluator Split Build",
        "",
        "Metadata-only deterministic construction; no image bytes were read.",
        "",
        "## Split counts",
        "",
        *[f"- {split}: {count}" for split, count in summary["split_counts"].items()],
        "",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        f"- Config SHA-256: `{summary['config_sha256']}`",
        "",
        "## Hash canonicalization",
        "",
        f"`{config['hash_serialization']}`",
        "",
        "Namespaces: " + ", ".join(NAMESPACES.values()),
    ]
    (output_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=PROCESSED_ROOT / "100k")
    parser.add_argument("--heldout-path", type=Path, default=PROCESSED_ROOT / "heldout_combinations.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = json.loads(args.heldout_path.read_text(encoding="utf-8"))
    definitions = document["final_ood_definitions"]
    rows, errors = load_100k_manifests(args.manifest_dir)
    if errors:
        raise ValueError(f"Cannot build evaluator splits with malformed metadata: {errors[:5]}")
    summary = write_outputs(build_assignments(rows, definitions), definitions, args.output_dir, SPLIT_SEED)
    print(f"Wrote {len(rows)} deterministic evaluator assignments; manifest sha256={summary['manifest_sha256']}")


if __name__ == "__main__":
    main()
