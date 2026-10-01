"""Deterministic, metadata-only DR04 support audit for Cartoon Set 100k."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from itertools import combinations
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data" / "processed" / "dr01_v1"
DEFAULT_OUTPUT = ROOT / "results" / "audits" / "dr04_100k_support"
SELECTED_ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
MANIFEST_FILENAMES = ("train.jsonl", "id_val.jsonl", "id_test.jsonl", "comp_val.jsonl", "final_ood.jsonl")


@dataclass(frozen=True)
class MetadataRow:
    image_id: str
    values: dict[str, int]


def load_100k_manifests(manifest_dir: Path) -> tuple[list[MetadataRow], list[str]]:
    """Read DR01's structured 100k manifests without reading any image bytes."""
    rows: list[MetadataRow] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for filename in MANIFEST_FILENAMES:
        path = manifest_dir / filename
        if not path.is_file():
            errors.append(f"missing manifest: {path}")
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    raw = json.loads(line)
                    image_id = raw["image_id"]
                    if not isinstance(image_id, str) or not image_id:
                        raise ValueError("image_id must be a non-empty string")
                    values = {attribute: raw[attribute] for attribute in SELECTED_ATTRIBUTES}
                    if any(value is None or isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
                        raise ValueError("selected attributes must be non-null integers")
                except (json.JSONDecodeError, KeyError, ValueError) as error:
                    errors.append(f"{path}:{line_number}: {error}")
                    continue
                if image_id in seen_ids:
                    errors.append(f"{path}:{line_number}: duplicate image_id {image_id!r}")
                    continue
                seen_ids.add(image_id)
                rows.append(MetadataRow(image_id=image_id, values=values))
    return rows, errors


def marginal_support(rows: list[MetadataRow]) -> dict[str, Counter[int]]:
    return {attribute: Counter(row.values[attribute] for row in rows) for attribute in SELECTED_ATTRIBUTES}


def pair_count(rows: list[MetadataRow], attribute_a: str, value_a: int, attribute_b: str, value_b: int) -> int:
    return sum(row.values[attribute_a] == value_a and row.values[attribute_b] == value_b for row in rows)


def matching_frozen_ood(row: MetadataRow, frozen_definitions: list[dict[str, Any]]) -> list[str]:
    """Return every frozen FINAL_OOD definition matched by one metadata row."""
    return [
        definition["name"]
        for definition in frozen_definitions
        if row.values[definition["attribute_a"]] == definition["value_a"]
        and row.values[definition["attribute_b"]] == definition["value_b"]
    ]


def ood_overlap_audit(rows: list[MetadataRow], frozen_definitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full four-definition OOD membership partition from metadata."""
    names = [definition["name"] for definition in frozen_definitions]
    pattern_counts: Counter[str] = Counter()
    raw_support: Counter[str] = Counter()
    pairwise: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[int] = Counter()
    for row in rows:
        matches = matching_frozen_ood(row, frozen_definitions)
        match_set = set(matches)
        pattern = "".join("1" if name in match_set else "0" for name in names)
        pattern_counts[pattern] += 1
        category_counts[len(matches)] += 1
        raw_support.update(matches)
        pairwise.update(combinations(matches, 2))
    exclusive = {name: pattern_counts["".join("1" if current == name else "0" for current in names)] for name in names}
    overlap_rows = {name: raw_support[name] - exclusive[name] for name in names}
    return {
        "names": names, "raw_support": dict(raw_support), "exclusive_support": exclusive,
        "overlap_with_other": overlap_rows, "pattern_counts": dict(sorted(pattern_counts.items())),
        "pairwise_intersections": {f"{left}__{right}": pairwise[(left, right)] for left, right in combinations(names, 2)},
        "category_counts": {str(size): category_counts[size] for size in range(len(names) + 1)},
    }


def enumerate_matched_controls(
    rows: list[MetadataRow], definition: dict[str, Any], frozen_definitions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    attribute_a, value_a = definition["attribute_a"], definition["value_a"]
    attribute_b, value_b = definition["attribute_b"], definition["value_b"]
    observed_a = sorted({row.values[attribute_a] for row in rows})
    observed_b = sorted({row.values[attribute_b] for row in rows})
    candidates: list[dict[str, Any]] = []

    def build_candidate(changed_attribute: str, changed_from: int, changed_to: int,
                        unchanged_attribute: str, unchanged_value: int) -> dict[str, Any] | None:
        matched_rows = [
            row for row in rows
            if row.values[changed_attribute] == changed_to
            and row.values[unchanged_attribute] == unchanged_value
        ]
        if not matched_rows:
            return None
        exclusions = Counter(
            ood_name for row in matched_rows for ood_name in matching_frozen_ood(row, frozen_definitions)
        )
        excluded_count = sum(bool(matching_frozen_ood(row, frozen_definitions)) for row in matched_rows)
        return {
            "source_ood_pair": definition["name"], "changed_attribute": changed_attribute,
            "changed_from_value": changed_from, "changed_to_value": changed_to,
            "unchanged_attribute": unchanged_attribute, "unchanged_value": unchanged_value,
            "raw_candidate_support": len(matched_rows),
            "globally_ordinary_support": len(matched_rows) - excluded_count,
            "excluded_cross_ood_count": excluded_count,
            "cross_ood_exclusions_json": json.dumps(dict(sorted(exclusions.items())), separators=(",", ":")),
        }

    for replacement in observed_b:
        if replacement == value_b:
            continue
        candidate = build_candidate(attribute_b, value_b, replacement, attribute_a, value_a)
        if candidate is not None:
            candidates.append(candidate)
    for replacement in observed_a:
        if replacement == value_a:
            continue
        candidate = build_candidate(attribute_a, value_a, replacement, attribute_b, value_b)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(manifest_dir: Path, heldout_path: Path, output_dir: Path) -> dict[str, Any]:
    definitions_doc = json.loads(heldout_path.read_text(encoding="utf-8"))
    definitions = definitions_doc["final_ood_definitions"]
    if tuple(definitions_doc["selected_attributes"]) != SELECTED_ATTRIBUTES or len(definitions) != 4:
        raise ValueError("Frozen held-out configuration is not the expected DR01 four-pair definition")
    rows, errors = load_100k_manifests(manifest_dir)
    supports = marginal_support(rows)
    valid_count = len(rows)
    attribute_rows: list[dict[str, Any]] = []
    attribute_summary: dict[str, Any] = {}
    for attribute in SELECTED_ATTRIBUTES:
        counts = supports[attribute]
        values = sorted(counts.values())
        attribute_summary[attribute] = {
            "cardinality": len(counts), "observed_values": sorted(counts),
            "min_support": min(values) if values else None, "median_support": median(values) if values else None,
            "max_support": max(values) if values else None,
        }
        for value, count in sorted(counts.items()):
            attribute_rows.append({"attribute": attribute, "value": value, "count": count,
                                   "percent_of_valid": 100 * count / valid_count if valid_count else 0.0})
    pair_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for definition in definitions:
        a, va, b, vb = definition["attribute_a"], definition["value_a"], definition["attribute_b"], definition["value_b"]
        count = pair_count(rows, a, va, b, vb)
        pair_rows.append({"pair_id": definition["name"], "attribute_a": a, "value_a": va, "attribute_b": b,
                          "value_b": vb, "support_count": count,
                          "percent_of_valid": 100 * count / valid_count if valid_count else 0.0,
                          "marginal_support_a": supports[a][va], "marginal_support_b": supports[b][vb],
                          "constituents_represented": supports[a][va] > 0 and supports[b][vb] > 0})
        control_rows.extend(enumerate_matched_controls(rows, definition, definitions))
    overlap_audit = ood_overlap_audit(rows, definitions)
    exclusive_ood_rows = [
        {
            "ood": name, "raw_support": overlap_audit["raw_support"][name],
            "overlap_with_at_least_one_other": overlap_audit["overlap_with_other"][name],
            "exclusive_support": overlap_audit["exclusive_support"][name],
            "exclusive_percent_of_raw": 100 * overlap_audit["exclusive_support"][name] / overlap_audit["raw_support"][name],
            "exclusive_support_ge_100": overlap_audit["exclusive_support"][name] >= 100,
        }
        for name in overlap_audit["names"]
    ]
    overlap_matrix_rows = [
        {"ood_a": left, "ood_b": right, "intersection_count": overlap_audit["pairwise_intersections"][f"{left}__{right}"]}
        for left, right in combinations(overlap_audit["names"], 2)
    ]
    overlap_pattern_rows = [
        {"pattern_ood1_ood2_ood3_ood4": pattern, "match_count": count}
        for pattern, count in overlap_audit["pattern_counts"].items() if count
    ]
    for row in control_rows:
        row["raw_percent_of_valid"] = 100 * row["raw_candidate_support"] / valid_count if valid_count else 0.0
        row["globally_ordinary_percent_of_valid"] = 100 * row["globally_ordinary_support"] / valid_count if valid_count else 0.0
    control_summary_rows: list[dict[str, Any]] = []
    for definition in definitions:
        candidate_rows = [row for row in control_rows if row["source_ood_pair"] == definition["name"]]
        global_supports = [row["globally_ordinary_support"] for row in candidate_rows]
        control_summary_rows.append({
            "source_ood_pair": definition["name"], "candidate_combination_count": len(candidate_rows),
            "min_globally_ordinary_support": min(global_supports) if global_supports else None,
            "median_globally_ordinary_support": median(global_supports) if global_supports else None,
            "max_globally_ordinary_support": max(global_supports) if global_supports else None,
            "total_globally_ordinary_candidate_rows_before_deduplication": sum(global_supports),
            "at_least_one_candidate_ge_100": any(value >= 100 for value in global_supports),
            "at_least_two_distinct_candidates_ge_100": sum(value >= 100 for value in global_supports) >= 2,
        })
    cross_ood_summary_rows: list[dict[str, Any]] = []
    for definition in definitions:
        candidate_rows = [row for row in control_rows if row["source_ood_pair"] == definition["name"]]
        by_ood = Counter(
            ood_name
            for row in candidate_rows
            for ood_name, count in json.loads(row["cross_ood_exclusions_json"]).items()
            for _ in range(count)
        )
        cross_ood_summary_rows.append({
            "source_ood_pair": definition["name"],
            "excluded_candidate_rows_before_deduplication": sum(row["excluded_cross_ood_count"] for row in candidate_rows),
            "exclusions_by_frozen_ood_json": json.dumps(dict(sorted(by_ood.items())), separators=(",", ":")),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "attribute_support.csv", ["attribute", "value", "count", "percent_of_valid"], attribute_rows)
    _write_csv(output_dir / "final_ood_pair_support.csv", list(pair_rows[0]), pair_rows)
    _write_csv(output_dir / "matched_control_candidates.csv", list(control_rows[0]) if control_rows else [], control_rows)
    _write_csv(output_dir / "matched_control_global_summary.csv", list(control_summary_rows[0]), control_summary_rows)
    _write_csv(output_dir / "cross_ood_exclusion_summary.csv", list(cross_ood_summary_rows[0]), cross_ood_summary_rows)
    _write_csv(output_dir / "exclusive_ood_support.csv", list(exclusive_ood_rows[0]), exclusive_ood_rows)
    _write_csv(output_dir / "ood_overlap_matrix.csv", list(overlap_matrix_rows[0]), overlap_matrix_rows)
    _write_csv(output_dir / "ood_overlap_patterns.csv", list(overlap_pattern_rows[0]), overlap_pattern_rows)
    summary = {
        "schema": "tiny_avatar_diffusion.dr04_100k_support_audit", "version": "dr04_a1",
        "timestamp_utc": datetime.now(UTC).isoformat(), "randomness": "none",
        "git_head": _git_head(), "dataset_root": str(manifest_dir),
        "official_archive_path": str(ROOT / "data" / "raw" / "cartoonset" / "archives" / "cartoonset100k.tgz"),
        "source_paths": {"manifests": [str(manifest_dir / name) for name in MANIFEST_FILENAMES],
                         "heldout_definitions": str(heldout_path), "canonical_attribute_config": str(PROCESSED_ROOT / "semantic_mapping.json"),
                         "dr01_parser": str(ROOT / "src" / "tiny_avatar_diffusion" / "dr01_split.py")},
        "discovered_records": valid_count + len(errors), "valid_example_count": valid_count,
        "malformed_or_missing_record_count": len(errors), "malformed_or_missing_records": errors,
        "unique_image_identifiers": len({row.image_id for row in rows}), "attribute_summary": attribute_summary,
        "final_ood_pair_support": pair_rows,
        "matched_control_global_eligibility": control_summary_rows,
        "cross_ood_exclusion_summary": cross_ood_summary_rows,
        "ood_overlap_structure": {
            **overlap_audit, "exclusive_ood_support": exclusive_ood_rows,
            "d9_exclusive_ood_ceiling_feasibility": "PASS" if all(row["exclusive_support_ge_100"] for row in exclusive_ood_rows) else "FAIL",
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pair_counts = [row["support_count"] for row in pair_rows]
    lines = ["# DR04-A1: 100k Metadata Support Audit", "", "Metadata-only audit: no image bytes were decoded or inspected.", "",
             "## Integrity", "", f"- Valid examples: {valid_count}", f"- Malformed/missing records: {len(errors)}", f"- Unique image identifiers: {len({row.image_id for row in rows})}", "",
             "## Frozen FINAL_OOD support", "", "| Pair | Attributes | Support | Percent |", "|---|---|---:|---:|"]
    lines.extend(f"| {r['pair_id']} | {r['attribute_a']}={r['value_a']}, {r['attribute_b']}={r['value_b']} | {r['support_count']} | {r['percent_of_valid']:.4f}% |" for r in pair_rows)
    lines.extend(["", "## Marginal cardinalities", ""])
    lines.extend(f"- {attribute}: {info['cardinality']} observed values; support min/median/max = {info['min_support']}/{info['median_support']}/{info['max_support']}" for attribute, info in attribute_summary.items())
    lines.extend(["", "## Coverage", "", f"- Minimum FINAL_OOD support: {min(pair_counts)}", f"- Maximum FINAL_OOD support: {max(pair_counts)}", f"- Max/min ratio: {max(pair_counts) / min(pair_counts):.6f}" if min(pair_counts) else "- Max/min ratio: undefined (zero minimum support)", "",
                  "## Globally ordinary matched-control candidates", "", "| Source OOD | Candidates | Min | Median | Max | Total before dedup | >=100 | >=2 at >=100 |", "|---|---:|---:|---:|---:|---:|---|---|"])
    lines.extend(
        f"| {row['source_ood_pair']} | {row['candidate_combination_count']} | {row['min_globally_ordinary_support']} | {row['median_globally_ordinary_support']} | {row['max_globally_ordinary_support']} | {row['total_globally_ordinary_candidate_rows_before_deduplication']} | {row['at_least_one_candidate_ge_100']} | {row['at_least_two_distinct_candidates_ge_100']} |"
        for row in control_summary_rows
    )
    lines.extend(["", "## Cross-OOD exclusions", "", "| Source OOD | Excluded rows before dedup | Exclusions by frozen OOD |", "|---|---:|---|"])
    lines.extend(
        f"| {row['source_ood_pair']} | {row['excluded_candidate_rows_before_deduplication']} | `{row['exclusions_by_frozen_ood_json']}` |"
        for row in cross_ood_summary_rows
    )
    lines.extend(["", "## Exclusive OOD support", "", "| OOD | Raw | Overlap | Exclusive | Exclusive % | >=100 |", "|---|---:|---:|---:|---:|---|"])
    lines.extend(
        f"| {row['ood']} | {row['raw_support']} | {row['overlap_with_at_least_one_other']} | {row['exclusive_support']} | {row['exclusive_percent_of_raw']:.4f}% | {row['exclusive_support_ge_100']} |"
        for row in exclusive_ood_rows
    )
    categories = overlap_audit["category_counts"]
    lines.extend(["", "## OOD overlap partition", "", f"- Ordinary (zero OOD): {categories['0']}", f"- Exactly one OOD: {categories['1']}", f"- Exactly two OODs: {categories['2']}", f"- Exactly three OODs: {categories['3']}", f"- All four OODs: {categories['4']}", f"- Total: {sum(int(value) for value in categories.values())}",
                  f"- D9 exclusive-OOD ceiling feasibility: {summary['ood_overlap_structure']['d9_exclusive_ood_ceiling_feasibility']}", ""])
    lines.extend(["",
                  "Global eligibility is metadata-only: a candidate row is retained only when it matches none of OOD1--OOD4.", ""])
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=PROCESSED_ROOT / "100k")
    parser.add_argument("--heldout-path", type=Path, default=PROCESSED_ROOT / "heldout_combinations.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_audit(args.manifest_dir, args.heldout_path, args.output_dir)
    print(f"Audited {summary['valid_example_count']} valid 100k metadata records into {args.output_dir}")


if __name__ == "__main__":
    main()
