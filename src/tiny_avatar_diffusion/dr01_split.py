"""Materialize and validate the frozen DR01 compositional splits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "processed" / "dr01_v1"
REPORT_ROOT = ROOT / "reports" / "dr01_final_split"
SELECTED_ATTRIBUTES = ("face_shape", "glasses", "hair_color", "face_color")
SPLIT_SEED = 2026
ID_SPLIT_PROPORTIONS = {"TRAIN": 0.90, "ID_VAL": 0.05, "ID_TEST": 0.05}
PRECEDENCE = ("FINAL_OOD", "COMP_VAL", "ID_POOL")
DATASETS = {
    "10k": ROOT / "data" / "raw" / "cartoonset" / "archives" / "cartoonset10k.tgz",
    "100k": ROOT / "data" / "raw" / "cartoonset" / "archives" / "cartoonset100k.tgz",
}

COMP_VAL_RULES = (
    ("CV1", "face_shape", 0, "glasses", 0),
    ("CV2", "face_shape", 2, "hair_color", 3),
    ("CV3", "glasses", 7, "face_color", 2),
    ("CV4", "hair_color", 2, "face_color", 9),
)
FINAL_OOD_RULES = (
    ("OOD1", "face_shape", 2, "glasses", 3),
    ("OOD2", "face_shape", 4, "hair_color", 6),
    ("OOD3", "glasses", 10, "face_color", 10),
    ("OOD4", "hair_color", 0, "face_color", 0),
)
ALL_RULES = COMP_VAL_RULES + FINAL_OOD_RULES
FINAL_ROLE_FILES = {
    "TRAIN": "train.jsonl",
    "ID_VAL": "id_val.jsonl",
    "ID_TEST": "id_test.jsonl",
    "COMP_VAL": "comp_val.jsonl",
    "FINAL_OOD": "final_ood.jsonl",
}


@dataclass(frozen=True)
class SourceRow:
    dataset_variant: str
    image_id: str
    source_image_path: str
    values: dict[str, int]
    image_sha256: str


def _rule_dict(rule: tuple[str, str, int, str, int]) -> dict[str, object]:
    name, attribute_a, value_a, attribute_b, value_b = rule
    return {
        "name": name,
        "attribute_a": attribute_a,
        "value_a": value_a,
        "attribute_b": attribute_b,
        "value_b": value_b,
    }


def _matches(row: SourceRow, rule: tuple[str, str, int, str, int]) -> bool:
    _, attribute_a, value_a, attribute_b, value_b = rule
    return row.values[attribute_a] == value_a and row.values[attribute_b] == value_b


def _read_archive(dataset_variant: str, archive_path: Path) -> list[SourceRow]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing official source archive: {archive_path}")

    metadata: dict[str, dict[str, int]] = {}
    image_info: dict[str, tuple[str, str]] = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in {".csv", ".png"}:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            image_id = Path(member.name).stem
            if suffix == ".csv":
                raw = extracted.read().decode("utf-8-sig")
                values: dict[str, int] = {}
                for csv_row in csv.reader(io.StringIO(raw), skipinitialspace=True):
                    if len(csv_row) < 2:
                        continue
                    values[csv_row[0].strip()] = int(csv_row[1])
                missing = set(SELECTED_ATTRIBUTES) - values.keys()
                if missing:
                    raise AssertionError(
                        f"{dataset_variant}/{image_id}: missing metadata {sorted(missing)}"
                    )
                if image_id in metadata:
                    raise AssertionError(f"Duplicate metadata ID: {dataset_variant}/{image_id}")
                metadata[image_id] = {name: values[name] for name in SELECTED_ATTRIBUTES}
            else:
                digest = hashlib.sha256()
                while chunk := extracted.read(1024 * 1024):
                    digest.update(chunk)
                if image_id in image_info:
                    raise AssertionError(f"Duplicate image ID: {dataset_variant}/{image_id}")
                source_ref = f"{archive_path.relative_to(ROOT).as_posix()}::{member.name}"
                image_info[image_id] = (source_ref, digest.hexdigest())

    if metadata.keys() != image_info.keys():
        missing_images = sorted(metadata.keys() - image_info.keys())[:10]
        missing_metadata = sorted(image_info.keys() - metadata.keys())[:10]
        raise AssertionError(
            f"{dataset_variant}: image/metadata mismatch; "
            f"missing images={missing_images}, missing metadata={missing_metadata}"
        )
    rows = [
        SourceRow(
            dataset_variant=dataset_variant,
            image_id=image_id,
            source_image_path=image_info[image_id][0],
            values=metadata[image_id],
            image_sha256=image_info[image_id][1],
        )
        for image_id in sorted(metadata)
    ]
    expected = 10_000 if dataset_variant == "10k" else 100_000
    if len(rows) != expected:
        raise AssertionError(f"{dataset_variant}: expected {expected} sources, found {len(rows)}")
    return rows


def _marginals(rows: Iterable[SourceRow]) -> dict[str, dict[str, int]]:
    rows = list(rows)
    return {
        attribute: {
            str(value): count
            for value, count in sorted(Counter(row.values[attribute] for row in rows).items())
        }
        for attribute in SELECTED_ATTRIBUTES
    }


def _partner_diversity(rows: Iterable[SourceRow]) -> dict[str, object]:
    rows = list(rows)
    details: dict[str, dict[str, dict[str, int]]] = {}
    minimum: int | None = None
    minimum_locations: list[dict[str, object]] = []
    for source_attribute in SELECTED_ATTRIBUTES:
        details[source_attribute] = {}
        source_values = sorted({row.values[source_attribute] for row in rows})
        for source_value in source_values:
            details[source_attribute][str(source_value)] = {}
            selected = [row for row in rows if row.values[source_attribute] == source_value]
            for partner_attribute in SELECTED_ATTRIBUTES:
                if partner_attribute == source_attribute:
                    continue
                count = len({row.values[partner_attribute] for row in selected})
                details[source_attribute][str(source_value)][partner_attribute] = count
                location = {
                    "source_attribute": source_attribute,
                    "source_value": source_value,
                    "partner_attribute": partner_attribute,
                }
                if minimum is None or count < minimum:
                    minimum = count
                    minimum_locations = [location]
                elif count == minimum:
                    minimum_locations.append(location)
    return {
        "minimum": minimum,
        "minimum_locations": minimum_locations,
        "details": details,
    }


def _pair_cells(rows: Iterable[SourceRow]) -> dict[tuple[str, str], set[tuple[int, int]]]:
    rows = list(rows)
    result: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for index, attribute_a in enumerate(SELECTED_ATTRIBUTES):
        for attribute_b in SELECTED_ATTRIBUTES[index + 1 :]:
            result[(attribute_a, attribute_b)] = {
                (row.values[attribute_a], row.values[attribute_b]) for row in rows
            }
    return result


def _normalized_cell(
    attribute_a: str, value_a: int, attribute_b: str, value_b: int
) -> tuple[tuple[str, str], tuple[int, int]]:
    if SELECTED_ATTRIBUTES.index(attribute_a) < SELECTED_ATTRIBUTES.index(attribute_b):
        return (attribute_a, attribute_b), (value_a, value_b)
    return (attribute_b, attribute_a), (value_b, value_a)


def _simultaneous_audit(rows: list[SourceRow]) -> tuple[dict[str, str], dict[str, object]]:
    role_by_id: dict[str, str] = {}
    raw_pair_counts = {rule[0]: sum(_matches(row, rule) for row in rows) for rule in ALL_RULES}
    effective_pair_counts = {rule[0]: 0 for rule in ALL_RULES}
    multiple_final = 0
    multiple_comp = 0
    cross_role_overlap = 0

    for row in rows:
        final_matches = [rule for rule in FINAL_OOD_RULES if _matches(row, rule)]
        comp_matches = [rule for rule in COMP_VAL_RULES if _matches(row, rule)]
        multiple_final += len(final_matches) > 1
        multiple_comp += len(comp_matches) > 1
        cross_role_overlap += bool(final_matches and comp_matches)
        if final_matches:
            role_by_id[row.image_id] = "FINAL_OOD"
            for rule in final_matches:
                effective_pair_counts[rule[0]] += 1
        elif comp_matches:
            role_by_id[row.image_id] = "COMP_VAL"
            for rule in comp_matches:
                effective_pair_counts[rule[0]] += 1
        else:
            role_by_id[row.image_id] = "ID_POOL"

    id_pool = [row for row in rows if role_by_id[row.image_id] == "ID_POOL"]
    marginals = _marginals(id_pool)
    source_marginals = _marginals(rows)
    if any(set(marginals[attr]) != set(source_marginals[attr]) for attr in SELECTED_ATTRIBUTES):
        raise AssertionError("At least one selected attribute value disappeared after holdout removal")

    heldout_primitive_support: dict[str, dict[str, int]] = defaultdict(dict)
    for _, attribute_a, value_a, attribute_b, value_b in ALL_RULES:
        heldout_primitive_support[attribute_a][str(value_a)] = int(
            marginals[attribute_a][str(value_a)]
        )
        heldout_primitive_support[attribute_b][str(value_b)] = int(
            marginals[attribute_b][str(value_b)]
        )
    if any(count <= 0 for values in heldout_primitive_support.values() for count in values.values()):
        raise AssertionError("A primitive used by a held-out rule has no post-holdout support")

    partner_diversity = _partner_diversity(id_pool)
    if int(partner_diversity["minimum"] or 0) <= 1:
        raise AssertionError("Post-holdout alternate-partner diversity is not greater than one")

    source_cells = _pair_cells(rows)
    pool_cells = _pair_cells(id_pool)
    intended_cells: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for _, attribute_a, value_a, attribute_b, value_b in ALL_RULES:
        family, cell = _normalized_cell(attribute_a, value_a, attribute_b, value_b)
        intended_cells[family].add(cell)
    unintended_absent: list[dict[str, object]] = []
    for family, cells in source_cells.items():
        for value_a, value_b in sorted(cells - pool_cells[family] - intended_cells[family]):
            unintended_absent.append(
                {
                    "attribute_a": family[0],
                    "value_a": value_a,
                    "attribute_b": family[1],
                    "value_b": value_b,
                }
            )
    if unintended_absent:
        raise AssertionError(f"Unintended absent post-holdout cells: {unintended_absent}")

    role_counts = Counter(role_by_id.values())
    audit = {
        "source_count": len(rows),
        "role_counts_before_id_split": {
            "FINAL_OOD": role_counts["FINAL_OOD"],
            "COMP_VAL": role_counts["COMP_VAL"],
            "ID_POOL": role_counts["ID_POOL"],
        },
        "pair_counts_before_precedence": raw_pair_counts,
        "pair_counts_after_precedence": effective_pair_counts,
        "overlap_counts": {
            "multiple_final_ood_definitions": multiple_final,
            "multiple_comp_val_definitions": multiple_comp,
            "both_final_ood_and_comp_val": cross_role_overlap,
        },
        "post_holdout_marginals": marginals,
        "post_holdout_marginal_minimum": min(
            count for values in marginals.values() for count in values.values()
        ),
        "heldout_primitive_support": dict(heldout_primitive_support),
        "post_holdout_partner_diversity": partner_diversity,
        "unintended_absent_pair_count": len(unintended_absent),
        "structural_invariants_passed": True,
    }
    return role_by_id, audit


def _allocate_split_counts(total: int) -> dict[str, int]:
    exact = {role: total * proportion for role, proportion in ID_SPLIT_PROPORTIONS.items()}
    allocated = {role: int(value) for role, value in exact.items()}
    remainder = total - sum(allocated.values())
    role_order = list(ID_SPLIT_PROPORTIONS)
    ranked = sorted(role_order, key=lambda role: (-(exact[role] - allocated[role]), role_order.index(role)))
    for role in ranked[:remainder]:
        allocated[role] += 1
    return allocated


def _split_id_pool(
    rows: list[SourceRow], role_by_id: dict[str, str]
) -> tuple[dict[str, str], dict[str, object]]:
    ids = sorted(image_id for image_id, role in role_by_id.items() if role == "ID_POOL")
    random.Random(SPLIT_SEED).shuffle(ids)
    counts = _allocate_split_counts(len(ids))
    final_roles = dict(role_by_id)
    train_end = counts["TRAIN"]
    val_end = train_end + counts["ID_VAL"]
    for image_id in ids[:train_end]:
        final_roles[image_id] = "TRAIN"
    for image_id in ids[train_end:val_end]:
        final_roles[image_id] = "ID_VAL"
    for image_id in ids[val_end:]:
        final_roles[image_id] = "ID_TEST"

    pool_rows = [row for row in rows if role_by_id[row.image_id] == "ID_POOL"]
    pool_marginals = _marginals(pool_rows)
    distribution: dict[str, object] = {}
    for role in ("TRAIN", "ID_VAL", "ID_TEST"):
        role_rows = [row for row in rows if final_roles[row.image_id] == role]
        role_marginals = _marginals(role_rows)
        missing_values = {
            attr: sorted(set(pool_marginals[attr]) - set(role_marginals[attr]))
            for attr in SELECTED_ATTRIBUTES
            if set(pool_marginals[attr]) - set(role_marginals[attr])
        }
        if missing_values:
            raise AssertionError(f"Simple seeded {role} split lost values: {missing_values}")
        max_pp_delta = 0.0
        for attribute in SELECTED_ATTRIBUTES:
            for value, pool_count in pool_marginals[attribute].items():
                pool_rate = pool_count / len(pool_rows)
                role_rate = role_marginals[attribute][value] / len(role_rows)
                max_pp_delta = max(max_pp_delta, abs(role_rate - pool_rate) * 100)
        distribution[role] = {
            "count": len(role_rows),
            "all_selected_values_present": True,
            "maximum_absolute_marginal_percentage_point_delta_from_id_pool": max_pp_delta,
        }
    return final_roles, distribution


def _validate_final_partition(
    rows: list[SourceRow], final_roles: dict[str, str]
) -> dict[str, object]:
    expected_roles = set(FINAL_ROLE_FILES)
    if set(final_roles) != {row.image_id for row in rows}:
        raise AssertionError("Final role map does not cover every source image exactly once")
    if set(final_roles.values()) != expected_roles:
        raise AssertionError(f"Unexpected or missing final roles: {set(final_roles.values())}")

    ids_by_role = {
        role: {image_id for image_id, assigned in final_roles.items() if assigned == role}
        for role in expected_roles
    }
    for index, role_a in enumerate(FINAL_ROLE_FILES):
        for role_b in list(FINAL_ROLE_FILES)[index + 1 :]:
            if ids_by_role[role_a] & ids_by_role[role_b]:
                raise AssertionError(f"Partition overlap: {role_a} and {role_b}")
    if set().union(*ids_by_role.values()) != {row.image_id for row in rows}:
        raise AssertionError("Final role union differs from source image IDs")

    train_rows = [row for row in rows if final_roles[row.image_id] == "TRAIN"]
    for rule in ALL_RULES:
        if any(_matches(row, rule) for row in train_rows):
            raise AssertionError(f"Held-out rule leaked into TRAIN: {rule[0]}")
    train_marginals = _marginals(train_rows)
    for _, attribute_a, value_a, attribute_b, value_b in ALL_RULES:
        if train_marginals[attribute_a].get(str(value_a), 0) <= 0:
            raise AssertionError(f"Held-out primitive absent in TRAIN: {attribute_a}={value_a}")
        if train_marginals[attribute_b].get(str(value_b), 0) <= 0:
            raise AssertionError(f"Held-out primitive absent in TRAIN: {attribute_b}={value_b}")
    train_partner_diversity = _partner_diversity(train_rows)
    if int(train_partner_diversity["minimum"] or 0) <= 1:
        raise AssertionError("TRAIN alternate-partner diversity is not greater than one")

    hash_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_roles[row.image_sha256].add(final_roles[row.image_id])
    cross_role_hashes = {digest: roles for digest, roles in hash_roles.items() if len(roles) > 1}
    if cross_role_hashes:
        examples = list(cross_role_hashes.items())[:5]
        raise AssertionError(f"Exact image hashes cross final roles: {examples}")

    return {
        "partition_pairwise_disjoint": True,
        "partition_union_equals_source": True,
        "every_source_image_appears_once": True,
        "heldout_pairs_absent_from_train": True,
        "heldout_primitives_present_in_train": True,
        "train_partner_diversity_minimum": train_partner_diversity["minimum"],
        "exact_image_hashes_crossing_roles": 0,
        "all_assertions_passed": True,
    }


def _manifest_row(row: SourceRow, role: str) -> dict[str, object]:
    return {
        "dataset_variant": row.dataset_variant,
        "image_id": row.image_id,
        "basename": f"{row.image_id}.png",
        "source_image_path": row.source_image_path,
        "face_shape": row.values["face_shape"],
        "glasses": row.values["glasses"],
        "hair_color": row.values["hair_color"],
        "face_color": row.values["face_color"],
        "split_role": role,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _build_configuration(generated_at: str) -> dict[str, object]:
    return {
        "schema": "tiny_avatar_diffusion.dr01_heldout_combinations",
        "version": "dr01_v1",
        "selected_attributes": list(SELECTED_ATTRIBUTES),
        "comp_val_definitions": [_rule_dict(rule) for rule in COMP_VAL_RULES],
        "final_ood_definitions": [_rule_dict(rule) for rule in FINAL_OOD_RULES],
        "precedence": list(PRECEDENCE),
        "split_seed": SPLIT_SEED,
        "id_split_proportions": ID_SPLIT_PROPORTIONS,
        "id_split_count_allocation": "largest remainder; ties ordered TRAIN, ID_VAL, ID_TEST",
        "id_split_method": "sorted image IDs followed by Python random.Random(seed).shuffle, independently per dataset",
        "dataset_variants": {
            variant: path.relative_to(ROOT).as_posix() for variant, path in DATASETS.items()
        },
        "main_resolution": [32, 32],
        "preprocessing": "native RGBA -> alpha composite on RGB(255,255,255) -> RGB -> resize",
        "generation_timestamp_utc": generated_at,
        "provenance": {
            "generator": "src/tiny_avatar_diffusion/dr01_split.py",
            "generator_sha256": _code_sha256(),
            "git_commit": _git_commit(),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_outputs(
    rows_by_dataset: dict[str, list[SourceRow]],
    final_roles_by_dataset: dict[str, dict[str, str]],
    configuration: dict[str, object],
    summary: dict[str, object],
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "heldout_combinations.json", configuration)
    _write_json(OUTPUT_ROOT / "split_summary.json", summary)

    for dataset_variant, rows in rows_by_dataset.items():
        dataset_dir = OUTPUT_ROOT / dataset_variant
        dataset_dir.mkdir(parents=True, exist_ok=True)
        role_map = final_roles_by_dataset[dataset_variant]
        for role, filename in FINAL_ROLE_FILES.items():
            selected = sorted(
                (row for row in rows if role_map[row.image_id] == role),
                key=lambda row: row.image_id,
            )
            with (dataset_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
                for row in selected:
                    handle.write(json.dumps(_manifest_row(row, role), separators=(",", ":")) + "\n")

    _write_report(summary)
    _write_counts_csv(summary)


def _write_report(summary: dict[str, object]) -> None:
    datasets = summary["datasets"]
    lines = [
        "# DR01 Final Compositional Split",
        "",
        "## Protocol",
        "",
        "This split is compositional rather than random-only: all examples matching any of eight frozen value-pair rules are removed before the remaining ID pool is split. COMP_VAL measures development-time generalization to four unseen combinations, while FINAL_OOD seals four different unseen combinations for final evaluation. FINAL_OOD has assignment precedence so an image matching both panels cannot leak into development-time compositional validation.",
        "",
        "The exact same held-out value-pair definitions, precedence, 90/5/5 ID policy, and seed 2026 are reused independently for the official 10k and 100k datasets. Absolute counts intentionally differ.",
        "",
        "All selected attribute values and every primitive used by a held-out definition remain represented in TRAIN. Vocabulary leakage is not assessed here because caption and tokenizer construction belongs to the next step.",
        "",
        "## Final counts",
        "",
        "| Dataset | Source | TRAIN | ID_VAL | ID_TEST | COMP_VAL | FINAL_OOD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in DATASETS:
        data = datasets[variant]
        counts = data["final_split_counts"]
        lines.append(
            f"| {variant} | {data['source_count']} | {counts['TRAIN']} | {counts['ID_VAL']} | "
            f"{counts['ID_TEST']} | {counts['COMP_VAL']} | {counts['FINAL_OOD']} |"
        )
    lines.extend(["", "## Held-out pair support", ""])
    for variant in DATASETS:
        data = datasets[variant]
        lines.extend(
            [
                f"### {variant}",
                "",
                "| Rule | Role | Before precedence | Effective after precedence |",
                "|---|---|---:|---:|",
            ]
        )
        for rule in ALL_RULES:
            name = rule[0]
            role = "COMP_VAL" if name.startswith("CV") else "FINAL_OOD"
            lines.append(
                f"| {name}: `{rule[1]}={rule[2]} x {rule[3]}={rule[4]}` | {role} | "
                f"{data['pair_counts_before_precedence'][name]} | "
                f"{data['pair_counts_after_precedence'][name]} |"
            )
        overlap = data["overlap_counts"]
        lines.extend(
            [
                "",
                (
                    f"Overlap counts: multiple FINAL_OOD = {overlap['multiple_final_ood_definitions']}; "
                    f"multiple COMP_VAL = {overlap['multiple_comp_val_definitions']}; "
                    f"both panels = {overlap['both_final_ood_and_comp_val']}."
                ),
                "",
                (
                    f"Post-holdout marginal minimum = {data['post_holdout_marginal_minimum']}; "
                    f"post-holdout partner-diversity minimum = "
                    f"{data['post_holdout_partner_diversity']['minimum']}; "
                    f"unintended absent pair cells = {data['unintended_absent_pair_count']}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Integrity result",
            "",
            "For both datasets, the five manifests are pairwise image-ID disjoint, their union equals the source set, every source image is assigned once, all eight held-out pairs have zero TRAIN support, held-out primitives retain positive TRAIN support, alternate-partner diversity is greater than one, and no exact image hash crosses final split roles.",
            "",
        ]
    )
    (REPORT_ROOT / "final_split_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_counts_csv(summary: dict[str, object]) -> None:
    path = REPORT_ROOT / "final_split_counts.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["dataset_variant", "metric_group", "metric", "count"])
        for variant in DATASETS:
            data = summary["datasets"][variant]
            writer.writerow([variant, "source", "source_count", data["source_count"]])
            for role, count in data["final_split_counts"].items():
                writer.writerow([variant, "final_split", role, count])
            for name, count in data["pair_counts_before_precedence"].items():
                writer.writerow([variant, "pair_before_precedence", name, count])
            for name, count in data["pair_counts_after_precedence"].items():
                writer.writerow([variant, "pair_after_precedence", name, count])
            for name, count in data["overlap_counts"].items():
                writer.writerow([variant, "overlap", name, count])
            writer.writerow(
                [variant, "support", "post_holdout_marginal_minimum", data["post_holdout_marginal_minimum"]]
            )
            writer.writerow(
                [
                    variant,
                    "support",
                    "post_holdout_partner_diversity_minimum",
                    data["post_holdout_partner_diversity"]["minimum"],
                ]
            )


def main() -> None:
    rows_by_dataset: dict[str, list[SourceRow]] = {}
    pre_roles_by_dataset: dict[str, dict[str, str]] = {}
    audits: dict[str, dict[str, object]] = {}
    for variant, archive_path in DATASETS.items():
        print(f"Reading and hashing official {variant} archive...", flush=True)
        rows = _read_archive(variant, archive_path)
        pre_roles, audit = _simultaneous_audit(rows)
        rows_by_dataset[variant] = rows
        pre_roles_by_dataset[variant] = pre_roles
        audits[variant] = audit
        print(
            f"{variant}: simultaneous audit passed; "
            f"ID_POOL={audit['role_counts_before_id_split']['ID_POOL']}, "
            f"COMP_VAL={audit['role_counts_before_id_split']['COMP_VAL']}, "
            f"FINAL_OOD={audit['role_counts_before_id_split']['FINAL_OOD']}",
            flush=True,
        )

    final_roles_by_dataset: dict[str, dict[str, str]] = {}
    for variant, rows in rows_by_dataset.items():
        final_roles, distribution = _split_id_pool(rows, pre_roles_by_dataset[variant])
        integrity = _validate_final_partition(rows, final_roles)
        final_roles_by_dataset[variant] = final_roles
        audits[variant]["id_split_distribution_check"] = distribution
        audits[variant]["final_split_counts"] = dict(
            sorted(Counter(final_roles.values()).items(), key=lambda item: list(FINAL_ROLE_FILES).index(item[0]))
        )
        audits[variant]["integrity_assertions"] = integrity

    generated_at = datetime.now(UTC).isoformat()
    configuration = _build_configuration(generated_at)
    summary = {
        "schema": "tiny_avatar_diffusion.dr01_split_summary",
        "version": "dr01_v1",
        "generation_timestamp_utc": generated_at,
        "cross_dataset_consistency": {
            "selected_attributes_identical": True,
            "heldout_definitions_identical": True,
            "precedence_identical": True,
            "id_split_policy_identical": True,
            "split_seed_identical": True,
            "all_checks_passed": True,
        },
        "datasets": audits,
    }
    _write_outputs(rows_by_dataset, final_roles_by_dataset, configuration, summary)
    print(f"Wrote {OUTPUT_ROOT.relative_to(ROOT)} and {REPORT_ROOT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
