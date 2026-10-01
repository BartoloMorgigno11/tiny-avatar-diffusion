"""Exact, metadata-only DR04-A15 joint OOD/control feasibility audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import scipy
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "audits" / "dr04_joint_panel_feasibility_v1"
ID = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "id_test.jsonl"
OOD = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "final_ood.jsonl"
ATTR = ("face_shape", "glasses", "hair_color", "face_color")
PAIRS = {"OOD1": ("face_shape", 2, "glasses", 3), "OOD2": ("face_shape", 4, "hair_color", 6), "OOD3": ("glasses", 10, "face_color", 10), "OOD4": ("hair_color", 0, "face_color", 0)}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(record["split_role"] != split for record in records):
        raise ValueError(f"unexpected split membership in {path}")
    return records


def belongs(record: dict, pair: str) -> bool:
    a, u, b, v = PAIRS[pair]
    return record[a] == u and record[b] == v


def ordinary(record: dict) -> bool:
    return not any(belongs(record, pair) for pair in PAIRS)


def canonical(rows: list[dict], split: str) -> list[dict]:
    result: dict[str, dict] = {}
    for row in rows:
        caption = row["caption"]
        key = digest(f"dr04_prompt_source_v1|{split}|{caption}|{row['image_id']}")
        if caption not in result or key < result[caption]["representative_source_hash"]:
            result[caption] = {**{attr: row[attr] for attr in ATTR}, "caption": caption, "representative_source_sample_id": row["image_id"], "representative_source_hash": key}
    return list(result.values())


def control_cost(q: dict, control: dict, pair: str, control_type: str) -> int | None:
    a, u, b, v = PAIRS[pair]
    if not ordinary(control):
        return None
    eligible = (control[a] == u and control[b] != v) if control_type == "A_FIXED" else (control[a] != u and control[b] == v)
    if not eligible:
        return None
    nuisance = [attr for attr in ATTR if attr not in (a, b)]
    return int(control[nuisance[0]] != q[nuisance[0]]) + int(control[nuisance[1]] != q[nuisance[1]])


def build_model(candidates: list[dict], controls: list[dict], reference: set[str] | None, retain: int | None, rank_objective: bool):
    edges: list[tuple[int, str, int, int, str]] = []
    for qi, q in enumerate(candidates):
        for typ in ("A_FIXED", "B_FIXED"):
            for ci, control in enumerate(controls):
                cost = control_cost(q, control, q["pair_id"], typ)
                if cost is not None:
                    tie = digest(f"dr04_matched_control_v1|{q['pair_id']}|{typ}|{q['candidate_id']}|{control['caption']}")
                    edges.append((qi, typ, ci, cost, tie))
    n_x, n_y = len(candidates), len(edges)
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []; lb: list[float] = []; ub: list[float] = []
    row = 0
    for pair in PAIRS:
        for qi, q in enumerate(candidates):
            if q["pair_id"] == pair: rows.append(row); cols.append(qi); vals.append(1)
        lb.append(25); ub.append(25); row += 1
    by_slot: dict[tuple[int, str], list[int]] = {}
    for yi, (qi, typ, _, _, _) in enumerate(edges): by_slot.setdefault((qi, typ), []).append(yi)
    for qi in range(n_x):
        for typ in ("A_FIXED", "B_FIXED"):
            for yi in by_slot.get((qi, typ), []): rows.append(row); cols.append(n_x + yi); vals.append(1)
            rows.append(row); cols.append(qi); vals.append(-1)
            lb.append(0); ub.append(0); row += 1
    for ci in range(len(controls)):
        for yi, (_, _, edge_ci, _, _) in enumerate(edges):
            if edge_ci == ci: rows.append(row); cols.append(n_x + yi); vals.append(1)
        lb.append(0); ub.append(1); row += 1
    if retain is not None:
        for qi, q in enumerate(candidates):
            if q["caption"] in reference: rows.append(row); cols.append(qi); vals.append(1)
        lb.append(retain); ub.append(retain); row += 1
    matrix = coo_matrix((vals, (rows, cols)), shape=(row, n_x + n_y)).tocsr()
    objective = [0.0] * (n_x + n_y)
    if reference is not None and retain is None:
        for qi, q in enumerate(candidates): objective[qi] = -1.0 if q["caption"] in reference else 0.0
    if rank_objective:
        ranks = {q["candidate_id"]: rank for rank, q in enumerate(sorted(candidates, key=lambda x: x["selection_hash"]))}
        for qi, q in enumerate(candidates): objective[qi] = ranks[q["candidate_id"]]
    return objective, matrix, lb, ub, edges, row


def solve(candidates: list[dict], controls: list[dict], reference: set[str] | None = None, retain: int | None = None, rank_objective: bool = False):
    objective, matrix, lb, ub, edges, constraint_count = build_model(candidates, controls, reference, retain, rank_objective)
    result = milp(c=objective, integrality=[1] * len(objective), bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lb, ub), options={"disp": False})
    return result, edges, {"x_variables": len(candidates), "y_variables": len(edges), "total_binary_variables": len(objective), "constraints": constraint_count}


def select_reference(candidates: list[dict]) -> set[str]:
    return {q["caption"] for pair in PAIRS for q in sorted((x for x in candidates if x["pair_id"] == pair), key=lambda x: x["selection_hash"])[:25]}


def selected_candidates(candidates: list[dict], solution) -> list[dict]:
    return [q for qi, q in enumerate(candidates) if solution.x[qi] > 0.5]


def match_controls(selected: list[dict], controls: list[dict]) -> tuple[list[dict], dict]:
    slots = [(q, typ) for q in selected for typ in ("A_FIXED", "B_FIXED")]
    edge_records = []
    for si, (q, typ) in enumerate(slots):
        for ci, c in enumerate(controls):
            cost = control_cost(q, c, q["pair_id"], typ)
            if cost is not None: edge_records.append((si, ci, cost, digest(f"dr04_matched_control_v1|{q['pair_id']}|{typ}|{q['prompt_id']}|{c['caption']}")))
    rank = {key: i for i, key in enumerate(sorted(edge[3] for edge in edge_records))}
    scale = len(slots) * len(edge_records) + 1; forbidden = 3 * scale + len(edge_records) + 1
    cost_matrix = [[forbidden] * len(controls) for _ in slots]
    for si, ci, cost, key in edge_records: cost_matrix[si][ci] = cost * scale + rank[key]
    ri, ci = linear_sum_assignment(cost_matrix)
    if len(ri) != 200 or any(cost_matrix[i][j] >= forbidden for i, j in zip(ri, ci)): raise ValueError("witness matching unexpectedly incomplete")
    records = []
    for si, ci in zip(ri, ci):
        q, typ = slots[si]; c = controls[ci]; cost = control_cost(q, c, q["pair_id"], typ)
        records.append({"prompt_id": f"AUDIT_ID_{q['pair_id']}_{typ}_{q['prompt_id'].rsplit('_', 1)[1]}", "domain": "ID", "source_pair_id": q["pair_id"], "control_type": typ, "matched_ood_prompt_id": q["prompt_id"], "caption": c["caption"], **{a: c[a] for a in ATTR}, "representative_source_sample_id": c["representative_source_sample_id"], "matching_cost": cost, "selection_hash": digest(f"dr04_matched_control_v1|{q['pair_id']}|{typ}|{q['prompt_id']}|{c['caption']}"), "ordinary_against_all_final_ood": True})
    diagnostics = {"overall_cost_histogram": {str(k): sum(r["matching_cost"] == k for r in records) for k in range(3)}, "mean_matching_cost": sum(r["matching_cost"] for r in records) / 200, "both_controls_cost0": sum(all(r["matching_cost"] == 0 for r in records if r["matched_ood_prompt_id"] == q["prompt_id"]) for q in selected), "any_cost2": sum(any(r["matching_cost"] == 2 for r in records if r["matched_ood_prompt_id"] == q["prompt_id"]) for q in selected)}
    return sorted(records, key=lambda r: r["prompt_id"]), diagnostics


def write_json(name: str, value: dict | list) -> None:
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main() -> None:
    if OUT.exists(): raise RuntimeError(f"audit output already exists: {OUT}")
    id_pool = canonical(read_rows(ID, "ID_TEST"), "ID_TEST")
    ood_pool = canonical(read_rows(OOD, "FINAL_OOD"), "FINAL_OOD")
    candidates = []
    for pair in PAIRS:
        for index, q in enumerate(sorted((x for x in ood_pool if belongs(x, pair) and sum(belongs(x, p) for p in PAIRS) == 1), key=lambda x: digest(f"dr04_final_ood_prompt_v1|{pair}|{x['caption']}"))):
            candidates.append({**q, "pair_id": pair, "candidate_id": f"{pair}_{index:03d}", "selection_hash": digest(f"dr04_final_ood_prompt_v1|{pair}|{q['caption']}" )})
    reference = select_reference(candidates)
    f0, _, counts = solve(candidates, id_pool)
    summary = {"status": "AUDIT_ONLY", "source_metadata_sha256": {"ID_TEST": sha256(ID), "FINAL_OOD": sha256(OOD)}, "candidate_counts_per_pair": {p: sum(q["pair_id"] == p for q in candidates) for p in PAIRS}, "reference_ood_count": len(reference), "f0_solver_status": int(f0.status), "f0_message": f0.message, "joint_panel_feasible": bool(f0.success), "variable_counts": counts}
    provenance = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "script_sha256": sha256(Path(__file__)), "solver": "scipy.optimize.milp / HiGHS", "scipy_version": scipy.__version__, "formulation": "binary x OOD selection and binary y eligible control assignment; pair equalities, y-to-x equalities, and globally unique control inequalities", "metadata_files_read": [str(ID.relative_to(ROOT)), str(OOD.relative_to(ROOT))], "permitted_fields_read": ["image_id", "split_role", *ATTR, "caption"], "image_content_accessed": False, "generator_executed": False, "evaluator_executed": False}
    OUT.mkdir(parents=True)
    if not f0.success:
        write_json("summary.json", summary); write_json("solver_provenance.json", provenance); print(json.dumps(summary, indent=2)); return
    f1, _, _ = solve(candidates, id_pool, reference=reference)
    retained = round(-f1.fun)
    f2, _, _ = solve(candidates, id_pool, reference=reference, retain=retained, rank_objective=True)
    selected = selected_candidates(candidates, f2)
    for pair in PAIRS:
        if sum(q["pair_id"] == pair for q in selected) != 25: raise ValueError("pair selection invariant failed")
    selected = sorted(selected, key=lambda q: (q["pair_id"], q["selection_hash"]))
    witness = [{**q, "prompt_id": f"AUDIT_OOD_{q['pair_id']}_{i:02d}", "domain": "OOD", "exclusive": True, "all_final_ood_memberships": [q["pair_id"]]} for pair in PAIRS for i, q in enumerate(q for q in selected if q["pair_id"] == pair)]
    witness_controls, diagnostics = match_controls(witness, id_pool)
    f2b, _, _ = solve(candidates, id_pool, reference=reference, retain=retained, rank_objective=True)
    second = selected_candidates(candidates, f2b)
    reproducible = {q["caption"] for q in selected} == {q["caption"] for q in second}
    if not reproducible: raise ValueError("MILP witness selection was not reproducible")
    replacements = {p: 25 - sum(q["pair_id"] == p and q["caption"] in reference for q in selected) for p in PAIRS}
    summary.update({"f1_retained_original_ood_prompts": retained, "f1_minimum_ood_replacements": 100 - retained, "replacements_per_pair": replacements, "f2_status": "AUDIT_WITNESS_NOT_FROZEN", "witness_ood_counts_per_pair": {p: sum(q["pair_id"] == p for q in witness) for p in PAIRS}, "witness_control_assignment": "200/200", "matching_diagnostics": diagnostics, "reproducible": reproducible})
    write_json("summary.json", summary); write_json("solver_provenance.json", provenance); write_json("audit_witness_ood.json", {"status": "AUDIT_WITNESS_NOT_FROZEN", "prompt_records": witness}); write_json("audit_witness_controls.json", {"status": "AUDIT_WITNESS_NOT_FROZEN", "prompt_records": witness_controls})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
