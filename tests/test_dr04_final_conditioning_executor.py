from __future__ import annotations
import json
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import tiny_avatar_diffusion.dr04_final_conditioning_executor as executor_module
from tiny_avatar_diffusion.dr04_final_conditioning_executor import (FinalConditioningExecutor, bootstrap_summary, build_release_addendum, expand_plan,
    FROZEN_PARAMETER_COUNTS, collect_parameter_counts, matched_control_diagnostics, prompt_cluster_bootstrap, secondary_attribute_metrics,
    frozen_bootstrap_config, id_ood_gap_bootstrap, ood_prediction_groups, validate_panel, validate_parameter_counts, validate_timings)
from tiny_avatar_diffusion.evaluator import CLASS_COUNTS

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_panel_expansion_and_hash_bound_geometry():
    panel = json.loads((ROOT / "data/processed/dr04_final_eval_v1/final_prompt_panel_manifest.json").read_text())
    controls = json.loads((ROOT / "data/processed/dr04_final_eval_v1/matched_controls_manifest.json").read_text())
    rows = validate_panel(panel, controls); plan = expand_plan(rows, "c.pt", "csha")
    assert len(plan) == 1260 and sum(x["panel_split"] == "ID" for x in plan) == 760
    assert len({(x["prompt_id"], x["seed"]) for x in plan}) == 1260

def _plan():
    return [{"planned_sample_index": i, "sample_id": f"s{i}", "prompt_id": f"p{i//5}", "prompt": "x", "panel_split": "ID", "ood_family": None, "control_role": None, "matched_ood_prompt_id": None, "seed": 2026+i%5, "checkpoint_path": "c", "checkpoint_sha256": "h", "targets": {"face_shape": 1, "glasses": 1, "hair_color": 1, "face_color": 1}} for i in range(5)]

def test_resume_integrity_and_prediction_gate(tmp_path):
    ex = FinalConditioningExecutor(tmp_path, _plan(), {"evaluator_checkpoint_path": "e", "evaluator_checkpoint_sha256": "h"})
    ex.record_parameter_counts(FROZEN_PARAMETER_COUNTS)
    calls=[]
    def gen(slot):
        calls.append(slot["seed"])
        if len(calls)==2: raise RuntimeError("interrupt")
        return np.zeros((32,32,3), dtype=np.uint8)
    with pytest.raises(RuntimeError): ex.generate(gen)
    retried=[]
    ex.generate(lambda slot: (retried.append(slot["seed"]), np.zeros((32,32,3), dtype=np.uint8))[1]); assert calls[1] == retried[0]
    ex.audit(); ex.predict(lambda _: {"face_shape":1,"glasses":1,"hair_color":1,"face_color":1})
    assert ex.metrics()["ID_MAA"] == 1.0
    with pytest.raises(FileExistsError):
        ex.finalize({}); ex.finalize({})


def test_frozen_parameter_counts_reject_mismatch_and_evaluator_double_counting():
    assert validate_parameter_counts(FROZEN_PARAMETER_COUNTS) == FROZEN_PARAMETER_COUNTS
    class Parameter:
        def __init__(self, number): self.number = number
        def numel(self): return self.number
    class Model:
        def __init__(self, number): self.number = number
        def parameters(self): return [Parameter(self.number)]
    assert collect_parameter_counts(Model(4_595_779), Model(25_152), Model(102_976), Model(293_064)) == FROZEN_PARAMETER_COUNTS
    wrong_conditional = {**FROZEN_PARAMETER_COUNTS, "conditional_generator": {**FROZEN_PARAMETER_COUNTS["conditional_generator"], "total": 4_723_908}}
    with pytest.raises(ValueError): validate_parameter_counts(wrong_conditional)
    with pytest.raises(ValueError): validate_parameter_counts({**FROZEN_PARAMETER_COUNTS, "evaluator": 293_065})
    with pytest.raises(ValueError): validate_parameter_counts({**FROZEN_PARAMETER_COUNTS, "conditional_generator": {**FROZEN_PARAMETER_COUNTS["conditional_generator"], "total": 4_723_907 + 293_064}})


def test_secondary_metrics_are_hand_computable_and_label_bound():
    targets = {"face_shape": [0, 1, 2, 2], "glasses": [0, 0, 1, 1], "hair_color": [0, 1, 2, 3], "face_color": [0, 1, 2, 3]}
    predicted = {"face_shape": [0, 2, 1, 2], "glasses": [0, 1, 1, 1], "hair_color": [0, 1, 2, 3], "face_color": [0, 1, 2, 3]}
    rows = [{"targets": {a: targets[a][i] for a in targets}, "predictions": {a: predicted[a][i] for a in predicted}} for i in range(4)]
    result = secondary_attribute_metrics(rows)
    face_classes = CLASS_COUNTS["face_shape"]
    face_matrix = [[0] * face_classes for _ in range(face_classes)]
    face_matrix[0][0] = face_matrix[1][2] = face_matrix[2][1] = face_matrix[2][2] = 1
    assert result["face_shape"] == {"label_order": list(range(face_classes)), "confusion_matrix": face_matrix, "macro_f1": pytest.approx(1.5 / face_classes), "balanced_accuracy": pytest.approx(1.5 / face_classes)}
    glasses_classes = CLASS_COUNTS["glasses"]
    glasses_matrix = [[0] * glasses_classes for _ in range(glasses_classes)]
    glasses_matrix[0][0] = glasses_matrix[0][1] = 1; glasses_matrix[1][1] = 2
    assert result["glasses"]["label_order"] == list(range(glasses_classes))
    assert result["glasses"]["confusion_matrix"] == glasses_matrix
    assert result["glasses"]["macro_f1"] == pytest.approx((2 / 3 + .8) / glasses_classes)
    assert result["glasses"]["balanced_accuracy"] == pytest.approx(1.5 / glasses_classes)
    assert set(result) == {"face_shape", "glasses", "hair_color", "face_color"}
    rows[0]["predictions"]["glasses"] = glasses_classes
    with pytest.raises(ValueError): secondary_attribute_metrics(rows)


def test_finalization_requires_counts_secondary_and_complete_finite_timings(tmp_path):
    ticks = iter(range(100))
    ex = FinalConditioningExecutor(tmp_path, _plan(), {"evaluator_checkpoint_path": "e", "evaluator_checkpoint_sha256": "h"}, clock=lambda: float(next(ticks)))
    ex.generate(lambda _: np.zeros((32, 32, 3), dtype=np.uint8)); ex.audit(); ex.predict(lambda _: {a: 1 for a in ("face_shape", "glasses", "hair_color", "face_color")}); ex.metrics()
    with pytest.raises(ValueError): ex.finalize({})
    ex.record_parameter_counts(FROZEN_PARAMETER_COUNTS)
    state = ex.state(); state["timings_seconds"]["metrics"] = float("nan"); ex._write(state)
    with pytest.raises(ValueError): ex.finalize({})
    state = ex.state(); state["timings_seconds"]["metrics"] = 1.0; ex._write(state)
    summary = ex.finalize({"NONSCIENTIFIC_DRYRUN": True, "NOT_FOR_REPORT_RESULTS": True})
    assert summary["version"] == 3 and set(summary["timings_seconds"]) == {"kind", "generation", "integrity", "evaluator_inference", "metrics", "total"}
    assert set(summary["secondary_metrics"]) == {"face_shape", "glasses", "hair_color", "face_color"}
    with pytest.raises(ValueError): validate_timings({"generation": 0.0})


def test_cli_double_guard_and_status_only_are_non_scientific(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_test", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    status_calls = []
    monkeypatch.setattr(module, "preflight", lambda: status_calls.append("preflight") or {"status": "read_only"})
    module.main([]); module.main(["--status-only"])
    assert status_calls == ["preflight", "preflight"] and not (tmp_path / "results").exists()
    with pytest.raises(SystemExit): module.main(["--mode", "final"])
    with pytest.raises(SystemExit): module.main(["--confirm-final-conditioning-eval"])


def test_real_cli_orchestration_synthetic_lifecycle_resume_and_reentry(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_integration", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path / "synthetic_final"; plan = _plan()
    identities = {"evaluator_checkpoint_path": "synthetic_e.pt", "evaluator_checkpoint_sha256": "e"}
    monkeypatch.setattr(module, "execution_context", lambda: (root, plan, identities, {"NONSCIENTIFIC_DRYRUN": True, "NOT_FOR_REPORT_RESULTS": True}))
    monkeypatch.setattr(module, "preflight", lambda: {"status": "read_only"})
    class Parameter:
        def __init__(self, number): self.number = number
        def numel(self): return self.number
    class Model:
        def __init__(self, number): self.number = number
        def parameters(self): return [Parameter(self.number)]
    generated, evaluated = [], []
    interrupted = [True]
    def generation_loader():
        def generate(slot):
            generated.append((slot["sample_id"], slot["seed"]))
            if interrupted[0] and slot["sample_id"] == "s1": raise RuntimeError("synthetic interruption")
            return np.full((32, 32, 3), slot["seed"] % 255, dtype=np.uint8)
        return generate, (Model(4_595_779), Model(25_152), Model(102_976))
    def evaluator_loader():
        def evaluate(_):
            evaluated.append(1)
            return {a: 1 for a in ("face_shape", "glasses", "hair_color", "face_color")}
        return evaluate, Model(293_064)
    monkeypatch.setattr(module, "_load_conditional_generator", generation_loader)
    monkeypatch.setattr(module, "_load_evaluator", evaluator_loader)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        module.main(["--mode", "final", "--confirm-final-conditioning-eval"])
    first_rows = json.loads((root / "manifests/generated_images.jsonl").read_text().splitlines()[0])
    assert first_rows["sample_id"] == "s0" and first_rows["seed"] == 2026
    interrupted[0] = False
    module.main(["--mode", "final", "--confirm-final-conditioning-eval"])
    summary = json.loads((root / "final_conditioning_summary.json").read_text())
    assert summary["generation"]["completed"] == 5 and summary["evaluator_inference"]["observed_predictions"] == 5
    assert len((root / "manifests/generated_images.jsonl").read_text().splitlines()) == 5
    assert len((root / "manifests/predictions.jsonl").read_text().splitlines()) == 5
    assert [seed for sample, seed in generated if sample == "s1"] == [2027, 2027]
    completed_generation, completed_evaluation = len(generated), len(evaluated)
    module.main(["--mode", "final", "--confirm-final-conditioning-eval"])
    assert len(generated) == completed_generation and len(evaluated) == completed_evaluation
    assert len(list(root.glob("final_conditioning_summary.json"))) == 1


def test_production_orchestration_rejects_mocked_parameter_mismatch_before_generation(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_parameter_test", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path / "parameter_rejection"; generated = []
    monkeypatch.setattr(module, "execution_context", lambda: (root, _plan(), {"evaluator_checkpoint_path": "e", "evaluator_checkpoint_sha256": "h"}, {}))
    class Parameter:
        def __init__(self, number): self.number = number
        def numel(self): return self.number
    class Model:
        def __init__(self, number): self.number = number
        def parameters(self): return [Parameter(self.number)]
    monkeypatch.setattr(module, "_load_conditional_generator", lambda: ((lambda _: generated.append(1)), (Model(4_595_780), Model(25_152), Model(102_976))))
    monkeypatch.setattr(module, "_load_evaluator", lambda: ((lambda _: {a: 1 for a in ("face_shape", "glasses", "hair_color", "face_color")}), Model(293_064)))
    with pytest.raises(ValueError, match="parameter-count mismatch"):
        module.execute_final_scientific_run()
    assert generated == [] and not (root / "images").exists()
    monkeypatch.setattr(module, "_load_conditional_generator", lambda: ((lambda _: generated.append(1)), (Model(4_595_779), Model(25_152), Model(102_976))))
    monkeypatch.setattr(module, "_load_evaluator", lambda: ((lambda _: {a: 1 for a in ("face_shape", "glasses", "hair_color", "face_color")}), Model(293_065)))
    with pytest.raises(ValueError, match="parameter-count mismatch"):
        module.execute_final_scientific_run()
    assert generated == [] and not (root / "images").exists()


def test_status_only_is_read_only_for_absent_partial_and_complete_synthetic_roots(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_status_test", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "preflight", lambda: {"status": "read_only"})
    callbacks = []
    monkeypatch.setattr(module, "_load_conditional_generator", lambda: callbacks.append("generator"))
    monkeypatch.setattr(module, "_load_evaluator", lambda: callbacks.append("evaluator"))
    absent = tmp_path / "absent"; partial = tmp_path / "partial"; complete = tmp_path / "complete"
    ex = FinalConditioningExecutor(partial, _plan(), {"evaluator_checkpoint_path": "e", "evaluator_checkpoint_sha256": "h"}); ex.state()
    ex = FinalConditioningExecutor(complete, _plan(), {"evaluator_checkpoint_path": "e", "evaluator_checkpoint_sha256": "h"}); ex.state()
    before = {root: sorted((p.relative_to(root).as_posix(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()) for root in (partial, complete)}
    module.main(["--status-only"]); module.main(["--status-only"]); module.main(["--status-only"])
    after = {root: sorted((p.relative_to(root).as_posix(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()) for root in (partial, complete)}
    assert not absent.exists() and callbacks == [] and before == after

def test_bootstrap_is_fixed_and_prompt_clustered():
    values = {"p0": [0, 0, 0, 0, 0], "p1": [1, 1, 1, 1, 1]}
    config = frozen_bootstrap_config()
    assert config == {"confidence_level": .95, "method": "percentile", "repetitions": 10_000, "seed": 2026, "cluster_unit": "prompt", "all_seeds_clustered": True}
    assert prompt_cluster_bootstrap(values) == prompt_cluster_bootstrap(values)
    with pytest.raises(ValueError): prompt_cluster_bootstrap({"p0": [1]}, repetitions=10_000)
    with pytest.raises(ValueError): prompt_cluster_bootstrap(values, seed=202605)

def test_hand_computable_maa_is_prompt_first_and_split_separated(tmp_path):
    plan = _plan() + [{**x, "planned_sample_index": x["planned_sample_index"] + 5, "sample_id": f"o{x['planned_sample_index']}", "prompt_id": "ood", "panel_split": "OOD", "ood_family": None} for x in _plan()]
    ex = FinalConditioningExecutor(tmp_path, plan, {"evaluator_checkpoint_path":"e", "evaluator_checkpoint_sha256":"h"})
    ex.generate(lambda _: np.zeros((32,32,3), dtype=np.uint8)); ex.audit()
    calls = iter([{"face_shape":1,"glasses":1,"hair_color":1,"face_color":1}] * 5 + [{"face_shape":0,"glasses":1,"hair_color":1,"face_color":1}] * 5)
    ex.predict(lambda _: next(calls)); result = ex.metrics()
    assert result["ID_MAA"] == 1.0 and result["OOD_MAA"] == .75

def test_matched_controls_and_bootstrap_are_frozen_and_auditable():
    rows=[]
    for family, attrs in {"OOD1":("face_shape","glasses"),"OOD2":("face_shape","hair_color"),"OOD3":("glasses","face_color"),"OOD4":("hair_color","face_color")}.items():
        for role in ("A_FIXED","B_FIXED"):
            for prompt in range(19):
                for seed in range(5): rows.append({"ood_family":family,"control_role":role,"correct":{a: True for a in ("face_shape","glasses","hair_color","face_color")}})
    diagnostics=matched_control_diagnostics(rows)
    assert len(diagnostics["rows"]) == 12 and all(x["n"] in (95,190) for x in diagnostics["rows"])
    with pytest.raises(ValueError): matched_control_diagnostics(rows[:-1])
    prompts=[{"prompt_id":"a","split":"ID","ood_family":None,"attributes":{a:0. for a in ("face_shape","glasses","hair_color","face_color")}}, {"prompt_id":"b","split":"OOD","ood_family":"OOD1","attributes":{a:1. for a in ("face_shape","glasses","hair_color","face_color")}}]
    result=bootstrap_summary(prompts)
    assert result["resamples"] == 10000 and result["seed"] == 2026 and result["unit"] == "prompt" and result["all_five_seeds_clustered"] is True
    assert result["scopes"]["OOD1"]["estimate"] == 1.0


def _canonical_like_prediction_rows():
    rows=[]
    for family in ("OOD1", "OOD2", "OOD3", "OOD4"):
        for prompt in range(25):
            for seed in (2026, 2027, 2028, 2029, 2030):
                rows.append({"sample_id": f"{family}-{prompt}-{seed}", "prompt_id": f"{family}-{prompt}", "seed": seed, "panel_split": "OOD", "ood_family": family})
        for role in ("A_FIXED", "B_FIXED"):
            for prompt in range(19):
                for seed in (2026, 2027, 2028, 2029, 2030):
                    rows.append({"sample_id": f"id-{family}-{role}-{prompt}-{seed}", "prompt_id": f"id-{family}-{role}-{prompt}", "seed": seed, "panel_split": "ID", "ood_family": family})
    return rows


def test_ood_psr_groups_are_ood_only_and_reject_malformed_populations():
    rows=_canonical_like_prediction_rows(); groups=ood_prediction_groups(rows)
    assert len(rows) == 1260 and {family: len(group) for family, group in groups.items()} == {family: 125 for family in groups}
    assert all({row["panel_split"] for row in group} == {"OOD"} and len({row["prompt_id"] for row in group}) == 25 for group in groups.values())
    missing=[dict(row) for row in rows]; next(row for row in missing if row["panel_split"] == "OOD")["ood_family"]=None
    unknown=[dict(row) for row in rows]; next(row for row in unknown if row["panel_split"] == "OOD")["ood_family"]="OTHER"
    short=[row for row in rows if row["sample_id"] != "OOD1-0-2026"]
    duplicate=rows + [dict(next(row for row in rows if row["sample_id"] == "OOD1-0-2026"))]
    wrong_seed=[dict(row) for row in rows]; next(row for row in wrong_seed if row["sample_id"] == "OOD1-0-2026")["seed"]=999
    all_missing=[dict(row) for row in rows]
    for row in all_missing:
        if row["panel_split"] == "OOD": row["ood_family"]=None
    for malformed in (missing, unknown, short, duplicate, wrong_seed, all_missing):
        with pytest.raises(ValueError): ood_prediction_groups(malformed)


def test_real_canonical_predictions_have_four_valid_ood_psr_groups_read_only():
    from tiny_avatar_diffusion.dr05_final_executor import load_jsonl
    rows=load_jsonl(ROOT / "results/runs/dr04_final_conditioning_v1/manifests/predictions.jsonl")
    groups=ood_prediction_groups(rows)
    assert {family: (len({row["prompt_id"] for row in group}), len(group)) for family, group in groups.items()} == {family: (25, 125) for family in ("OOD1", "OOD2", "OOD3", "OOD4")}
    assert len({row["prompt_id"] for group in groups.values() for row in group}) == 100

def test_hand_computable_psr_and_macro_are_not_micro_averages(tmp_path):
    pairs={"OOD1":("face_shape","glasses"),"OOD2":("face_shape","hair_color"),"OOD3":("glasses","face_color"),"OOD4":("hair_color","face_color")}
    plan=[]
    for family in pairs:
        for prompt in range(25):
            for seed in range(5):
                i=len(plan); plan.append({"planned_sample_index":i,"sample_id":f"x{i}","prompt_id":f"{family}-{prompt}","prompt":"x","panel_split":"OOD","ood_family":family,"control_role":None,"matched_ood_prompt_id":None,"seed":2026+seed,"checkpoint_path":"c","checkpoint_sha256":"h","targets":{a:1 for a in ("face_shape","glasses","hair_color","face_color")}})
    ex=FinalConditioningExecutor(tmp_path,plan,{"evaluator_checkpoint_path":"e","evaluator_checkpoint_sha256":"h"}); ex.generate(lambda _:np.zeros((32,32,3),dtype=np.uint8)); ex.audit()
    cursor=[0]
    def evaluator(_):
        local=cursor[0]%125; family=plan[cursor[0]]["ood_family"]; cursor[0]+=1; first,second=pairs[family]
        output={a:1 for a in ("face_shape","glasses","hair_color","face_color")}
        if 50 <= local < 75: output[first]=0 # p01
        elif 75 <= local < 100: output[second]=0 # p10
        elif local >= 100: output[first]=output[second]=0 # p00
        return output
    ex.predict(evaluator); result=ex.metrics()
    assert all(result["PSR"][family]["PSR"] == .4 for family in pairs)
    assert result["MacroPSR"] == .4
    assert result["PSR"]["OOD1"] == {"PSR":.4,"p00":25,"p01":25,"p10":25,"p11":50}


def test_preflight_binds_the_canonical_vocabulary_hash():
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_vocab_pass", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.preflight()
    assert result["status"] == "preflight_pass"
    assert result["hashes"]["data/processed/dr01_v1/text/10k/vocab.json"] == module.EXPECTED[module.VOCAB]


def test_preflight_rejects_a_changed_vocabulary_before_execution(monkeypatch):
    path = ROOT / "scripts" / "dr04_final_conditioning_evaluation.py"
    spec = importlib.util.spec_from_file_location("dr04_final_conditioning_evaluation_vocab_fail", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    real_hash = module.canonical_sha256
    monkeypatch.setattr(module, "canonical_sha256", lambda candidate: "0" * 64 if candidate == module.VOCAB else real_hash(candidate))
    with pytest.raises(RuntimeError, match="frozen identity mismatch"):
        module.preflight()


def test_release_addendum_derives_frozen_gap_without_mutating_sources(tmp_path, monkeypatch):
    root = tmp_path / "final"; manifests = root / "manifests"; manifests.mkdir(parents=True)
    predictions = manifests / "predictions.jsonl"; predictions.write_text('{"synthetic": true}\n', encoding="utf-8")
    summary = root / "final_conditioning_summary.json"
    summary.write_text(json.dumps({"run_identity": {"plan_sha256": "plan", "conditional_checkpoint_sha256": "generator", "evaluator_checkpoint_sha256": "evaluator"}}), encoding="utf-8")
    metrics = root / "metrics.json"; metrics.write_text('{"sealed": true}\n', encoding="utf-8")
    prompt_values = [
        {"prompt_id": "id", "split": "ID", "ood_family": None, "attributes": {a: 1.0 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
        {"prompt_id": "ood", "split": "OOD", "ood_family": "OOD1", "attributes": {a: 0.0 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
    ]
    monkeypatch.setattr(executor_module, "prompt_values_from_canonical_predictions", lambda _: prompt_values)
    before = {path: path.read_bytes() for path in (predictions, summary, metrics)}
    addendum = build_release_addendum(root, vocabulary_path="data/processed/dr01_v1/text/10k/vocab.json", vocabulary_sha256="vocab")
    assert addendum["id_ood_gap"]["estimate"] == 1.0
    assert addendum["id_ood_gap"]["bootstrap"]["id_ood_resampling"] == "independent"
    assert {path: path.read_bytes() for path in (predictions, summary, metrics)} == before


def test_id_ood_gap_ci_uses_the_frozen_independent_prompt_bootstrap():
    values = [
        {"prompt_id": "id0", "split": "ID", "attributes": {a: 0.0 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
        {"prompt_id": "id1", "split": "ID", "attributes": {a: 1.0 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
        {"prompt_id": "ood0", "split": "OOD", "attributes": {a: 0.0 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
        {"prompt_id": "ood1", "split": "OOD", "attributes": {a: 0.5 for a in ("face_shape", "glasses", "hair_color", "face_color")}},
    ]
    result = id_ood_gap_bootstrap(values)
    assert result["estimate"] == pytest.approx(0.25)
    assert result["bootstrap"] == {"confidence_level": .95, "method": "percentile", "repetitions": 10_000, "seed": 2026, "cluster_unit": "prompt", "all_five_seeds_clustered": True, "id_ood_resampling": "independent"}
    assert result["percentile_95"] == id_ood_gap_bootstrap(values)["percentile_95"]


def test_post_finalization_bootstrap_repair_is_not_in_canonical_execution_code():
    assert not hasattr(executor_module, "repair_canonical_bootstrap")
