from __future__ import annotations

import inspect
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import tiny_avatar_diffusion.ca16_evaluation as evaluation_module
import tiny_avatar_diffusion.ca16_r11_orchestration as r11
from tiny_avatar_diffusion.ca16_evaluation import (
    EVALUATOR_CHECKPOINT_SHA256,
    EVALUATOR_FREEZE_SHA256,
    load_checkpoint_selection_manifest,
)
from tiny_avatar_diffusion.ca16_experiment import ArchitectureArm, CA16RunSpec

CHECKPOINT_SHA = "a" * 64
EVALUATOR = {
    "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
    "freeze_manifest_sha256": EVALUATOR_FREEZE_SHA256,
}


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch):
    manifest = load_checkpoint_selection_manifest()
    records = [manifest["id_entries"][0]]
    for family in ("CV1", "CV2", "CV3", "CV4"):
        records.append(
            next(
                row
                for row in manifest["comp_entries"]
                if row["comp_families"] == [family]
            )
        )
    monkeypatch.setattr(r11, "TOTAL_SAMPLES", 5)
    monkeypatch.setattr(r11, "EXPECTED_COUNTS", {"ID_VAL": 1, "COMP_VAL": 4})
    monkeypatch.setattr(
        r11,
        "EXPECTED_SELECTION_FAMILY_COUNTS",
        {family: 1 for family in ("CV1", "CV2", "CV3", "CV4")},
    )
    monkeypatch.setattr(r11, "canonical_records", lambda: tuple(records))
    return tuple(records)


def pixels(value: int = 0) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def spec() -> CA16RunSpec:
    return CA16RunSpec(ArchitectureArm.CA8, 2026)


def populate_candidate(
    tmp_path: Path, records, milestone: int = 5000, run_spec: CA16RunSpec | None = None
) -> None:
    r11.resume_candidate_generation(
        output_root=tmp_path,
        spec=run_spec or spec(),
        milestone=milestone,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate=lambda _record: pixels(),
    )


def evaluate_candidate(tmp_path: Path, records, milestone: int = 5000):
    return r11.recompute_candidate_metrics(
        output_root=tmp_path,
        spec=spec(),
        milestone=milestone,
        checkpoint_sha256=CHECKPOINT_SHA,
        evaluator_identity=EVALUATOR,
        predict=lambda path: dict(
            next(row for row in records if row["stable_record_key"] in path.name).get(
                "ground_truth_attributes", {}
            )
        ),
    )


def perfect_predictor(records):
    ordered = iter(records)
    return lambda _path: dict(next(ordered)["ground_truth_attributes"])


def test_exact_six_by_four_plan_and_inventory() -> None:
    plan = r11.expected_plan()
    assert len(plan) == 24
    assert [row[0] for row in plan[::4]] == list(r11.RUN_ORDER)
    assert [row[3] for row in plan[:4]] == [5000, 10000, 15000, 20000]
    inventory = r11.load_checkpoint_inventory()
    assert len(inventory["entries"]) == 24


def test_canonical_panel_contract_and_forbidden_roles() -> None:
    records = r11.canonical_records()
    assert len(records) == 493
    assert sum(row["split_role"] == "ID_VAL" for row in records) == 256
    assert sum(row["split_role"] == "COMP_VAL" for row in records) == 237
    bad = dict(records[0], split_role="ID_TEST")
    with pytest.raises(ValueError, match="forbidden"):
        r11.expected_sample_record(
            output_root=Path("x"),
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            canonical_record=bad,
            sample_index=0,
        )
    bad["split_role"] = "FINAL_OOD"
    with pytest.raises(ValueError, match="forbidden"):
        r11.expected_sample_record(
            output_root=Path("x"),
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            canonical_record=bad,
            sample_index=0,
        )


@pytest.mark.parametrize("role", ["ID_VAL", "COMP_VAL"])
def test_real_selection_record_joins_authoritative_tokens(role: str) -> None:
    manifest = load_checkpoint_selection_manifest()
    record = (
        manifest["id_entries"][0] if role == "ID_VAL" else manifest["comp_entries"][0]
    )
    source_index = r11.load_tokenized_source_index()
    source = source_index[record["stable_record_key"]]

    hydrated = r11.hydrate_selection_record(record, source_index)

    assert hydrated["stable_record_key"] == record["stable_record_key"]
    assert hydrated["generation_seed"] == record["generation_seed"]
    assert hydrated["caption"] == record["caption"]
    assert hydrated["token_ids"] == source["token_ids"]
    assert hydrated["attention_mask"] == source["attention_mask"]


def test_token_join_missing_and_duplicate_source_fail_closed() -> None:
    record = load_checkpoint_selection_manifest()["id_entries"][0]
    source = r11.load_tokenized_source_index()[record["stable_record_key"]]
    with pytest.raises(ValueError, match="no matching"):
        r11.hydrate_selection_record(record, {})
    with pytest.raises(ValueError, match="duplicate"):
        r11._index_tokenized_source_records(
            {"ID_VAL": [source, deepcopy(source)], "COMP_VAL": []}
        )


def test_token_source_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r11, "sha256_file", lambda _path: "wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        r11.load_tokenized_source_index()


@pytest.mark.parametrize("field", ["caption", "image_id", "split_role"])
def test_token_join_identity_mismatch_fails_closed(field: str) -> None:
    record = load_checkpoint_selection_manifest()["id_entries"][0]
    source_index = r11.load_tokenized_source_index()
    key = record["stable_record_key"]
    source_index[key] = {**source_index[key], field: "mismatch"}
    with pytest.raises(ValueError, match=field):
        r11.hydrate_selection_record(record, source_index)


@pytest.mark.parametrize("field", ["token_ids", "attention_mask"])
def test_token_join_missing_conditioning_field_fails_closed(field: str) -> None:
    record = load_checkpoint_selection_manifest()["comp_entries"][0]
    source_index = r11.load_tokenized_source_index()
    key = record["stable_record_key"]
    source_index[key].pop(field)
    with pytest.raises(ValueError, match=field):
        r11.hydrate_selection_record(record, source_index)


def test_sample_ema_record_preserves_joined_tokens_and_canonical_public_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = load_checkpoint_selection_manifest()["id_entries"][0]
    hydrated = r11.hydrate_selection_record(record, r11.load_tokenized_source_index())
    supplied = {
        **hydrated,
        "caption": "caller override",
        "image_id": "caller-override",
        "generation_seed": 9999,
    }
    captured: dict[str, object] = {}

    def fake_sample(
        schedule: object,
        text_encoder: object,
        denoiser: object,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["input_ids"] = input_ids.tolist()
        captured["attention_mask"] = attention_mask.tolist()
        captured["seed"] = kwargs["generator"].initial_seed()
        return torch.zeros((1, 3, 32, 32))

    monkeypatch.setattr(evaluation_module, "conditional_sample", fake_sample)
    result = r11.sample_ema_record(
        object(), object(), object(), supplied, device=torch.device("cpu")
    )

    assert captured["input_ids"] == [hydrated["token_ids"]]
    assert captured["attention_mask"] == [hydrated["attention_mask"]]
    assert captured["seed"] == record["generation_seed"]
    assert result.shape == (3, 32, 32)


@pytest.mark.parametrize(
    ("token_ids", "attention_mask"),
    [
        (None, [1]),
        ([], []),
        ([1, -1], [1, 1]),
        ([1, 2], [1]),
        ([1, 2], [1, 2]),
    ],
)
def test_sample_ema_record_rejects_invalid_conditioning_schema(
    token_ids: object, attention_mask: object
) -> None:
    record = {
        **load_checkpoint_selection_manifest()["id_entries"][0],
        "token_ids": token_ids,
        "attention_mask": attention_mask,
    }
    with pytest.raises(ValueError, match="invalid"):
        r11.sample_ema_record(
            object(), object(), object(), record, device=torch.device("cpu")
        )


@pytest.mark.parametrize("partition", [(2,), (1, 1)])
def test_batched_sampler_noise_matches_legacy_per_record_streams(
    monkeypatch: pytest.MonkeyPatch, partition: tuple[int, ...]
) -> None:
    canonical = r11.canonical_records()
    records = [
        r11.hydrate_selection_record(record, r11.load_tokenized_source_index())
        for record in (canonical[0], canonical[256])
    ]
    observed: list[tuple[torch.Tensor, list[torch.Tensor]]] = []

    def fake_sample(*args: object, **kwargs: object) -> torch.Tensor:
        initial = kwargs["x_T"]
        provider = kwargs["reverse_noise_provider"]
        assert isinstance(initial, torch.Tensor) and callable(provider)
        noise = [provider(timestep, initial) for timestep in range(1_000)]
        observed.append((initial, noise))
        return torch.zeros((initial.shape[0], 3, 32, 32))

    monkeypatch.setattr(evaluation_module, "conditional_sample", fake_sample)
    start = 0
    for size in partition:
        evaluation_module.sample_ema_records_batch(
            object(),
            object(),
            object(),
            records[start : start + size],
            device=torch.device("cpu"),
        )
        start += size
    expected_generators = [
        torch.Generator().manual_seed(record["generation_seed"]) for record in records
    ]
    expected = [
        [torch.randn((1, 3, 32, 32), generator=generator) for _ in range(1_001)]
        for generator in expected_generators
    ]
    reconstructed = [[], []]
    offset = 0
    for initial, reverse in observed:
        for index in range(initial.shape[0]):
            reconstructed[offset + index].append(initial[index : index + 1])
            reconstructed[offset + index].extend(
                noise[index : index + 1] for noise in reverse
            )
        offset += initial.shape[0]
    for actual, expected_stream in zip(reconstructed, expected, strict=True):
        assert len(actual) == 1_001
        assert all(
            torch.equal(value, reference)
            for value, reference in zip(actual, expected_stream, strict=True)
        )


def test_public_generation_api_has_no_records_override_and_seeds_are_frozen() -> None:
    assert (
        "records" not in inspect.signature(r11.resume_candidate_generation).parameters
    )
    manifest = load_checkpoint_selection_manifest()
    assert [row["generation_seed"] for row in manifest["id_entries"]] == [
        2026 + index for index in range(256)
    ]
    canonical = r11.canonical_records()
    comp = [row for row in canonical if row["split_role"] == "COMP_VAL"]
    assert [row["generation_seed"] for row in comp] == [
        row["generation_seed"] for row in manifest["comp_entries"]
    ]


def test_output_classification_new_orphan_and_exact_resume(tmp_path: Path) -> None:
    root = tmp_path / "new"
    assert r11.classify_output_root(root) is r11.OutputState.NEW
    root.mkdir()
    (root / "orphan.png").write_bytes(b"png")
    assert r11.classify_output_root(root) is r11.OutputState.INVALID_OR_UNPROVEN
    root = tmp_path / "valid"
    manifest = {"schema": r11.OUTPUT_SCHEMA, "value": 1}
    r11.initialize_or_resume(root, manifest)
    assert r11.classify_output_root(root, manifest) is r11.OutputState.RESUMABLE_VALID
    with pytest.raises(RuntimeError, match="INVALID"):
        r11.initialize_or_resume(root, {**manifest, "value": 2})


def test_lock_rejects_second_writer_and_stale_lock(tmp_path: Path) -> None:
    with (
        r11.exclusive_run_lock(tmp_path),
        pytest.raises(RuntimeError, match="lock exists"),
        r11.exclusive_run_lock(tmp_path),
    ):
        pass
    (tmp_path / r11.LOCK_NAME).write_text('{"pid":999999}', encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="forensic resolution"),
        r11.exclusive_run_lock(tmp_path),
    ):
        pass


@pytest.mark.parametrize(
    ("state", "expected", "message"),
    [
        (r11.GitState("wrong", "exp/ca16-ablation", True), "right", "HEAD mismatch"),
        (r11.GitState("right", "exp/ca16-ablation", False), "right", "must be clean"),
        (r11.GitState("right", "wrong", True), "right", "branch"),
    ],
)
def test_evaluation_git_guards(state, expected, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        r11.verify_evaluation_git(expected, state)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def committed_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "r11@example.invalid")
    _git(repo, "config", "user.name", "R11 Test")
    _git(repo, "switch", "-c", "exp/ca16-ablation")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_git_provenance_accepts_clean_committed_repository(
    committed_git_repo: Path,
) -> None:
    assert r11.git_state(root=committed_git_repo).tracked_clean


def test_git_provenance_rejects_modified_tracked(committed_git_repo: Path) -> None:
    (committed_git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    assert not r11.git_state(root=committed_git_repo).tracked_clean


def test_git_provenance_rejects_staged(committed_git_repo: Path) -> None:
    (committed_git_repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(committed_git_repo, "add", "staged.txt")
    assert not r11.git_state(root=committed_git_repo).tracked_clean


def test_git_provenance_rejects_untracked_nonignored(committed_git_repo: Path) -> None:
    (committed_git_repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    assert not r11.git_state(root=committed_git_repo).tracked_clean


def test_git_provenance_ignores_gitignored_file(committed_git_repo: Path) -> None:
    (committed_git_repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    assert r11.git_state(root=committed_git_repo).tracked_clean


def test_training_sha_is_independent_from_evaluation_head(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-r10")
    entry = {
        "checkpoint_path": checkpoint.name,
        "sha256": r11.sha256_file(checkpoint),
        "training_git_sha": r11.TRAINING_GIT_SHA,
    }
    assert (
        r11.verify_training_checkpoint_before_load(entry, root=tmp_path) == checkpoint
    )
    r11.verify_evaluation_git(
        "future-eval", r11.GitState("future-eval", "exp/ca16-ablation", True)
    )
    with pytest.raises(ValueError, match="training provenance"):
        r11.verify_training_checkpoint_before_load(
            {**entry, "training_git_sha": "future-eval"}, root=tmp_path
        )


def test_sample_round_trip_partial_resume_and_no_overwrite(
    tmp_path: Path, small_contract
) -> None:
    calls = []
    result = r11.resume_candidate_generation(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate=lambda record: calls.append(record["stable_record_key"]) or pixels(),
    )
    assert result["completed"] == 5 and len(calls) == 5
    again = r11.resume_candidate_generation(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate=lambda _record: pytest.fail("valid sample must not be overwritten"),
    )
    assert len(again["reused"]) == 5 and not again["generated"]


def test_batched_generation_preserves_order_partial_batch_and_resume(
    tmp_path: Path, small_contract
) -> None:
    batches: list[list[str]] = []

    def generate_batch(records):
        batches.append([record["stable_record_key"] for record in records])
        return [pixels(index + 1) for index, _ in enumerate(records)]

    result = r11.resume_candidate_generation_batched(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate_batch=generate_batch,
        sampling_batch_size=2,
    )

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert result["generated"] == [
        record["stable_record_key"] for record in small_contract
    ]
    assert result["completed"] == 5
    resumed = r11.resume_candidate_generation_batched(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate_batch=lambda _records: pytest.fail("validated sample was regenerated"),
        sampling_batch_size=2,
    )
    assert len(resumed["reused"]) == 5


def test_batched_resume_retains_original_group_lanes_after_partial_publication(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    manifest = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(manifest)
    for row in rows[3:]:
        (tmp_path / row["png_relative_path"]).unlink()
    r11.atomic_jsonl(manifest, rows[:3])
    observed: list[list[str]] = []
    r11.resume_candidate_generation_batched(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate_batch=lambda records: (
            observed.append([record["stable_record_key"] for record in records])
            or [pixels(9) for _ in records]
        ),
        sampling_batch_size=8,
    )
    assert observed == [[record["stable_record_key"] for record in small_contract]]


@pytest.mark.parametrize(
    "boundary",
    [
        "TEMP_WRITTEN",
        "TRANSACTION_PERSISTED",
        "FINALIZED",
        "MANIFEST_PERSISTED",
        "TEMP_CLEANED",
    ],
)
def test_sample_transaction_recovers_every_crash_boundary(
    tmp_path: Path,
    small_contract,
    boundary: str,
) -> None:
    run_manifest = {"schema": r11.OUTPUT_SCHEMA, "test": "transaction-recovery"}
    r11.atomic_json(tmp_path / "run_manifest.json", run_manifest)
    crashed = False

    def interrupt(observed: str) -> None:
        nonlocal crashed
        if observed == boundary and not crashed:
            crashed = True
            raise RuntimeError("simulated process death")

    with pytest.raises(RuntimeError, match="process death"):
        r11._resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=lambda _record: pixels(3),
            canonical=small_contract,
            after_boundary=interrupt,
        )
    assert (
        r11.classify_output_root(tmp_path, run_manifest)
        is r11.OutputState.RESUMABLE_VALID
    )
    generation_calls = []
    result = r11.resume_candidate_generation(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate=lambda record: (
            generation_calls.append(record["stable_record_key"]) or pixels(3)
        ),
    )
    assert result["completed"] == 5
    assert len(generation_calls) == (5 if boundary == "TEMP_WRITTEN" else 4)
    transaction_directory = (
        r11.candidate_paths(tmp_path, spec(), 5000).root / "transactions"
    )
    assert not list(transaction_directory.iterdir())


def test_invalid_persisted_sample_resume_never_invokes_generator(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    manifest = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(manifest)
    rows[0]["generation_seed"] += 1
    r11.atomic_jsonl(manifest, rows)
    calls = []
    with pytest.raises(ValueError, match="provenance"):
        r11.resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=lambda record: (
                calls.append(record) or pytest.fail("generator invoked")
            ),
        )
    assert calls == []


def test_transaction_identity_mismatch_fails_closed(
    tmp_path: Path, small_contract
) -> None:
    with pytest.raises(RuntimeError, match="process death"):
        r11._resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=lambda _record: pixels(),
            canonical=small_contract,
            after_boundary=lambda boundary: (
                (_ for _ in ()).throw(RuntimeError("simulated process death"))
                if boundary == "TRANSACTION_PERSISTED"
                else None
            ),
        )
    transaction = next(
        (r11.candidate_paths(tmp_path, spec(), 5000).root / "transactions").glob(
            "*.transaction.json"
        )
    )
    payload = r11.read_json(transaction)
    payload["sample"]["generation_seed"] += 1
    r11.atomic_json(transaction, payload)
    with pytest.raises(ValueError, match="identity or provenance mismatch"):
        r11.resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=lambda _record: pytest.fail("mismatched transaction regenerated"),
        )


def test_interrupted_candidate_generates_only_missing_records(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    manifest = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(manifest)
    retained_hashes = {row["stable_record_key"]: row["png_sha256"] for row in rows[:3]}
    for row in rows[3:]:
        (tmp_path / row["png_relative_path"]).unlink()
    r11.atomic_jsonl(manifest, rows[:3])
    calls = []
    result = r11.resume_candidate_generation(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        generate=lambda record: calls.append(record["stable_record_key"]) or pixels(7),
    )
    assert len(calls) == 2 and len(result["reused"]) == 3
    final = r11.read_jsonl(manifest)
    assert {
        row["stable_record_key"]: row["png_sha256"] for row in final[:3]
    } == retained_hashes


def test_orphan_png_and_invalid_sample_are_never_repaired(
    tmp_path: Path, small_contract
) -> None:
    expected = r11.expected_sample_record(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        canonical_record=small_contract[0],
        sample_index=0,
    )
    r11.persist_new_png(tmp_path / expected["png_relative_path"], pixels())
    with pytest.raises(ValueError, match="orphan"):
        r11.resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=lambda _: pixels(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_seed", 9999, "provenance"),
        ("checkpoint_sha256", "b" * 64, "provenance"),
        ("split_role", "FINAL_OOD", "provenance"),
    ],
)
def test_wrong_sample_provenance_rejected(
    tmp_path: Path, small_contract, field, value, message
) -> None:
    populate_candidate(tmp_path, small_contract)
    path = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(path)
    rows[0][field] = value
    r11.atomic_jsonl(path, rows)
    with pytest.raises(ValueError, match=message):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)


def test_wrong_png_hash_and_duplicate_record_rejected(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    path = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(path)
    rows[0]["png_sha256"] = "0" * 64
    r11.atomic_jsonl(path, rows)
    with pytest.raises(ValueError, match="SHA256"):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)
    rows[0] = deepcopy(rows[1])
    r11.atomic_jsonl(path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)


def test_candidate_cannot_complete_one_short(tmp_path: Path, small_contract) -> None:
    populate_candidate(tmp_path, small_contract)
    path = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(path)
    (tmp_path / rows[-1]["png_relative_path"]).unlink()
    r11.atomic_jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match="exactly 493"):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)


def test_real_canonical_492_of_493_is_incomplete(tmp_path: Path) -> None:
    calls = 0

    def generate(_record):
        nonlocal calls
        calls += 1
        if calls == 493:
            raise RuntimeError("stop at exact boundary")
        return pixels()

    with pytest.raises(RuntimeError, match="exact boundary"):
        r11.resume_candidate_generation(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            generate=generate,
        )
    assert (
        len(r11.read_jsonl(r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest))
        == 492
    )
    with pytest.raises(ValueError, match="exactly 493"):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)


def test_predictions_bind_png_evaluator_and_canonical_ground_truth(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    metrics = r11.recompute_candidate_metrics(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        evaluator_identity=EVALUATOR,
        predict=perfect_predictor(small_contract),
    )
    assert metrics["id_maa"] == 1.0 and metrics["comp_family_macro_maa"] == 1.0
    rows = r11.read_jsonl(r11.candidate_paths(tmp_path, spec(), 5000).predictions)
    assert all(
        row["png_sha256"]
        and row["evaluator_checkpoint_sha256"] == EVALUATOR_CHECKPOINT_SHA256
        for row in rows
    )
    assert rows[0]["ground_truth"] == small_contract[0]["ground_truth_attributes"]


def test_aggregate_metrics_cannot_override_recomputation(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    r11.recompute_candidate_metrics(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        evaluator_identity=EVALUATOR,
        predict=perfect_predictor(small_contract),
    )
    path = r11.candidate_paths(tmp_path, spec(), 5000).metrics
    altered = r11.read_json(path)
    altered["id_maa"] = 0.123
    r11.atomic_json(path, altered)
    with pytest.raises(ValueError, match="do not match"):
        r11.recompute_candidate_metrics(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            evaluator_identity=EVALUATOR,
            predict=lambda _: pytest.fail("predictions should be reused"),
        )


def test_prediction_evaluator_hash_and_ema_mismatch_rejected(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    bad = {**EVALUATOR, "checkpoint_sha256": "0" * 64}
    with pytest.raises(ValueError, match="evaluator checkpoint"):
        r11.recompute_candidate_metrics(
            output_root=tmp_path,
            spec=spec(),
            milestone=5000,
            checkpoint_sha256=CHECKPOINT_SHA,
            evaluator_identity=bad,
            predict=perfect_predictor(small_contract),
        )
    path = r11.candidate_paths(tmp_path, spec(), 5000).sample_manifest
    rows = r11.read_jsonl(path)
    rows[0]["ema"] = False
    r11.atomic_jsonl(path, rows)
    with pytest.raises(ValueError, match="provenance"):
        r11.validate_candidate_samples(tmp_path, spec(), 5000, CHECKPOINT_SHA)


def test_atomic_json_interruption_leaves_no_valid_final(tmp_path: Path) -> None:
    target = tmp_path / "result.json"

    def fail(_source, _target):
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        r11.atomic_json(target, {"complete": True}, replace=fail)
    assert not target.exists() and not list(tmp_path.glob("*.tmp"))


def test_atomic_jsonl_retries_transient_windows_access_denied(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    calls = 0

    def transient(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            error = PermissionError("access denied")
            error.winerror = 5
            raise error
        os.replace(source, destination)

    r11.atomic_jsonl(target, [{"value": 1}], replace=transient)
    assert calls == 3 and r11.read_jsonl(target) == [{"value": 1}]


def test_atomic_jsonl_reraises_persistent_windows_access_denied(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"

    def permanent(_source: Path, _destination: Path) -> None:
        error = PermissionError("access denied")
        error.winerror = 5
        raise error

    with pytest.raises(PermissionError, match="access denied"):
        r11.atomic_jsonl(target, [{"value": 1}], replace=permanent)
    assert not target.exists() and not list(tmp_path.glob("*.tmp"))


def test_known_operational_temp_recovers_but_unknown_temp_fails(tmp_path: Path) -> None:
    target = tmp_path / "progress.json"
    payload = {
        "schema": "tiny_avatar_diffusion.ca16_r11_progress_v1",
        "completed_candidates": [],
    }
    temporary = r11._operational_temp(target)
    r11.atomic_json(temporary, payload)
    # atomic_json adds its own staging suffix, so move the valid final payload to
    # the exact operational name used after a process death.
    nested = r11._operational_temp(temporary)
    if nested.exists():
        nested.replace(temporary)
    r11.recover_operational_temps(tmp_path, mutate=True)
    assert r11.read_json(target) == payload and not temporary.exists()
    unknown = tmp_path / "arbitrary.tmp"
    unknown.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown temporary"):
        r11.recover_operational_temps(tmp_path, mutate=False)


def test_arbitrary_sample_pending_file_is_invalid_or_unproven(tmp_path: Path) -> None:
    manifest = {"schema": r11.OUTPUT_SCHEMA}
    r11.atomic_json(tmp_path / "run_manifest.json", manifest)
    transaction_dir = tmp_path / "candidates" / spec().run_id / "05000" / "transactions"
    transaction_dir.mkdir(parents=True)
    (transaction_dir / "arbitrary.png.pending").write_bytes(b"not a proven PNG")
    assert (
        r11.classify_output_root(tmp_path, manifest)
        is r11.OutputState.INVALID_OR_UNPROVEN
    )


def test_predictions_survive_metrics_crash_without_evaluator_rerun(
    tmp_path: Path, small_contract
) -> None:
    populate_candidate(tmp_path, small_contract)
    original = r11.recompute_candidate_metrics(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        evaluator_identity=EVALUATOR,
        predict=perfect_predictor(small_contract),
    )
    paths = r11.candidate_paths(tmp_path, spec(), 5000)
    predictions_hash = r11.sha256_file(paths.predictions)
    paths.metrics.unlink()
    resumed = r11.recompute_candidate_metrics(
        output_root=tmp_path,
        spec=spec(),
        milestone=5000,
        checkpoint_sha256=CHECKPOINT_SHA,
        evaluator_identity=EVALUATOR,
        predict=lambda _path: pytest.fail("evaluator inference reran"),
    )
    assert resumed == original
    assert r11.sha256_file(paths.predictions) == predictions_hash


def synthetic_inventory() -> dict:
    return {
        "entries": [
            {
                "run_id": spec().run_id,
                "architecture_arm": "ca8",
                "training_seed": 2026,
                "milestone": milestone,
                "checkpoint_path": f"m{milestone}.pt",
                "sha256": CHECKPOINT_SHA,
                "checkpoint_schema": r11.CA16_CHECKPOINT_SCHEMA,
                "training_git_sha": r11.TRAINING_GIT_SHA,
            }
            for milestone in r11.MILESTONE_ORDER
        ]
    }


def full_synthetic_inventory() -> dict:
    return {
        "entries": [
            {
                "run_id": run_spec.run_id,
                "architecture_arm": run_spec.architecture_arm.value,
                "training_seed": run_spec.training_seed,
                "milestone": milestone,
                "checkpoint_path": f"{run_spec.run_id}_{milestone}.pt",
                "sha256": CHECKPOINT_SHA,
                "checkpoint_schema": r11.CA16_CHECKPOINT_SCHEMA,
                "training_git_sha": r11.TRAINING_GIT_SHA,
            }
            for run_spec in r11.ca16_run_matrix()
            for milestone in r11.MILESTONE_ORDER
        ]
    }


def test_kid_uses_existing_comp_artifacts_and_ranking_selects_one(
    tmp_path: Path,
    small_contract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for milestone in r11.MILESTONE_ORDER:
        populate_candidate(tmp_path, small_contract, milestone)
        r11.recompute_candidate_metrics(
            output_root=tmp_path,
            spec=spec(),
            milestone=milestone,
            checkpoint_sha256=CHECKPOINT_SHA,
            evaluator_identity=EVALUATOR,
            predict=perfect_predictor(small_contract),
        )
    monkeypatch.setattr(
        r11,
        "sample_ema_record",
        lambda *_args, **_kwargs: pytest.fail("KID/ranking invoked generation"),
    )
    calls = []
    selection = r11.select_run(
        output_root=tmp_path,
        spec=spec(),
        inventory=synthetic_inventory(),
        evaluator_identity=EVALUATOR,
        kid_compute=lambda candidate, artifacts: (
            calls.append((candidate.milestone_step, artifacts))
            or {"status": "COMPUTED", "value": candidate.milestone_step / 10000}
        ),
    )
    assert calls and all(
        len(artifacts) == 4
        and all(Path(item.artifact_path).is_file() for item in artifacts)
        for _, artifacts in calls
    )
    assert sum(row["selected"] for row in selection["candidates"]) == 1
    assert selection["selected_milestone"] == 5000


def test_full_precision_ranking_and_frozen_kid_constants() -> None:
    candidates = [
        r11.SelectionCandidate(
            "r", "ca8", 2026, step, 0.5 + (1e-15 if step == 10000 else 0), 0.2, 0.3
        )
        for step in r11.MILESTONE_ORDER
    ]
    assert r11.rank_candidates(candidates)[0].milestone_step == 10000
    assert r11.CLEAN_FID_VERSION == "0.1.35"
    assert r11.compute_comp_kid_from_artifacts.__module__.endswith("ca16_evaluation")


def test_completion_rejected_before_full_state(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        r11.validate_completion_state(
            output_root=tmp_path,
            inventory=synthetic_inventory(),
            evaluator_identity=EVALUATOR,
            evaluation=r11.GitState("future", "exp/ca16-ablation", True),
        )


def test_valid_synthetic_state_can_complete(
    tmp_path: Path, small_contract, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r11, "ca16_run_matrix", lambda: (spec(),))
    monkeypatch.setattr(r11, "INVENTORY_PATH", tmp_path / "inventory.json")
    r11.atomic_json(r11.INVENTORY_PATH, synthetic_inventory())
    manifest = {"schema": r11.OUTPUT_SCHEMA, "evaluation_git_sha": "future"}
    r11.atomic_json(tmp_path / "run_manifest.json", manifest)
    for milestone in r11.MILESTONE_ORDER:
        populate_candidate(tmp_path, small_contract, milestone)
        r11.recompute_candidate_metrics(
            output_root=tmp_path,
            spec=spec(),
            milestone=milestone,
            checkpoint_sha256=CHECKPOINT_SHA,
            evaluator_identity=EVALUATOR,
            predict=perfect_predictor(small_contract),
        )
    r11.select_run(
        output_root=tmp_path,
        spec=spec(),
        inventory=synthetic_inventory(),
        evaluator_identity=EVALUATOR,
        kid_compute=lambda candidate, artifacts: {
            "status": "COMPUTED",
            "value": candidate.milestone_step / 10000,
        },
    )
    marker = r11.write_completion_marker(
        output_root=tmp_path,
        inventory=synthetic_inventory(),
        evaluator_identity=EVALUATOR,
        evaluation=r11.GitState("future", "exp/ca16-ablation", True),
        verify_checkpoint_files=False,
    )
    assert marker["status"] == "COMPLETE" and (tmp_path / "R11_COMPLETE.json").is_file()
    validated = r11.validate_existing_completion(
        output_root=tmp_path,
        inventory=synthetic_inventory(),
        evaluator_identity=EVALUATOR,
        evaluation=r11.GitState("future", "exp/ca16-ablation", True),
        verify_checkpoint_files=False,
    )
    assert validated == marker


def test_completion_requires_exactly_24_candidates_and_6_selections(
    tmp_path: Path,
    small_contract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = full_synthetic_inventory()
    monkeypatch.setattr(r11, "INVENTORY_PATH", tmp_path / "inventory.json")
    r11.atomic_json(r11.INVENTORY_PATH, inventory)
    r11.atomic_json(
        tmp_path / "run_manifest.json",
        {"schema": r11.OUTPUT_SCHEMA, "evaluation_git_sha": "future"},
    )
    for run_spec in r11.ca16_run_matrix():
        for milestone in r11.MILESTONE_ORDER:
            populate_candidate(tmp_path, small_contract, milestone, run_spec)
            r11.recompute_candidate_metrics(
                output_root=tmp_path,
                spec=run_spec,
                milestone=milestone,
                checkpoint_sha256=CHECKPOINT_SHA,
                evaluator_identity=EVALUATOR,
                predict=perfect_predictor(small_contract),
            )
        r11.select_run(
            output_root=tmp_path,
            spec=run_spec,
            inventory=inventory,
            evaluator_identity=EVALUATOR,
            kid_compute=lambda candidate, _artifacts: {
                "status": "COMPUTED",
                "value": candidate.milestone_step / 10000,
            },
        )

    evaluation = r11.GitState("future", "exp/ca16-ablation", True)
    missing_metrics = r11.candidate_paths(
        tmp_path, tuple(r11.ca16_run_matrix())[-1], 20000
    ).metrics
    metrics_bytes = missing_metrics.read_bytes()
    missing_metrics.unlink()
    with pytest.raises(ValueError, match="metrics artifact"):
        r11.validate_completion_state(
            output_root=tmp_path,
            inventory=inventory,
            evaluator_identity=EVALUATOR,
            evaluation=evaluation,
            verify_checkpoint_files=False,
        )
    missing_metrics.write_bytes(metrics_bytes)

    missing_selection = (
        tmp_path / "selections" / f"{tuple(r11.ca16_run_matrix())[-1].run_id}.json"
    )
    selection_bytes = missing_selection.read_bytes()
    missing_selection.unlink()
    with pytest.raises(FileNotFoundError):
        r11.validate_completion_state(
            output_root=tmp_path,
            inventory=inventory,
            evaluator_identity=EVALUATOR,
            evaluation=evaluation,
            verify_checkpoint_files=False,
        )
    missing_selection.write_bytes(selection_bytes)

    summary = r11.validate_completion_state(
        output_root=tmp_path,
        inventory=inventory,
        evaluator_identity=EVALUATOR,
        evaluation=evaluation,
        verify_checkpoint_files=False,
    )
    assert summary["candidate_count"] == 24
    assert summary["selection_count"] == 6


def test_run_manifest_records_two_git_identities() -> None:
    manifest = r11.build_run_manifest(
        evaluation=r11.GitState("future", "exp/ca16-ablation", True),
        inventory_sha256="f" * 64,
        environment={"clean_fid": "0.1.35", "python": "3.12", "torch": "2.5.1"},
        timestamp="fixed",
    )
    assert manifest["evaluation_git_sha"] == "future"
    assert manifest["expected_training_git_sha"] == r11.TRAINING_GIT_SHA
    assert manifest["generation"]["weight_set"] == "EMA"
