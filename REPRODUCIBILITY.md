# Reproducibility Guide

This document describes the frozen Tiny Avatar Diffusion V1 repository as implemented. Displayed metrics use reporting precision; canonical serialized values remain unchanged in the frozen evidence artifacts.

## 1. Reproducibility scope

Three activities must not be confused:

1. **Software reproduction** — install and use the package, CLI, Streamlit application, inference API, and notebooks.
2. **Frozen-evidence verification** — validate existing canonical checkpoints and DR04/DR05 evidence without expensive recomputation.
3. **Full experimental recomputation** — rebuild DR01 artifacts, train models, regenerate evaluation populations, and rerun metrics. These workflows are expensive and write scientific artifacts.

The final clean-release audit procedure is documented for later execution. This guide does not claim that a clean-clone audit has already passed.

### CA16 closure boundary

The completed CA16 CA8-versus-CA8+CA16 comparison is documented in [the closure record](docs/experiments/ca16_ablation/FINAL_CLOSURE.md). A clean clone can inspect its tracked protocols, source, and documentation, but cannot reproduce R10–R14 results without the local evidence archive: the six R10 checkpoint roots and the R11–R14 result roots are intentionally ignored by Git. They require the original CUDA-capable environment and the manually acquired datasets/evaluator asset; they are not restored by `pip install`.

### H7 continuation and reporting boundary

The completed H7 CA8+CA16 continuation extends the three selected CA16 20k checkpoints to 40k successful updates for seeds 2026, 2027, and 2028. Its frozen R11 semantic results and corrected R13 batch-8 KID/FID results are summarized in [the H7 experiment report](docs/experiments/h7_extended_training.md); CA16 examiner-facing context remains in [the CA16 closure record](docs/experiments/ca16_ablation/FINAL_CLOSURE.md).

**CPU-only snapshot reproduction** needs Python with Matplotlib, but no PyTorch, dataset, checkpoint, CA16 root, or H7 result root:

```powershell
python scripts/generate_h7_report.py --snapshot docs/experiments/data/h7_results_v1.json --figures outputs/h7_report_figures
```

This renders figures from the tracked lightweight snapshot; it does not reproduce training or scientific evaluation. To read-only verify that snapshot against frozen external artifacts, supply explicit roots:

```powershell
python scripts/generate_h7_report.py --snapshot docs/experiments/data/h7_results_v1.json --figures outputs/h7_report_figures --verify --ca16-root <CA16-root> --h7-root <H7-root>
```

Strict verification fails on an identity or metric mismatch and never substitutes the snapshot. Full H7 training, sampling, R11, R13, KID, and FID reproduction requires the external historical CA16/H7 checkpoints, result roots, Cartoon Set assets, evaluator/metric assets, and CUDA where the corresponding frozen workflow requires it; those assets are intentionally not Git-shipped.

## 2. Repository and software requirements

- Python: `>=3.11`
- Core: Pillow and PyTorch `2.5.1`
- Demo extra: Streamlit `>=1.32,<2`
- Notebook extra: matplotlib, nbformat, nbconvert, and ipykernel
- Evaluation extra: clean-fid `0.1.35`, LPIPS `0.1.4`, and SciPy `1.15.3`
- Development extra: pytest and Ruff

These requirements come from `pyproject.toml`. Frozen evaluation-provenance versions below are not additional minimum install requirements.

## 3. Environment setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,demo,eval]"
python -c "import tiny_avatar_diffusion; print('package import: OK')"
tiny-avatar --help
```

On Linux or macOS, activate with `source .venv/bin/activate`. A minimal inference install can use `python -m pip install -e .`; add only the extras needed for the intended workflow.

## 4. Data acquisition and expected layout

The project uses the official **Google Cartoon Set** 10k and 100k archives. The repository contains no automatic downloader. Obtain the archives manually from the [official Google Cartoon Set download page](https://google.github.io/cartoonset/download.html), retaining these names and locations:

```text
data/raw/cartoonset/archives/
├── cartoonset10k.tgz
└── cartoonset100k.tgz
```

`src/tiny_avatar_diffusion/dr01_split.py` opens both archives directly, expects paired PNG/CSV members, validates exactly 10,000 and 100,000 identities, and fails if an archive is missing or malformed. It does not extract the archives permanently. The official page and the repository semantic-mapping audit are the available acquisition references; no downloader or additional licensing statement is encoded in project code.

## 5. DR01 — data preparation and frozen protocol

### Frozen task contract

- Attributes, in order: `face_shape`, `glasses`, `hair_color`, `face_color`.
- Caption: `avatar <face_shape> <glasses> <hair_color> <face_color>`.
- Tokenizer: lowercase normalized whitespace, word-level whitespace splitting, atomic underscores.
- Special IDs: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`.
- Vocabulary: TRAIN captions only; frozen 10k size 45.
- Maximum serialized length: 7 (`<bos>`, five lexical tokens, `<eos>`).
- Split seed: 2026; precedence: `FINAL_OOD`, `COMP_VAL`, `ID_POOL`.

| Frozen 10k role | Count |
|---|---:|
| TRAIN | 8,388 |
| ID_VAL | 466 |
| ID_TEST | 466 |
| COMP_VAL | 338 |
| FINAL_OOD | 342 |

Held-out definitions preserve individual tokens in TRAIN while excluding these combinations:

| Role | ID | Held-out pair |
|---|---|---|
| COMP_VAL | CV1 | `face_shape=0` (`pointed`) + `glasses=0` (`oversized_glasses`) |
| COMP_VAL | CV2 | `face_shape=2` (`oval`) + `hair_color=3` (`red`) |
| COMP_VAL | CV3 | `glasses=7` (`oval_glasses`) + `face_color=2` (`medium_brown`) |
| COMP_VAL | CV4 | `hair_color=2` (`orange`) + `face_color=9` (`light_peach`) |
| FINAL_OOD | OOD1 | `face_shape=2` (`oval`) + `glasses=3` (`rimless_glasses`) |
| FINAL_OOD | OOD2 | `face_shape=4` (`round`) + `hair_color=6` (`dark_brown`) |
| FINAL_OOD | OOD3 | `glasses=10` (`heart_glasses`) + `face_color=10` (`pale_pink`) |
| FINAL_OOD | OOD4 | `hair_color=0` (`light_blond`) + `face_color=0` (`deep_brown`) |

### Verify frozen DR01 artifacts

Canonical files are under `data/processed/dr01_v1/`: `split_summary.json`, `heldout_combinations.json`, five manifests per dataset, `semantic_mapping.json`, `ui_prompt_contract.json`, and the `text/` tokenizer/vocabulary/tokenized manifests. Notebook `notebooks/01_data_and_protocol.ipynb` performs read-only checks of counts, held-out definitions, tokenizer behavior, and leakage assertions. Caption/tokenizer tests are safe:

```powershell
python -m pytest tests/test_caption_tokenizer.py
```

### Regenerate DR01 artifacts — full recomputation only

```powershell
python -m tiny_avatar_diffusion.dr01_split
python -m tiny_avatar_diffusion.dr01_text
```

These real entry points require both official archives and rewrite processed/report outputs. Run them only in an isolated full-recomputation checkout, not over frozen artifacts during verification.

## 6. DR02 — model and diffusion configuration

| Component | Frozen configuration |
|---|---|
| Image | RGB, 32×32, pixel space |
| U-Net stages | 64 channels at 32×32; 128 at 16×16 and 8×8 |
| Residual processing | Two ResBlocks at each encoder, bottleneck, and decoder stage |
| Normalization / activation | GroupNorm with 32 groups / SiLU |
| Resampling | Strided-convolution downsampling; nearest resize plus convolution upsampling; skip connections |
| Visual self-attention | None |
| Text encoder | Vocabulary 45; length 7; width 64; two pre-LN Transformer layers; four heads; FFN 256; GELU; sinusoidal positions; dropout 0.1 |
| Conditioning | Image-query/text-key-value cross-attention at 8×8; width 64; four heads |
| Diffusion | `T=1000`; linear beta `1e-4` to `0.02` |
| Objective | Epsilon-prediction MSE |
| Reverse process | Ancestral DDPM; fixed-small posterior variance; predicted-x0 clipping |
| CFG | Not used in MAIN |
| Conditional parameters | 4,723,907 |
| Unconditional parameters | 4,595,779 |

The source of truth is `unet.py`, `text_conditioning.py`, and `diffusion.py`. Notebook 02 supplies CPU-safe shape, parameter, and schedule checks without full sampling.

## 7. DR03 — conditional training and checkpoint selection

Conditional MAIN is fixed to Google Cartoon Set 10k TRAIN (8,388 records), seed 2026, batch 32, no accumulation, AdamW at `2e-4` with zero weight decay, CUDA AMP FP16, EMA `0.9999`, validation every 1,000 successful updates, milestones at 5k/10k/15k/20k, and 20,000 successful updates.

Actual conditional training command:

```powershell
python -m tiny_avatar_diffusion.conditional_main_training --run-id dr03_conditional_main_10k_seed2026_r2 --max-steps 20000
```

It writes to `reports/dr03_main_training/<run-id>/`, logs training/validation JSONL, and saves `latest.pt`, milestones, summaries, and diagnostic EMA previews. It requires CUDA. Phase 6 verified its parser only and did not execute training. The released conditional checkpoint selects EMA weights at exactly step 20,000.

The unconditional baseline used this controlled real launcher:

```powershell
python scripts/dr05_unconditional_main_r1.py
```

It starts a fresh 20,000-successful-update CUDA AMP run and refuses to reuse its authoritative run directory. It was not executed during documentation work.

## 8. Resume and checkpoint semantics

Conditional resume uses the same run ID and an explicit checkpoint:

```powershell
python -m tiny_avatar_diffusion.conditional_main_training --run-id dr03_conditional_main_10k_seed2026_r2 --max-steps 20000 --resume-from reports/dr03_main_training/dr03_conditional_main_10k_seed2026_r2/latest.pt
```

The loader validates schema, run ID, and full frozen configuration before strict loading. The checkpoint stores raw text-encoder and denoiser state, EMA, optimizer, GradScaler, successful step, images seen, skipped updates, data permutation/offset, sampler and corruption-generator state, process RNG state, losses, validation history, and manifest provenance.

Conditional and unconditional MAIN write unique sibling temporary files, flush and `fsync`, then atomically replace the destination. Transient Windows sharing/access failures receive bounded retries. Unconditional resume supports an observational continuity audit over model, EMA, optimizer, scaler, counters, order, RNG, configuration, and provenance. The evaluator has separate epoch-based resume with strict identity validation, optimizer state, completed epoch, global step, selection state, and history consistency.

## 9. DR04 — evaluator and compositional evaluation

The evaluator is separate from the generator and is trained only on real Cartoon Set 100k data with deterministic auxiliary train/validation/ceiling roles. It is a 32→64→128 CNN, global-average-pooled to 128 features, with four attribute heads and 293,064 parameters. The selected checkpoint is epoch 41.

```powershell
python -m tiny_avatar_diffusion.evaluator_main --preflight-only
python -m tiny_avatar_diffusion.evaluator_main
```

The first command validates the real cache/configuration; the second performs the 50-epoch campaign. Their parser was inspected, but neither campaign was run in Phase 6.

The canonical panel contains 252 prompts × five seeds (`2026`–`2030`) = 1,260 samples: 760 ID slots and 500 OOD slots. Its full campaign entry point is:

```powershell
python scripts/dr04_final_conditioning_evaluation.py --mode final --confirm-final-conditioning-eval
```

| DR04 result | Frozen value |
|---|---:|
| ID MAA | 0.5224 |
| ID marginal bootstrap 95% interval | [0.4984, 0.5464] |
| OOD MAA | 0.4250 |
| OOD marginal bootstrap 95% interval | [0.4010, 0.4490] |
| Absolute ID→OOD gap | 0.0974 |
| MacroPSR | 0.2240 |

The bootstrap uses 10,000 percentile resamples with prompt-clustered units retaining all five seeds. The intervals quantify ID and OOD MAA separately. **No canonical bootstrap confidence interval exists for the ID→OOD gap; it is a point estimate only, with no significance claim.**

CLI/UI evaluator output is an uncalibrated **live single-image diagnostic**, not canonical dataset-level MAA or MacroPSR.

## 10. DR05 — quality, diversity, and engineering

**QUALITY != CONDITIONING.** KID/FID describe distribution-level quality; MAA/MacroPSR describe requested-attribute adherence.

Lower KID/FID is better under the frozen protocol:

| Model | Scope | KID | FID |
|---|---|---:|---:|
| Conditional | ID | 0.019296 | 49.38 |
| Conditional | OOD | 0.019467 | 58.70 |
| Unconditional | ID | 0.028688 | 58.08 |
| Unconditional | OOD | 0.043785 | 83.18 |

Global conditional/unconditional values are KID `0.018366` / `0.031913` and FID `41.00` / `53.20`. Frozen global prompt-macro LPIPS for conditional samples is `0.0362`; higher means greater perceptual variation only under this protocol and is not universally better.

| Model | Batch-1 median seconds/image | Batch-1 peak allocated MiB | Batch-16 median seconds/image | Batch-16 peak allocated MiB |
|---|---:|---:|---:|---:|
| Conditional | 8.747 | 29.82 | 0.631 | 77.73 |
| Unconditional | 7.568 | 29.31 | 0.552 | 77.02 |

These are environment-specific engineering measurements. The quality environment records Python 3.12.4, PyTorch 2.5.1+cu121, NumPy 2.4.6, SciPy 1.15.3, clean-fid 0.1.35, and LPIPS 0.1.4. Latency/memory used an NVIDIA GeForce RTX 4060 Laptop GPU, CUDA runtime 12.1, and FP32 sampling.

Real full-recomputation entry points include:

```powershell
python scripts/dr05_final_quality_evaluation.py --mode final --confirm-sealed-final-eval
python scripts/dr05_final_diversity_evaluation.py
python -m scripts.dr05_final_parameter_count
python -m scripts.dr05_final_latency_benchmark
python -m scripts.dr05_final_memory_benchmark
```

They write or benchmark scientific outputs, can refuse existing destinations, and are full-recomputation commands rather than verification commands.

## 11. Canonical checkpoints and integrity hashes

These hashes were computed from the actual files. The inference loader also enforces artifact format, role, architecture, parameter count, selected step/epoch, and provenance.

| Role | Logical repository path | SHA256 | Expected metadata |
|---|---|---|---|
| Conditional generator | `artifacts/checkpoints/conditional_main_ema_step20000.pt` | `a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb` | `conditional_main`; EMA; step 20000; 4,723,907 parameters |
| Unconditional generator | `artifacts/checkpoints/unconditional_main_ema_step20000.pt` | `70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e` | `unconditional_main`; EMA; step 20000; 4,595,779 parameters |
| Evaluator | `artifacts/checkpoints/dr04_evaluator_epoch41.pt` | `414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc` | `dr04_evaluator`; epoch 41; 293,064 parameters |

```powershell
Get-FileHash -Algorithm SHA256 artifacts/checkpoints/conditional_main_ema_step20000.pt,artifacts/checkpoints/unconditional_main_ema_step20000.pt,artifacts/checkpoints/dr04_evaluator_epoch41.pt
```

## 12. Frozen evidence and canonical artifacts

`tiny_avatar_diffusion.frozen_results` consumes exactly these files:

| Role | Logical path | SHA256 | Schema / finalization |
|---|---|---|---|
| DR04 conditioning | `results/runs/dr04_final_conditioning_v1/final_conditioning_summary.json` | `307138ac4aa1d26a0b478ab2a1a1703fa248406187681896f4834147902e26c6` | `dr04_final_conditioning_summary`, v3, final |
| DR05 quality | `results/runs/dr05_final_quality_v1/final_quality_summary.json` | `20d6b20a59908443c3d963e935d06f0eccc907b813dce8ce242d1a2da7b27592` | `dr05_final_quality_main`, v1, final |
| DR05 diversity | `results/runs/dr05_final_diversity_v1/diversity_summary.json` | `85ccd22f592a1ac91a0760895e6b5100b5110b53c2843b2e4076dbc469348e7b` | `dr05_final_diversity_v1` |
| Parameter counts | `results/runs/dr05_final_efficiency_v1/parameter_count_summary.json` | `0d8593c59c5608873b2c7d67240af18024bc4e6427724cd397aed684add24deb` | `tiny_avatar_diffusion.dr05_parameter_count_summary`, v1 |
| Latency | `results/runs/dr05_final_efficiency_v1/latency_summary.json` | `d706c13bb6872971168757ec41a62145bc92e807b2f55673de0c06a5ac9144d4` | `dr05_final_latency_v1` |
| Conditional memory, batch 1 | `results/runs/dr05_final_efficiency_v1/conditional_b1.json` | `302b3b64e0be3a56c8c7e980bb6d33d13db1edd91aa01e3c49911185228f5d89` | `dr05_final_sampling_memory_v1` |
| Unconditional memory, batch 1 | `results/runs/dr05_final_efficiency_v1/unconditional_b1.json` | `7cdd8a645476ce3bb1b5f5b79aa1126cf3fdbe9a4fc928087ce771d346da6c25` | `dr05_final_sampling_memory_v1` |
| Conditional memory, batch 16 | `results/runs/dr05_final_efficiency_v1/conditional_b16.json` | `f870591096618207f10db21ee09cb04451b87508dd6c9706ba1e432084f62a35` | `dr05_final_sampling_memory_v1` |
| Unconditional memory, batch 16 | `results/runs/dr05_final_efficiency_v1/unconditional_b16.json` | `6e8d476bd36c2302905c51b54a75873eb837ee6bfa218d4d60865eb03bccb03d` | `dr05_final_sampling_memory_v1` |

Read-only verification:

```powershell
tiny-avatar results
python -c "from tiny_avatar_diffusion.frozen_results import load_frozen_results; load_frozen_results(); print('frozen evidence: OK')"
```

### CA16 historical evidence and verification

The CA16 execution entry points are present, but are historical campaign workflows rather than clean-clone commands: `scripts/run_ca16_r11_selection.py`, `scripts/run_ca16_r12_evaluation.py`, `scripts/run_ca16_r13_quality.py`, `scripts/run_ca16_r13_fid_secondary.py`, `scripts/run_ca16_r14_diversity.py`, and `scripts/run_ca16_r14_efficiency.py`. They need the local R10 checkpoints, local R11–R14 roots, and CUDA; do not run them merely to inspect the study. Their authoritative local evidence locations and the SHA256 identity/manifests are listed in the [CA16 closure record](docs/experiments/ca16_ablation/FINAL_CLOSURE.md#evidence-and-reproduction).

For a clean clone, the supported read-only checks are the commands in the preceding section, source/protocol inspection, and SHA256 checks of the tracked V1 artifacts. Full CA16 recomputation additionally needs both Cartoon Set archives; the 100k archive and the external Inception asset used by the FID campaign are not Git-shipped. The 10k archive is also an externally supplied raw asset, even when present in a local working directory.

## 13. Inference with released V1

The inference facade supports conditional, unconditional, and paired generation; seeds in `[0, 2**63)`; `auto`, `cuda`, or `cpu`; explicit `x_T`; observational predicted-x0 snapshots; checkpoint/prompt/runtime/pixel provenance; and optional evaluator diagnostics. Paired generation uses the same `x_T` and equivalent post-`x_T` reverse RNG state.

The UI/CLI snapshot schedule is `999, 749, 499, 249, 0`. Initial noise is a clipped display of `x_T`; later images are predicted-x0 observations, not `x_t` states.

## 14. CLI

```powershell
tiny-avatar generate conditional --face-shape pointed --glasses no_glasses --hair-color black --face-color pale_pink --seed 2026 --metadata --save-steps --output outputs/demo/conditional.png --overwrite
tiny-avatar generate unconditional --seed 2026 --metadata --output outputs/demo/unconditional.png --overwrite
tiny-avatar compare --face-shape pointed --glasses no_glasses --hair-color black --face-color pale_pink --seed 2026 --metadata --save-steps --output-dir outputs/demo --overwrite
tiny-avatar results
tiny-avatar demo
```

Important controls are `--seed`, `--device`, `--metadata`, `--save-steps`, `--overwrite`, `--output`/`--output-dir`, conditional `--no-evaluate`, and unconditional `--evaluate`. Use `tiny-avatar <command> --help` for exhaustive syntax.

## 15. Streamlit demo

```powershell
python -m pip install -e ".[demo]"
tiny-avatar demo
```

The tabs are **Generate**, **Compare**, and **Frozen Results**. Images, metadata, and optional steps are in-memory downloads; the application does not automatically write project files. Evaluator scores are uncalibrated, and paired comparison is qualitative. Installed app construction was smoke-tested in the development environment, but the final clean-release audit has not yet been performed.

## 16. Scientific notebooks

| Notebook | Default behavior |
|---|---|
| `notebooks/01_data_and_protocol.ipynb` | Read-only DR01 protocol and leakage inspection. |
| `notebooks/02_model_and_diffusion.ipynb` | CPU-safe architecture, shape, parameter, and schedule checks. |
| `notebooks/03_evaluation_and_results.ipynb` | Analysis through the validated frozen-results loader. |
| `notebooks/04_interactive_generation.ipynb` | Inference contract; live generation and paired comparison default to `False`. |

All four passed independent fresh-kernel default Run All during Phase 5. No training or expensive generation/metric/benchmark campaign occurs by default.

## 17. Expected outputs and protected directories

| Directory | Role | Demo/user writes allowed? |
|---|---|---|
| `data/raw/` | Manually supplied source archives | Deliberate dataset placement only |
| `data/processed/` | Canonical protocols/manifests | No |
| `artifacts/` | Canonical inference checkpoints | No |
| `results/` | Canonical runs and evidence | No |
| `reports/` | Training/protocol records | No |
| `notebooks/` | Canonical executed lab books | No automatic CLI/UI writes |
| `outputs/demo/` | User/demo outputs; created on demand | Yes |

The CLI enforces protected boundaries and refuses existing outputs unless `--overwrite` is explicit. Streamlit uses in-memory downloads.

## 18. Hardware and runtime provenance

- Conditional DR03 launch records configuration but not the training GPU model; none is inferred.
- Unconditional MAIN records an NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes VRAM, CUDA 12.1, driver 595.79, PyTorch 2.5.1+cu121, and AMP FP16.
- DR05 latency/memory records the same GPU family and CUDA runtime with FP32 ancestral sampling.
- User inference may use CPU or another CUDA device. Frozen timing, memory, and bit patterns are not universal guarantees.

## 19. Final clean-release audit procedure

This procedure is intended for the later release audit and is not a completed claim:

1. Fresh checkout at the intended release commit.
2. New isolated environment and extras installation.
3. Package import plus module and console CLI checks.
4. `tiny-avatar results` and focused frozen-results/inference tests.
5. Checkpoint/evidence hash verification.
6. Fresh-kernel default execution of all canonical notebooks.
7. Installed Streamlit shell smoke test without canonical generation.
8. Safe regression suite.
9. Git-state and protected-artifact non-mutation checks.

## 20. Troubleshooting

- **`tiny-avatar` not found:** activate the environment and rerun `python -m pip install -e ".[demo]"`; module equivalent: `python -m tiny_avatar_diffusion.cli`.
- **PowerShell activation blocked:** follow local execution-policy rules or invoke `.venv\Scripts\python.exe` directly.
- **Streamlit missing:** install `.[demo]`.
- **CUDA unavailable:** use `--device cpu` or `--device auto`; explicit unavailable CUDA fails clearly.
- **Missing checkpoint/hash mismatch:** restore the exact canonical artifact; do not bypass validation.
- **Missing/malformed evidence:** restore the documented JSON; `tiny-avatar results` intentionally fails.
- **Output exists:** choose another path or pass `--overwrite` deliberately.
- **Invalid seed:** use an integer in `[0, 2**63)`.
- **Windows pytest temporary-directory permission issue:** use a repository-local `--basetemp`, then remove it after verifying its resolved path.

## 21. Reproducibility boundaries

Supported guarantees include checkpoint/evidence identity checks, explicit seeded sampling semantics, validated loading, recorded software provenance, and fresh-kernel default notebook execution.

The project does **not** claim bit-exact equivalence across arbitrary stacks; that quick verification recomputes KID/FID/LPIPS/bootstrap/benchmarks; robust compositional control; calibrated evaluator probabilities; equivalence of one-image diagnostics and dataset-level MAA/MacroPSR; a confidence interval or significance test for the ID→OOD gap; or completion of the final clean-clone audit.
