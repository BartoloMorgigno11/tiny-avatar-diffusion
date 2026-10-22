# Reproducibility Guide

This guide distinguishes what can be verified from the distributed academic release from historical experimental workflows. Displayed values use reporting precision; canonical serialized values remain in the frozen evidence artifacts.

## Public release identities

The only public model names are **DDPM-U32**, **DDPM-C32**, **DDPM-U64**, and **DDPM-C64**. Their canonical checkpoint paths, architectures, adapters, source provenance, and SHA-256 values are immutable fields in [`src/tiny_avatar_diffusion/public_model_manifest.json`](src/tiny_avatar_diffusion/public_model_manifest.json). Historical identifiers appear only inside explicitly historical source and evidence records; they are not public model selections.

## 1. Public verification

This capability works from the distributed repository and its four public EMA checkpoints. Independent clean-clone acceptance passed: fresh installation, SHA-256 verification and CPU loading of all four checkpoints, 18 selected automated tests, CPU execution of all four final notebooks, Streamlit HTTP/interface smoke tests, and one end-to-end DDPM-U32 CPU PNG generation.

### Installation

Python 3.11 or later is required. Core inference dependencies are Pillow, PyTorch 2.5.1, and NumPy; see [`pyproject.toml`](pyproject.toml).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
tiny-avatar-public --help
```

Install only the optional extra needed for an activity:

```powershell
python -m pip install -e ".[notebook]"  # Matplotlib and notebook support
python -m pip install -e ".[demo]"      # Streamlit
python -m pip install -e ".[dev]"       # pytest and Ruff
python -m pip install -e ".[eval]"      # historical evaluation tooling
```

A platform-appropriate PyTorch/CUDA installation is required for GPU use. The project has no lockfile.

### Verify public checkpoint identities

The four public packages are the only current release checkpoints:

| Public model | Checkpoint |
|---|---|
| DDPM-U32 | `artifacts/checkpoints/ddpm-u32-ema-40k.pt` |
| DDPM-C32 | `artifacts/checkpoints/ddpm-c32-ema-40k.pt` |
| DDPM-U64 | `artifacts/checkpoints/ddpm-u64-ema-60k.pt` |
| DDPM-C64 | `artifacts/checkpoints/ddpm-c64-ema-60k.pt` |

```powershell
Get-FileHash -Algorithm SHA256 artifacts\checkpoints\ddpm-u32-ema-40k.pt,artifacts\checkpoints\ddpm-c32-ema-40k.pt,artifacts\checkpoints\ddpm-u64-ema-60k.pt,artifacts\checkpoints\ddpm-c64-ema-60k.pt
```

Compare the values with the public model manifest. Do not substitute historical checkpoint names for these packages.

### Public CLI and Streamlit

The public CLI accepts only the four public names:

```powershell
tiny-avatar-public --model DDPM-C32 --face-shape wide --glasses thick_rectangular_glasses --hair-color silver --face-color deep_brown --seed 2026 --device cpu --output outputs\demo\ddpm-c32_seed2026.png
```

Sampling uses 1,000 ancestral DDPM steps and can be slow on CPU.

```powershell
python -m pip install -e ".[demo]"
python -m streamlit run src/tiny_avatar_diffusion/final_streamlit_app.py
```

The UI verifies a checkpoint hash before loading it. It is an inference demonstration, not a benchmark runner.

### Final notebooks and selected regression checks

The four professor-facing notebooks are in `notebooks/final/`. Their default cells use packaged metadata, frozen summaries, figures, and qualitative PNGs on CPU; optional regeneration cells are disabled.

```powershell
python -m pytest tests/test_import.py tests/test_caption_tokenizer.py tests/test_inference.py tests/test_cli.py tests/test_final_streamlit_app.py -q
```

These are selected regression checks, not a claim that every historical campaign test is runnable from a clean clone.

## 2. Frozen scientific evidence

Frozen evidence is inspected without rerunning training, sampling, FID/KID, LPIPS, evaluator campaigns, or benchmarks.

- `data/processed/dr01_v1/` preserves split, tokenizer, vocabulary, semantic mapping, and compositional-OOD provenance.
- `results/runs/dr04_final_conditioning_v1/` and `results/runs/dr05_final_*/` preserve historical conditioning, quality, diversity, and efficiency summaries.
- `results/runs/v2a_step9b_final_supplementary_ema60k_v1/` and `results/runs/step13_final_unconditional_benchmark_v1/` supply the final-notebook comparison records.
- `docs/release/` documents public artifacts, dataset attribution, benchmark boundaries, and historical decision context.

The final notebooks are the primary scientific walkthrough:

| Notebook | Default scope |
|---|---|
| `01_dataset_and_compositional_ood.ipynb` | Dataset contract, tokenizer, captions, and OOD design |
| `02_ddpm_architecture_and_conditioning.ipynb` | DDPM, U-Net, text conditioning, and public architecture |
| `03_conditional_ablation_study.ipynb` | Historical conditional ablation evidence |
| `04_final_evaluation_and_reproducibility.ipynb` | Final frozen evidence and release boundaries |

Their exact inputs and exclusions are recorded in [`docs/RELEASE_CONTENTS.md`](docs/RELEASE_CONTENTS.md). The public EMA inference packages do not recreate every historical full-state trajectory or optional notebook regeneration path.

For legacy frozen-summary inspection, the original CLI remains available:

```powershell
tiny-avatar results
```

It reads tracked DR04/DR05 summaries; it does not recompute their metrics.

### Scientific interpretation boundaries

Conditional adherence is classifier-derived, not a human evaluation or a calibrated probability of correctness. Conditional and unconditional diversity have different estimands. Their KID/FID, LPIPS, conditioning, and efficiency values are governed by distinct named protocols and must not be reduced to an unqualified four-model ranking.

## 3. Full experimental reproduction

Full reproduction is intentionally broader than public verification. It may create scientific artifacts and requires manually acquired data, CUDA resources, external metric assets, and/or unavailable historical full-state checkpoints.

### Data

The project uses official Google Cartoon Set 10k and 100k archives. They are not redistributed and there is no automatic downloader. Obtain them from the [official Cartoon Set download page](https://google.github.io/cartoonset/download.html) and place them as:

```text
data/raw/cartoonset/archives/cartoonset10k.tgz
data/raw/cartoonset/archives/cartoonset100k.tgz
```

See [dataset attribution](docs/release/DATASET_ATTRIBUTION.md). The project source is Apache-2.0; Cartoon Set is Google LLC material under CC BY 4.0.

### Historical workflows and availability

The DR01–DR05 modules and historical CA16/H7 scripts are retained as provenance for the original research program. They are not public quickstart commands. In particular, historical CA16/H7 continuation, selection, quality, FID/KID, and diversity workflows require original local checkpoint/result roots, datasets, evaluator/metric assets, and CUDA where applicable. The four public EMA inference packages do **not** contain optimizer, scaler, RNG, sampler-progress, or full-state checkpoint information and cannot resume those experiments.

The historical decision record is [`docs/release/HISTORICAL_EVALUATION_AND_DECISION_AUDIT.md`](docs/release/HISTORICAL_EVALUATION_AND_DECISION_AUDIT.md); CA16/H7 records remain in `docs/experiments/`. These documents preserve the historical naming and availability limits rather than presenting those phases as current release instructions.

## Hardware and runtime provenance

Frozen timing and memory measurements were recorded on the documented NVIDIA RTX 4060 Laptop GPU / CUDA environment. User inference can run on CPU or another CUDA device; frozen timing, memory, and bit patterns are not universal guarantees.

## Reproducibility boundaries

The release supports public artifact identity checks, validated checkpoint loading, explicit seeded sampling semantics, CPU-by-default notebook evidence, and inspection of frozen records. It does not claim bit-exact equivalence across arbitrary software stacks, automatic access to CartoonSet or third-party metric assets, clean-clone recomputation of every historical campaign, calibrated evaluator probabilities, or universal generalization from the reported protocols.
