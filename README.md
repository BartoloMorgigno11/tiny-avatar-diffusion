# Tiny Avatar Diffusion

Course project for the **Deep Learning 2026** exam. This repository implements pixel-space denoising diffusion models from scratch for CartoonSet-style avatar generation. The examination question is whether a compact model can generate avatars that follow a structured four-attribute text prompt, while making the limits of the evaluation explicit.

The public release passed an independent clean-clone acceptance: fresh installation, four-checkpoint SHA-256 verification and CPU loading, 18 selected automated tests, CPU execution of the four final notebooks, Streamlit smoke tests, and one end-to-end CPU DDPM-U32 generation.

## Public model family

All endpoints use a compact U-Net denoiser, epsilon prediction, and a 1,000-step linear ancestral DDPM. Conditional models add a scratch-trained Transformer text encoder and cross-attention at the 8x8 and 16x16 stages.

| Public model | Native resolution | Conditioning | Released weights |
|---|---:|---|---|
| DDPM-U32 | 32x32 | Unconditional | EMA 40k |
| DDPM-C32 | 32x32 | Text-conditioned | EMA 40k |
| DDPM-U64 | 64x64 | Unconditional | EMA 60k |
| DDPM-C64 | 64x64 | Text-conditioned | EMA 60k |

The canonical paths, architecture metadata, adapters, and SHA-256 identities are in [`src/tiny_avatar_diffusion/public_model_manifest.json`](src/tiny_avatar_diffusion/public_model_manifest.json).

## Scientific scope

Prompts use the fixed grammar `avatar <face_shape> <glasses> <hair_color> <face_color>`. The project evaluates conditional adherence, distributional quality, diversity, and efficiency under frozen, named protocols.

Conditional adherence is estimated by an independent classifier and is not a human evaluation or calibrated probability of correctness. Conditional and unconditional metrics use different estimands; they must not be collapsed into a single four-model ranking. Full training and metric recomputation require external CartoonSet archives, CUDA resources, and—in some historical workflows—full-state checkpoints that are not distributed here.

## Install

Python 3.11 or later is required. Minimal public inference needs only the core dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
tiny-avatar-public --help
```

Optional extras are installed only for the corresponding activity:

```powershell
# Four scientific notebooks
python -m pip install -e ".[notebook]"

# Streamlit demonstration
python -m pip install -e ".[demo]"

# Selected tests and development tools
python -m pip install -e ".[dev]"

# Historical evaluation tooling; not needed for public inference
python -m pip install -e ".[eval]"
```

PyTorch is pinned in [`pyproject.toml`](pyproject.toml); choose a compatible platform/CUDA build when GPU use is intended. The project has no lockfile.

## Quickstart: public inference

Verify the four packaged artifacts against the manifest before loading them:

```powershell
Get-FileHash -Algorithm SHA256 artifacts\checkpoints\ddpm-u32-ema-40k.pt,artifacts\checkpoints\ddpm-c32-ema-40k.pt,artifacts\checkpoints\ddpm-u64-ema-60k.pt,artifacts\checkpoints\ddpm-c64-ema-60k.pt
```

Generate a conditional native-32 image:

```powershell
tiny-avatar-public --model DDPM-C32 --face-shape wide --glasses thick_rectangular_glasses --hair-color silver --face-color deep_brown --seed 2026 --device cpu --output outputs\demo\ddpm-c32_seed2026.png
```

The sampler performs 1,000 reverse steps and can be slow on CPU. The public CLI accepts only the four names listed above; use `tiny-avatar-public --help` for the supported controls.

## Four-model Streamlit demonstration

```powershell
python -m pip install -e ".[demo]"
python -m streamlit run src/tiny_avatar_diffusion/final_streamlit_app.py
```

The UI validates each public checkpoint hash before loading it. It is an inference demonstration, not a benchmark runner.

## Final scientific walkthrough

The four notebooks in [`notebooks/final/`](notebooks/final/) are the primary professor-facing walkthrough. Their default cells read packaged metadata, frozen summaries, and provenance-recorded visual evidence on CPU; they do not train models, sample images, or recompute metrics.

1. `01_dataset_and_compositional_ood.ipynb` — data contract, captions, and OOD design.
2. `02_ddpm_architecture_and_conditioning.ipynb` — DDPM, U-Net, and conditioning design.
3. `03_conditional_ablation_study.ipynb` — historical conditional ablation evidence.
4. `04_final_evaluation_and_reproducibility.ipynb` — final frozen evidence and release boundaries.

Their inputs and intentional exclusions are listed in [`docs/RELEASE_CONTENTS.md`](docs/RELEASE_CONTENTS.md).

## Reproducibility and repository map

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) distinguishes public verification, frozen scientific evidence, and full experimental reproduction. CartoonSet source archives are not redistributed; see [`docs/release/DATASET_ATTRIBUTION.md`](docs/release/DATASET_ATTRIBUTION.md).

```text
src/tiny_avatar_diffusion/   Models, diffusion, public loaders, CLI, UI
artifacts/checkpoints/       Four public EMA inference packages
data/processed/              Frozen split, tokenizer, and protocol metadata
results/runs/                Frozen summaries and provenance records
notebooks/final/             Professor-facing scientific walkthrough
docs/release/                Release and evidence documentation
```

Historical development scripts, checkpoints, and detailed campaign records are retained for provenance where available. They are not additional public models and should not be treated as clean-clone workflows.

## License and data

Project source code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Cartoon Set data belong to Google LLC under CC BY 4.0 and are not redistributed by this repository. Model artifacts and generated examples have separate provenance and distribution scope; see [`docs/release/MODEL_ARTIFACT_NOTICE.md`](docs/release/MODEL_ARTIFACT_NOTICE.md).
