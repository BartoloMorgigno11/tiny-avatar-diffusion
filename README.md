# Tiny Text-Conditioned Avatar Diffusion from Scratch

Course project for the **Deep Learning 2026** exam. The project studies whether a compact pixel-space diffusion model trained from scratch can generate CartoonSet-style avatars that follow a structured four-attribute prompt.

> **Release status:** professor-facing release assembly is in progress. This repository packages exactly four authenticated public EMA endpoints. Frozen scientific records remain immutable; a clean-clone validation is still required before delivery.

## Objective

Each prompt has the fixed grammar:

```text
avatar <face_shape> <glasses> <hair_color> <face_color>
```

The work evaluates conditional adherence, visual quality, diversity, efficiency, and whether additional cross-attention improves glasses adherence. The dataset is Google Cartoon Set (CartoonSet); source archives are not redistributed by this repository. See [dataset attribution](docs/release/DATASET_ATTRIBUTION.md) and [reproducibility](REPRODUCIBILITY.md).

The canonical attributes are `face_shape` (7 classes), `glasses` (12), `hair_color` (10), and `face_color` (11). Captions use the frozen DR01 vocabulary and tokenizer; labels are never silently aliased.

## Model family

All generators use a scratch text Transformer, a small U-Net denoiser, epsilon prediction, and a 1,000-step linear ancestral DDPM. Cross-attention is from image features to text context.

| Public name | Release endpoint | Resolution | Conditioning | Weights |
|---|---:|---|---:|---|
| DDPM-U32 | 32x32 | unconditional | EMA 40k |
| DDPM-C32 | 32x32 | CA8+CA16 | EMA 40k |
| DDPM-U64 | 64x64 | unconditional | EMA 60k |
| DDPM-C64 | 64x64 | CA8+CA16 | EMA 60k |

The exact paths, SHA-256 identities and adapters are in `src/tiny_avatar_diffusion/public_model_manifest.json`. Experiments A-D are retained as historical/development evidence; they are not silently converted into a new controlled cross-resolution claim.

## Evaluation

The shared benchmark contract fixes prompts, canonical targets, seeds, source-image identities, split membership, and metric definitions across models. Generated samples are evaluated at their native resolution:

- Conditioning: per-attribute accuracy, mean attribute accuracy (MAA), prompt success rate (PSR), MacroPSR, and glasses-specific diagnostics.
- Quality: Clean-FID and KID against source-identical real references at the matching native resolution.
- Diversity: LPIPS AlexNet under a fixed prompt/seed-pair schedule.
- Engineering: parameter count, latency, throughput, and available peak-VRAM measurements.

Evaluator32 and Evaluator64 are separate classifiers. Evaluator64 is required for primary native64 attribute scoring. A native32 evaluator applied after 64-to-32 conversion is a secondary diagnostic only. Development validation—not protected final-test outcomes—will determine whether a final model can be selected. See [the four-model benchmark contract](docs/release/FOUR_MODEL_BENCHMARK_CONTRACT.md) and [native64 evaluator readiness](docs/release/NATIVE64_EVALUATOR_READINESS.md).

## Install

Python 3.11 or later is required. The commands below are repository-relative and suitable for a clean environment, but a final clean-clone installation audit remains pending.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,demo,eval]"
tiny-avatar --help
```

Minimal inference installation:

```powershell
python -m pip install -e .
python -m tiny_avatar_diffusion.cli --help
```

Core dependencies are declared in [pyproject.toml](pyproject.toml). Evaluation additionally requires `clean-fid==0.1.35`, `lpips==0.1.4`, and `scipy==1.15.3`. The project currently has no lockfile and Pillow is not pinned; platform-specific PyTorch/CUDA installation and a fully locked environment remain release work.

## Quickstart: authenticated public inference

The packaged DDPM-C32 EMA checkpoint is at `artifacts/checkpoints/ddpm-c32-ema-40k.pt`. Verify it before use:

```powershell
Get-FileHash -Algorithm SHA256 artifacts\checkpoints\ddpm-c32-ema-40k.pt
# Expected: 438e4672888e6d9177820db5756e32a71649fa1ec16b7ccef11c6e38d2f89944
```

Generate a native32 image:

```powershell
tiny-avatar-public --model DDPM-C32 --face-shape wide --glasses thick_rectangular_glasses --hair-color silver --face-color deep_brown --seed 2026 --device auto --output outputs\demo\ddpm-c32_seed2026.png
```

This CLI route and checkpoint package were previously smoke-tested in the project environment; they have not yet passed the final clean-clone audit. The 1,000-step sampler can be slow on CPU.

## Four-model Streamlit demo

The release UI serves the four packaged endpoints and validates each checkpoint hash before loading it:

```powershell
python -m streamlit run src/tiny_avatar_diffusion/final_streamlit_app.py
```

The 1,000-step sampler can be slow on CPU. The UI is an inference demonstration, not a benchmark runner and does not rewrite frozen records.

## Reproducibility workflows

Lightweight CPU checks, previously used in the project environment and intended for rerun after installation:

```powershell
python -m pytest tests/test_caption_tokenizer.py -q
python -m pytest tests/test_v2_native64_release.py -q
python -m tiny_avatar_diffusion.four_model_benchmark_runner --help
```

The V2 test requires the authenticated sibling-worktree checkpoints above and is expected to skip when they are unavailable. Full training, DDPM sampling, FID/KID, LPIPS, and evaluator campaigns are intentionally not quickstart operations: they require manually acquired CartoonSet archives, authenticated artifacts, and in several cases CUDA. Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for DR01–DR05 frozen-protocol details and release documentation for artifact identities.

To inspect compact evidence without rerunning metrics:

```powershell
tiny-avatar results
Get-ChildItem results\runs
Get-ChildItem docs\release
```

The four professor-facing notebooks are in [`notebooks/final/`](notebooks/final/): dataset/OOD protocol, DDPM architecture, historical A-D ablation context, and final frozen-evidence/reproducibility. They only read saved records by default. Their exact inputs and intentional exclusions are listed in [the release contents manifest](docs/RELEASE_CONTENTS.md).

Development pilot outputs, evaluator64 training records, CA8-40k records, and large generated pools remain local scientific artifacts under `results/`; they are deliberately excluded from ordinary Git distribution.

## Repository map

```text
src/tiny_avatar_diffusion/   Models, DDPM, inference, CLI, evaluators, benchmark tools
tests/                       CPU regression and contract tests
docs/release/                Checkpoint/evidence registry, benchmark and release notes
docs/experiments/            Historical experiment records
notebooks/                   Scientific notebooks (not rewritten by release setup)
data/processed/              Frozen manifests and development contracts (local/protocol scoped)
artifacts/checkpoints/       Packaged inference/evaluator artifacts when approved
results/                     Local training, sampling, and evaluation evidence
```

## Checkpoints and frozen evidence

Every distributed model artifact should have a versioned manifest containing role, architecture, resolution, training step, weight selection (RAW/EMA), source provenance, and SHA-256. Verify files with `Get-FileHash` before loading; the strict V2 loader also performs this check internally.

Release packaging policy:

1. The four public EMA inference packages are included directly with their mandatory SHA-256 identities.
2. Publish essential inference packages separately, beginning with H7 EMA and—after final selection—approved CA8/V2 inference packages with checksummed manifests.
3. Preserve full resumable research checkpoints, optimizer/RNG state, source manifests, evaluator checkpoints, and generated evidence outside this professor-facing package.
4. Treat large metric-network dependencies, for example `inception-2015-12-05.pt`, as explicitly versioned external assets or a reviewed LFS/release-asset decision; do not silently commit duplicates.

`tiny-avatar results` reads the tracked DR04/DR05 frozen summaries without recomputing metrics. Raw CartoonSet archives, generated pools, caches, review packages and metric-network weights are not part of this release.

## Remaining release checks

- Run a clean-environment installation, CPU notebook, CLI and Streamlit audit on the staged content.
- Add the compiled final report PDF manually if the delivery channel requires it; it is not present in this working directory.
- Do not claim a new benchmark result beyond the frozen records and their documented scopes.

## License and data

Project source code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Cartoon Set data belong to Google LLC under CC BY 4.0 and are not redistributed here. Model artifacts and generated examples have separate provenance and distribution scope; see [model artifact notice](docs/release/MODEL_ARTIFACT_NOTICE.md).
