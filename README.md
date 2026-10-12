# Tiny Avatar Diffusion

Tiny Avatar Diffusion is a compact, 32×32, text-conditioned pixel-space DDPM for cartoon avatars. Both the image denoiser and the text encoder are trained from scratch. The frozen V1 study asks whether this small system can follow structured descriptions and generalize to attribute combinations excluded from training.

```text
structured attributes → scratch text encoder → cross-attention
                      → Tiny U-Net DDPM → 32×32 RGB avatar
```

## Study status

- **Original V1 baseline:** the released conditional/unconditional system and its frozen DR01–DR05 evidence, described below.
- **Completed CA16 comparison:** a separate, paired three-seed comparison of bottleneck cross-attention (CA8) with CA8+CA16 on the 10k compositional task. At the selected 20k checkpoints, CA8+CA16 had higher mean R12 ID_TEST MAA (0.7184 vs 0.5854), FINAL_OOD MAA (0.6758 vs 0.5163), and MacroPSR (0.488 vs 0.267). The full scope, provenance, metrics, and limitations are in [the CA16 closure record](docs/experiments/ca16_ablation/FINAL_CLOSURE.md).
- **Available now:** the V1 package, CLI, inference, comparison, and Streamlit demo. CA16 experiment checkpoints and result roots are local-only evidence, not Git-shipped inference assets.
- **Completed H7 continuation:** CA8+CA16 was continued from 20k to 40k successful updates for seeds 2026, 2027, and 2028. Frozen R11 semantic evaluation and R13 batch-8 KID/FID evaluation are complete. A portable CPU-only report with five figure pairs is available in [the H7 experiment report](docs/experiments/h7_extended_training.md). Native 64×64 output remains out of scope; H7 does not select a final project architecture.

## Project at a glance

- Dataset: Google Cartoon Set, with four controlled attributes: `face_shape`, `glasses`, `hair_color`, and `face_color`.
- Prompt grammar: `avatar <face_shape> <glasses> <hair_color> <face_color>`.
- Main model: a 4,723,907-parameter conditional DDPM trained for 20,000 successful updates.
- Generalization test: individual attribute values remain known, while selected pairs of values are held out as unseen compositions.
- Released interfaces: Python package, `tiny-avatar` CLI, optional Streamlit application, and six scientific notebooks.

## Research question

**Can a very small diffusion model, with both denoiser and text encoder trained from scratch, generalize to unseen combinations of avatar attributes?**

## Method overview

| Component | Frozen V1 configuration |
|---|---|
| Resolution | 32×32 RGB |
| Diffusion | Pixel-space ancestral DDPM |
| Prediction target | Epsilon |
| Timesteps | 1,000; linear beta schedule from `1e-4` to `0.02` |
| Image denoiser | Tiny U-Net; channels 64 → 128 → 128; two residual blocks per stage |
| Text encoder | 64-wide, two-layer, four-head scratch Transformer; FFN 256 |
| Conditioning | Image-to-text cross-attention at the 8×8 bottleneck stage |
| Reverse variance | Fixed-small posterior variance, with predicted-x0 clipping |
| Classifier-free guidance | Not used in MAIN |
| Pretrained generator components | None; no pretrained diffusion model, text encoder, or conditioning image encoder |

## Key frozen results

| Conditioning result | Value |
|---|---:|
| ID mean attribute accuracy (MAA) | 0.5224 |
| Compositional OOD MAA | 0.4250 |
| Absolute ID→OOD MAA gap | 0.0974 |
| Macro prompt success rate (MacroPSR) | 0.2240 |

| Model | Global KID ↓ | Global FID ↓ | Parameters |
|---|---:|---:|---:|
| Conditional | 0.018366 | 41.00 | 4,723,907 |
| Unconditional | 0.031913 | 53.20 | 4,595,779 |

**QUALITY != CONDITIONING.** MAA and MacroPSR measure adherence to requested structured attributes. KID and FID measure distribution-level generative quality; they do not establish whether a request was followed.

The frozen evidence supports partial rather than robust compositional control: adherence is moderate and decreases on unseen combinations. Checkpoint and evidence identities are integrity-checked; complete hashes and protocols are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Quick start

Python 3.11 or newer is required. From Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[demo]"
tiny-avatar --help
```

On Linux or macOS, replace the activation command with:

```bash
source .venv/bin/activate
```

Use `tiny-avatar <command> --help` for the complete option surface.

## Generate an avatar

Conditional generation with valid canonical attribute labels:

```powershell
tiny-avatar generate conditional --face-shape pointed --glasses no_glasses --hair-color black --face-color pale_pink --seed 2026 --metadata --output outputs/demo/conditional.png --overwrite
```

Unconditional baseline generation:

```powershell
tiny-avatar generate unconditional --seed 2026 --metadata --output outputs/demo/unconditional.png --overwrite
```

Generation performs the complete frozen 1,000-step sampler. `--device auto` is the default. The CLI refuses demo writes under protected scientific namespaces such as `results/`, `data/processed/`, `artifacts/`, and `reports/`.

## Compare conditional vs unconditional

```powershell
tiny-avatar compare --face-shape pointed --glasses no_glasses --hair-color black --face-color pale_pink --seed 2026 --metadata --output-dir outputs/demo --overwrite
```

This is a **QUALITATIVE PAIRED COMPARISON**: both branches use the same explicit initial Gaussian `x_T` and equivalent post-`x_T` reverse-noise RNG state. It is not a dataset-level metric and does not provide a numerical ranking or causal estimate.

## Interactive demo

```powershell
python -m pip install -e ".[demo]"
tiny-avatar demo
```

The Streamlit application provides three tabs:

- **Generate** — conditional or unconditional inference with optional denoising observations.
- **Compare** — the qualitative paired comparison.
- **Frozen Results** — validated read-only DR04/DR05 evidence.

Downloads are prepared in memory; the application does not automatically write project outputs. Evaluator scores are uncalibrated single-image diagnostics. Installed application construction was smoke-tested in the development environment; the final clean-release audit has not yet been performed.

## Inspect frozen results

```powershell
tiny-avatar results
```

This reads validated frozen DR04/DR05 evidence. It does not train models, generate evaluation panels, recompute bootstrap intervals, or rerun KID, FID, LPIPS, latency, or memory campaigns.

## Scientific notebooks

| Notebook | Purpose |
|---|---|
| [01_data_and_protocol.ipynb](notebooks/01_data_and_protocol.ipynb) | Learning task, captions, tokenizer, splits, held-out combinations, and leakage safeguards. |
| [02_model_and_diffusion.ipynb](notebooks/02_model_and_diffusion.ipynb) | Tiny U-Net, scratch text encoder, cross-attention, and DDPM mechanics. |
| [03_evaluation_and_results.ipynb](notebooks/03_evaluation_and_results.ipynb) | Validated analysis of frozen conditioning, quality, diversity, and engineering evidence. |
| [04_interactive_generation.ipynb](notebooks/04_interactive_generation.ipynb) | Frozen inference contract and optional live trajectory/paired demonstration. |
| [05_ca16_ablation_results.ipynb](notebooks/05_ca16_ablation_results.ipynb) | CPU-only examiner summary of the completed CA8 versus CA8+CA16 study; uses verified compact summaries, not local checkpoints or generated images. |
| [06_h7_extended_training_results.ipynb](notebooks/06_h7_extended_training_results.ipynb) | CPU-only, snapshot-driven analysis of CA8+CA16 extended training from 20k to 40k successful optimizer steps across three seeds. |

Notebooks 01 through 04 passed fresh-kernel default Run All during the Phase-5 notebook release pass. Notebooks 05 and 06 were separately validated through complete CPU-only execution. The fourth notebook keeps live generation and paired comparison disabled by default, so no expensive sampling occurs during its default run.

## Repository structure

```text
src/tiny_avatar_diffusion/   package, models, training, inference, CLI, and UI
notebooks/                   canonical scientific lab books
tests/                       automated regression tests
data/                        source archives, processed protocols, and caches
artifacts/checkpoints/       canonical inference checkpoints
results/                     frozen evaluation runs and evidence
reports/                     training and protocol records
outputs/demo/                user/demo outputs, created on demand
```

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup, source-data layout, DR01–DR05 procedures, training and resume semantics, checkpoint/evidence hashes, inference behavior, notebook verification, filesystem boundaries, and troubleshooting.

## V1 scope and constraints

V1 is deliberately compact and frozen: 32×32 pixel-space diffusion, no pretrained diffusion or text model, no pretrained image encoder for conditioning, no visual self-attention, and no classifier-free guidance in MAIN. Frozen evidence is descriptive and does not establish robust compositional control, statistical significance of the ID→OOD gap, or calibrated correctness probabilities from the evaluator.
