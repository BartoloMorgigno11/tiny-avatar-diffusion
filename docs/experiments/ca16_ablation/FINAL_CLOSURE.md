# CA16 scientific closure

**Frozen baseline:** `82db5255caa86d7ed8201b6dfa90346680e4448d` (`ca16-ablation-r14-diversity-efficiency-v3`).  This document closes the CA16 scientific comparison; it does not select the ultimate project architecture.

## Question, data, and intervention

CA16 asked whether adding a second cross-attention block at the 16x16 U-Net stage (CA8+CA16) improves a small text-conditioned diffusion model's attribute adherence and compositional generalization relative to the original bottleneck-only CA8 design. The task uses Google Cartoon Set 10k with four conditioning attributes: `face_shape`, `glasses`, `hair_color`, and `face_color`. Captions use the canonical four-attribute grammar. The DR01 split keeps every individual token in TRAIN while withholding specified attribute-value pairs, with COMP_VAL used for development and FINAL_OOD held out for the final evaluation. Definitions, precedence, and leakage assertions are in [PREREGISTRATION.md](PREREGISTRATION.md) and [PROTOCOL.md](PROTOCOL.md).

Both arms are 32x32 pixel-space, epsilon-prediction DDPMs with 1,000 linear-schedule timesteps, a scratch Transformer text encoder, and a Tiny U-Net. CA8 uses cross-attention at the 8x8 bottleneck; CA8+CA16 adds cross-attention at 16x16. The parameter counts are 4,723,907 and 4,749,059, respectively: a 25,152-parameter difference. Therefore this is an intervention comparison with a small capacity change, not an isolation of placement alone.

## Training and selection

R10 trained three paired seeds (2026, 2027, 2028) per arm for 20,000 **successful** optimizer updates on the 8,388-record 10k TRAIN split: batch 32, AdamW 2e-4 with zero weight decay, CUDA AMP FP16, EMA 0.9999, and milestones at 5k/10k/15k/20k. Pairing used shared initialization for the corresponding seed. The recorded checkpoint schema preserves raw denoiser/text-encoder weights, their EMA states, optimizer and scaler state, counters/skipped-update bookkeeping, sampler state, corruption generator, process RNG state, configuration, and manifest provenance.

R11 selected the 20k milestone for every run using COMP_VAL only. It was a development selection exercise, so R11 values are not independent final-test evidence and must not be combined with FINAL_OOD. The R11 mean COMP_VAL MAA at 20k was 0.5854 (CA8) and 0.7374 (CA8+CA16); MacroPSR was 0.1060 and 0.2656. The observed checkpoints do not establish a general stopping rule or a learning-curve claim beyond the recorded milestones.

## R12 held-out adherence

R12 evaluated the R11-selected EMA checkpoints on ID_TEST and FINAL_OOD with the frozen evaluator, ancestral FP32 DDPM sampling, 1,000 steps, and batch size 8. The table reports means across the three independently trained seeds. EAM is prompt-level exact attribute match; MacroPSR is the macro mean over the four held-out-pair families.

| Arm | ID_TEST MAA | FINAL_OOD MAA | MacroPSR | ID EAM | FINAL_OOD EAM |
|---|---:|---:|---:|---:|---:|
| CA8 | 0.5854 | 0.5163 | 0.2667 | 0.1118 | 0.0520 |
| CA8+CA16 | 0.7184 | 0.6758 | 0.4880 | 0.2504 | 0.1760 |
| Paired difference (CA8+CA16 - CA8) | +0.1330 | +0.1595 | +0.2213 | +0.1386 | +0.1240 |

The paired FINAL_OOD MAA differences were +0.1435, +0.1625, and +0.1725 for seeds 2026, 2027, and 2028; the MacroPSR differences were +0.102, +0.398, and +0.164. The corresponding FINAL_OOD EAM differences were +0.052, +0.192, and +0.128. These descriptive, three-pair results support the observed CA16-arm advantage under this protocol; they are not confidence intervals, significance tests, or proof of a placement-only causal mechanism. FINAL_OOD is now historically known and cannot be made blind again.

Matched OOD-to-ID controls were a secondary diagnostic. They match the two primitive values of an OOD family against ID prompts where possible, but do not make an interaction-specific causal estimate: contexts, attribute difficulty, the evaluator, and finite prompts remain entangled.

## R13 quality and R14 diversity/efficiency

R13 used the R11 selections on ID_VAL and COMP_VAL. KID was the prespecified distribution-quality guardrail; lower is better. Mean KID was 0.01756/0.02041/0.01827 for CA8 and 0.01713/0.01778/0.01694 for CA8+CA16 (ID/COMP/global). Secondary post-hoc FID was also lower for CA8+CA16: 52.282 vs 53.507 (ID), 60.963 vs 64.030 (COMP), and 42.074 vs 44.231 (global). FID was not the primary endpoint and neither KID nor FID establishes prompt adherence.

The final R14 V3 diversity protocol generated 100 ID and 100 COMP prompts per arm/seed, five sampling seeds per prompt, and estimated LPIPS diversity (AlexNet; 10 pairs per prompt; 10,000 bootstrap resamples). The paired mean LPIPS difference (CA8+CA16 - CA8) was +0.00030 for ID and +0.00204 for COMP. LPIPS here measures within-prompt perceptual variation, not visual quality.

Efficiency measured the same parameter delta above on an RTX 4060 Laptop GPU/CUDA environment, using B1 latency as primary and B8 throughput as secondary. R14 V3 retains an anomalous dual-arm B8 seed-2028 throughput observation: CA8 median was 11.2880 seconds/batch (0.7087 images/s), while CA8+CA16 was 23.9360 seconds/batch (0.3342 images/s), with large within-run variation. It is retained rather than averaged away; the B8 result is environment-specific and should not be generalized as a stable architecture cost. R14 V1 and V2 were superseded engineering-output attempts; their local roots remain provenance, while V3 is the final evidence root.

## Diagnostics, interpretation, and validity

The train-only DR01 frequency audit identifies rare primitives including `rimless_glasses` (321 occurrences) and `pale_pink` (774); `no_glasses` is common (4,266/8,388). These frequencies, and the semantic-mapping distinction between visually clear `pale_pink` and ambiguous face-color tokens, constrain interpretation. Frozen evaluator diagnostics also retain persistent glasses confusion (notably one glasses-class 4-to-5 pattern) and weaker face-shape behavior. They are diagnostic limitations, not evidence that a particular generated error has one cause.

The evidence supports a protocol-specific result: under the fixed R10–R12 design, CA8+CA16 had higher measured adherence across all three paired seeds, with no worse reported R13 distribution-quality summaries. It does not test whether the effect survives more seeds, longer training, balanced sampling, a 100k dataset, a different evaluator, alternative attention/capacity controls, or a newly blind held-out evaluation.

Threats include only three independent training seeds; evaluator dependence; rare and semantically ambiguous classes; checkpoint selection on COMP_VAL; historical exposure of FINAL_OOD; finite-prompt sampling; the intervention/capacity confound; and environment-specific speed measurements. No statistical significance or universal quality conclusion is claimed.

## Evidence and reproduction

Tracked definitions and code include this directory, `data/processed/ca16_r12_eval_v1/r12_execution_protocol.json`, `data/processed/ca16_r13_quality_v1/r13_execution_protocol.json`, `data/processed/ca16_r14_diversity_efficiency_v1/r14_execution_protocol.json`, and the R11–R14 orchestration scripts named in [REPRODUCIBILITY.md](../../../REPRODUCIBILITY.md). The historical output roots are local-only/ignored: `results/runs/ca16_ablation_*`, `results/r11_checkpoint_selection_reproducible_tokenfix_batched_v2` (`r11_summary.json` and `R11_COMPLETE.json`), `results/r12_selected_checkpoint_evaluation_batched_v1/<run>/r12_summary.json`, `results/r13_quality_guardrail_batched_v1/<run>/r13_summary.json`, `results/r13_fid_secondary_v1/r13_fid_secondary_summary.json`, and `results/r14_diversity_efficiency_b8_v{1,2,3}` (`diversity/R14_COMPLETE.json` and per-run `r14_diversity_summary.json`). The raw 10k/100k archives and the FID Inception asset are also external/local inputs; a clean clone cannot rerun CA16 merely from Git.

Use SHA256 values recorded in the R11 selection inventory and each R12–R14 summary/manifest to verify a restored evidence archive. The available historical commands are listed in the reproducibility guide; they are CUDA campaign commands and can write outputs, so they are not prescribed as routine verification. The supported clean-clone read-only workflow remains `tiny-avatar results` plus the tracked V1 hash checks.

## Closure and follow-up

CA16 is closed as a documented scientific comparison. The project has **not** selected an ultimate architecture. A prospective H7 continuation to 40k successful updates would require separate authorization and an isolated artifact root; it was not executed here. A single-model study of 32x32 and potential 64x64 support is also future work: native 64x64 support does not exist in the current implementation.
