# Historical evaluation and decision audit

## Scope and reading rule

This is a provenance audit, not a new experiment. Historical observations are
reported only as the evidence available at the time of a decision. Later
standardized results are reported separately and must not be retroactively
described as the motivation for an earlier architectural change. A metric name
alone does not establish comparability: checkpoint, evaluator, image route,
pool, prompt/seed plan, and aggregation all have to agree.

## Chronological decision ledger

| Transition | Prior observation and motivation available then | Change | Evidence subsequently available | Supported statement and confounders |
|---|---|---|---|---|
| V1 conditional -> H7 conditional | Historical CA16 ablation closure recorded a three-seed development comparison; CA16 was considered to preserve text information in the decoder. Source: `docs/experiments/ca16_ablation/FINAL_CLOSURE.md` (historical closure, not final-v3). | Add decoder 16x16 image--text cross-attention (CA16) to bottleneck CA8. | H7 checkpoints and later development/final protocol results. | CA16 is a motivated architectural intervention. It also adds 25,152 parameters, and V1/H7 historical runs are separate training runs; no strict causal isolation from all run-to-run variation. Do not treat unauthenticated historical glasses figures as verified data. |
| H7 20k -> H7 40k | A 20k H7 boundary existed on the authenticated H7 trajectory. | Continue the same trajectory for 20k successful updates. | Full-state 40k trajectory and later supplemental scoring. | This is the only within-trajectory training-budget comparison in this sequence; it retains model family but remains contingent on the evaluator/panel used. |
| H7 native32 -> V2-A native64 | Native32 limits spatial detail. The native64 experiment retained the V2-A image hierarchy and conditional CA8+CA16 design. | Add the symmetric 64px outer image stage; use native RGB64/cache/preprocessing. | V2-A 40k development records and later native64 evidence. | Resolution, architecture and preprocessing change together; this is not an architecture-only ablation. |
| V2-A 40k -> 60k | The authenticated V2-A 40k state was retained with exact continuation state. | Exact 40k->60k continuation. | `results/v2_a_matched_budget_continuation_40k_to_60k_seed20261017/telemetry.json`; STEP 9A and STEP 9B. | Controlled continuation in the same trajectory. Development diagnostics include background-sensitive evaluator behavior, so numerical adherence must not be promoted to a human-visual causal claim. |

## Historical measurement ledger

| Stage / source | Generator identity | Evaluator / pool / route | Prompt and seed plan | Metric/result actually recorded | Timing and interpretation |
|---|---|---|---|---|---|
| Early V1/CA16 development | Historical V1 CA8 and CA8+CA16 runs; see `docs/experiments/ca16_ablation/FINAL_CLOSURE.md` | Historical native32 evaluator/protocol; details are not equivalent to later dual-view scoring. | Historical development design; consult the closure and notebook records. | Qualitative/ablation closure only in this audit; no unverified glasses accuracy is transcribed. | Predates the H7 selection and motivated testing decoder CA16. |
| H7 20k boundary | H7 source SHA `ad3093446be3958d322d7af9d1109636e50e0fa7decc766177f015d04b51848d` in STEP 10 authentication. | Later supplementary dual-view evaluator SHA `963a8c…61d075`; not historical evidence. | 80 frozen-panel prompts, five seeds each. | See standardized table below. | The later measurement describes the checkpoint, not the reason CA16 was originally attempted. |
| V2-A continuation | Full state `results/v2_a_matched_budget_continuation_40k_to_60k_seed20261017/checkpoint_success_060000.pt`, SHA `f47132dc9e247861c6697e41b5d352ccfa347ba6a19e48dcee7f061206cb0f14`; 60,000 successful updates. | STEP 9A: 24 non-protected RGB64 cases × five seeds; frozen evaluator, development-only. | Identical captions/seeds across 40k/50k/60k; ancestral DDPM, 1,000 steps, EMA. | STEP 9A records background Mahalanobis/brightness/channel-imbalance and evaluator MAA/EAM; no KID/FID. | Post-continuation development diagnostic; not sealed benchmark evidence. |
| STEP 9B | Same V2-A EMA-60k SHA `f47132…0f14`. | Dual-view evaluator SHA `963a8c…61d075`; sealed-case manifest SHA `705635…b0d8d`; native64 plus Common-32 derived route. | 80 prompts × five seeds = 400 authenticated PNGs; ancestral DDPM 1,000 steps. | OOD MAA .980, EAM .920, pair-specific MacroPSR .960; KID .016694, FID 65.007; LPIPS ID/OOD .095535/.089085. | Supplementary application; not a new independent held-out test. |
| STEP 10 V1 20k | V1 CA8 EMA SHA `a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb`. | Same evaluator SHA `963a8c…61d075`, panel manifest SHA `705635…b0d8d`; native32. | 40 ID + 40 OOD prompts × five paired seeds = 400 PNGs. | KID .03919; FID 96.52; OOD MAA .2950; EAM .000; MacroPSR .000. | Verified from `results/runs/step10_supplementary_conditional_20k_v1/scoring/step10_results.json`; supplementary, not sealed v3. |
| STEP 10 H7 20k | H7 EMA source SHA `ad3093446be3958d322d7af9d1109636e50e0fa7decc766177f015d04b51848d`. | Same STEP 10 evaluator/panel; native32. | Same 80 prompts/five seeds = 400 PNGs. | KID .03648; FID 93.53; OOD MAA .4013; EAM .010; MacroPSR .045. | Same source/result file and same supplementary qualification. |

## Evaluator and protocol evolution

| Layer | Identity and role | Consequence for comparison |
|---|---|---|
| A. Early native32 evaluator | Historical four-head native32 attribute evaluator used by early V1 development. | Its pool, checkpoint and preprocessing must be recovered from its own record before quoting a number. It is not interchangeable with later evaluators. |
| B. Evaluator development | Native64 readiness documents preserve four heads (7/12/10/11 classes), 293,064 parameters, and test direct 64px versus downsampled routes. See `docs/release/NATIVE64_EVALUATOR_READINESS.md`. | The earlier downsample-to-32 native64 diagnostic is explicitly secondary; small glasses panels are neither calibrated nor final evidence. |
| C. Frozen CartoonSet100k dual-view evaluator | SHA `963a8c3592a91c3b43928cfd0728fdd6e9113a78136ebd01276478e01561d075`; direct64 and official32-then-bilinear64 views. | It provides the evaluator used by STEP 9B/10, but background sensitivity remains a documented limitation for synthetic V2-A images. |
| D. Sealed final-v3 | Sealed case manifest SHA `705635894f17147fb8dd20874aa2da094850b834171e63df5d15d4314c1b0d8d`. | The registry is immutable. Supplementary applications reuse the panel and are not extensions or independent held-out tests. |
| E. M1.2 Common-32 / diversity | Clean-FID 0.1.35; V2-A 64px images are resized 64->32 with Pillow LANCZOS; LPIPS AlexNet is native-resolution and ten unordered pairs/prompt. | Common-32 KID/FID enable a specified common-quality route, not a native-resolution equivalence claim. Native32 and native64 LPIPS are distinct protocols. |
| F. STEP 9B / STEP 10 | Authenticated later supplementary applications of the above panel/contracts. | Values are comparable only when the same endpoint, panel, evaluator, preprocessing and metric definition are documented. |

## Valid and invalid cross-protocol claims

Valid: compare the two STEP 10 20k endpoints for their stated supplementary
panel; compare V2-A 40k/60k only under their explicitly matched continuation
records; report STEP 9B metrics with its own Common-32/native64 definitions.

Invalid: infer that a later STEP 10/9B result motivated CA16 or native64;
compare historical evaluator accuracy numerically with dual-view accuracy merely
because both are called MAA; treat Common-32 FID as direct native32/native64
quality equivalence; treat small glasses diagnostics as authenticated general
glasses accuracy; or replace an unavailable historical metric with a same-named
later metric.

## Final five-endpoint supplementary progression

The following is deliberately separated from decision-time evidence. It is a
reuse of a prior panel/methodology, not a new held-out test.

| Endpoint | Updates | KID / FID (Common-32) | OOD MAA / EAM / MacroPSR | Main controlled factor |
|---|---:|---:|---:|---|
| V1 Conditional CA8 | 20k | .03919 / 96.52 | .2950 / .000 / .000 | historical CA8 run |
| H7 CA8+CA16 | 20k | .03648 / 93.53 | .4013 / .010 / .045 | decoder CA16 plus separate-run confounder |
| H7 CA8+CA16 | 40k | retain separately authenticated summary | retain separately authenticated summary | same-trajectory continuation |
| V2-A CA8+CA16 | 40k | .02788 / 75.38 | .645 / .055 / .385 | resolution/architecture/preprocessing change |
| V2-A CA8+CA16 | 60k | .01669 / 65.01 | .980 / .920 / .960 | same-trajectory continuation |

The H7/V2-A 40k rows are from `results/runs/v2a_step9b_final_supplementary_ema60k_v1/five_endpoint_comparison.json`; their reuse must retain that file's supplementary status and endpoint-specific authentication.

## Figure candidates and unresolved evidence

Candidates: `results/runs/v2a_step9b_final_supplementary_ema60k_v1/native64_ema40k_vs_ema60k.png`, `conditional_matched_comparison.png`, and `figure_step9a_development.png`. Captions must label the first two as illustrative matched prompt/seed outputs and the development figure as non-protected.

Unresolved: a single normalized table of early-V1 evaluator checkpoint/pool/seed metadata was not found in this audit. Therefore no early glasses number is verified here. The documented native64 evaluator background sensitivity precludes a purely automatic causal visual-quality claim. The causal effect of CA16 is also confounded by its parameter increase and separate-run design.

## Manuscript-ready section: Experimental Development and Evaluation-Protocol Evolution

The project proceeded through an iterative development sequence rather than a
single static benchmark. Decoder cross-attention at 16x16 was introduced after
historical native32 development suggested that bottleneck-only conditioning
could lose attribute information during decoding. Native64 V2-A subsequently
extended the image path to represent finer spatial structure, while the 40k to
60k V2-A comparison was an exact continuation within one authenticated
trajectory. These decisions were informed by contemporaneous development
evidence; later standardized STEP 9B and STEP 10 measurements are reported as
post hoc supplementary characterization, not as their motivation.

Evaluator protocols also evolved. Early native32 results used an earlier
attribute-evaluator route, whereas later supplementary studies use the frozen
CartoonSet100k dual-view evaluator and a fixed 80-prompt/five-seed panel.
Consequently, identically named MAA, EAM, MacroPSR, KID/FID, and LPIPS values
are comparable only within a fully matched contract. Common-32 quality metrics
explicitly resize native64 outputs, and native64 LPIPS remains a separate
native-resolution quantity. The dual-view evaluator's documented background
sensitivity, especially for synthetic V2-A imagery, further limits causal
interpretation of automatic attribute scores.

## Paper update recommendations

1. Add this protocol-evolution section before the final-results table.
2. Label all STEP 9B/10 values as supplementary panel reuse, not new held-out tests.
3. State the CA16 capacity/run confounder and native32-to-native64 multi-factor confounder adjacent to ablation claims.
4. Keep historical glasses claims qualitative unless their original evaluator, pool, and generation record are independently authenticated.
5. Caption Common-32 and native64 LPIPS metrics with their preprocessing routes.
