# R1 numerical consistency review

Reviewed against frozen sources on 21 October 2026; no result file was changed.

| Claim in final report | Verified canonical source | Status |
|---|---|---|
| Conditional H7-40k: KID/FID .010000/59.430; OOD MAA/EAM/MacroPSR .965/.860/.895 | `results/runs/v2a_step9b_final_supplementary_ema60k_v1/five_endpoint_comparison.json` | Consistent. |
| Conditional V2-A-60k: .016694/65.007; .980/.920/.960 | Same STEP 9B file | Consistent. |
| Unconditional V1-40k: KID .0081339451, FID 41.5468880, LPIPS .08889223 | `results/runs/step13_final_unconditional_benchmark_v1/summary.json` | Consistent, rounded in manuscript. |
| Unconditional V2-A-60k: KID .0097242091, FID 47.2149742, LPIPS .23463342 | Same STEP 13 summary | Consistent, rounded in manuscript. |
| Unconditional parameters, latency and peak memory | STEP 13 per-endpoint `metrics.json` and `efficiency.json` | Consistent. |
| Four public package SHA-256 identities | `src/tiny_avatar_diffusion/public_model_manifest.json` and `results/release_validation/m4_9_gpu_validation_20261021/validation.json` | Consistent. |
| Conditional 20k chain values | `results/runs/step10_supplementary_conditional_20k_v1/scoring/step10_results.json` | Consistent. |

Review decisions:

- FID is labelled exploratory in both final quantitative tables.
- No MAA, EAM or MacroPSR is attributed to an unconditional endpoint.
- Conditional LPIPS and unconditional marginal LPIPS are described as distinct estimands.
- Native64 Common-32 and LPIPS caveats are retained.
- V1 unconditional 20k is absent from final report prose, figures and tables.
- The persisted STEP 13 setup-state files are not used as completion evidence; `summary.json` is the status authority.
