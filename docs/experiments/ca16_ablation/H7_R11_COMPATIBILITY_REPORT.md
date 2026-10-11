# H7 R11 terminal-evaluation compatibility report

Verdict: **compatible at the binding/preflight level**. The H7 adapter is a
separate module and does not invoke or alter the frozen R11 orchestrator.
It accepts only an H7 `terminal_40000.pt` under the H7 results tree, validates
its CA8+CA16 continuation identity, and resolves all frozen R11 inputs from
the explicit read-only CA16 asset root. H7-local historical copies are never
used as a fallback.

## Bound R11 inputs

- Selection prompt manifest: `checkpoint_selection_manifest.json`, SHA256
  `192191235af7992b8c332019b7e948414ba0b87c5f8df8d70f044b74e85e3d40`.
- Robust/family manifest: `robust_conditioning_manifest.json`, SHA256
  `e3319f65f67c19ae8b458f7450ff8e6b86e5913e0fd542e9c0c3009802451aec`.
- Evaluator freeze: SHA256
  `85b0cef08bb5a298442d7d863c4f76b114ecf023335d312f319c6feef168f13f`;
  frozen checkpoint `dr04_evaluator_main_100k_seed2026_r1/best.pt`, SHA256
  `3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f`.
- Authoritative R11 inventory: SHA256
  `cace0eb71751c770c4e4c1318e9683813566fd1a36b7595b4bb8ab84a37e7790`.
- Authoritative successful R11 completion marker: SHA256
  `3fd6c3b3caf8df527c5d93d2320398ce492dea57a0b76243c420ac02ffde0122`.

The adapter verifies these external hashes before exposing the panel. It does
not add CA16 to the import path.

## Preserved R11 semantic contract

The plan consists of exactly 256 `ID_VAL` and 237 `COMP_VAL` prompts. Prompt
records are selected from the immutable manifest; their canonical token rows
are consequently still sourced by the frozen R11 semantics. Generation uses
the manifest assignment `2026 + canonical index`, EMA weights, FP32 ancestral
pixel DDPM, `T=1000`, `clip_denoised=True`, and no CFG or conditioning dropout.

Evaluator preprocessing, ordered attributes (`face_shape`, `glasses`,
`hair_color`, `face_color`), per-prompt prediction mapping, and aggregation
are reused from `ca16_evaluation`: COMP has CV1/CV2/CV3/CV4 counts
40/114/27/56; its primary MAA and MacroPSR are equal-weight family macros.
`ID_TEST`, `FINAL_OOD`, and `TRAIN` are structurally excluded.

## 20k comparison binding

The authoritative R11 V2 selections bind CA8+CA16 20k COMP family-macro MAA
to seed 2026 `0.670254194514063`, 2027 `0.7919420948203844`, and 2028
`0.7501475041771094`, with source checkpoint SHA256 values checked both by the
historical R11 inventory and H7 continuation contract. A future completed H7
evaluation may report `MAA_40k - MAA_20k` per seed and the three-seed mean;
these are paired, protocol-compatible comparisons, not held-out confirmation.

## R13 KID feasibility (inspection only)

Comparable 40k KID is technically feasible but deliberately unimplemented.
The frozen R13 protocol is `ca16_r13_quality_guardrail_batched_v1`, hash
`fe292f1a67b4f755bfae0512601b206f3abd7d4ec4a2cae4bc820098631f240b`.
It requires 804 generated 32x32 RGB images per checkpoint (466 ID_VAL, 338
COMP_VAL), exactly the corresponding read-only real-reference populations,
EMA FP32 ancestral DDPM T=1000, Clean-FID 0.1.35 / standard Inception-v3,
and 100 KID subsets seeded 2026. The external CA16 protocol and source assets
are readable, but H7 would need its own H7-only generated-image manifests,
real-reference manifests/pools, and metric output provenance. It requires GPU
sampling plus GPU (or potentially slow CPU) feature extraction; no defensible
time estimate is available from the frozen artifacts, so no estimate is
invented. No FID or LPIPS work is included.
