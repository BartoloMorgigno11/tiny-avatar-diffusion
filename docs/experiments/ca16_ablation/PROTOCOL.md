# CA16 Ablation Protocol

## Intervention and matched constants

The V1 control path is `bottleneck_block1 -> CA8 -> bottleneck_block2`. The treatment retains this path and additionally uses `concat with s16 -> decoder16_block1 -> CA16 -> decoder16_block2`. Here, decoder16 is the 16x16 decoder stage after concatenation with `s16`. In both CA8 and CA16, image features supply Q and text context supplies K,V.

U-Net backbone width, depth, residual-block structure, skip connections, and all non-CA16 components remain unchanged between arms. The only planned architectural change relative to the control is the added second image-to-text cross-attention block between `decoder16_block1` and `decoder16_block2`; CA8 remains present and unchanged.

Both arms use the frozen Google Cartoon Set 10k MAIN split; 32x32 RGB pixel-space images; the existing tokenizer/vocabulary; the unchanged scratch Transformer text encoder; and pixel-space DDPM epsilon prediction with `T=1000`, linear beta schedule, ancestral DDPM sampling, fixed-small variance, and x0 clipping. No CFG, conditioning dropout, FiLM/AdaGN, attention at all resolutions, larger Transformer, larger U-Net, 64x64, 100k data, or objective change is in scope.

Training is AdamW with learning rate `2e-4`, weight decay `0`, physical batch size `32`, AMP FP16, EMA `0.9999`, a budget of 20,000 successful optimizer steps, milestones at 5k/10k/15k/20k, and validation loss every 1k successful steps.

## Runs and pairing

Required full runs are CA8 and CA8+CA16 at each training seed: `2026`, `2027`, and `2028`. All six runs must use the same experimental code commit.

For each paired seed, shared components must initialize identically; future implementation must verify that equality at initialization. The same seed should govern the training sampler, diffusion-corruption generator, and runtime RNG state where feasible. V1 AMP semantics must be preserved. An attempted batch is consumed when GradScaler skips an update; attempted-batch count, successful-step count, skipped-attempt count, and (where feasible) skipped-attempt positions must be recorded. If skips differ within a pair, the same-seed/matched-protocol designation remains, but imperfect post-divergence sample-trajectory pairing must be reported as a limitation.

## Evaluation plan

Candidate checkpoints are 5k, 10k, 15k, and 20k successful-step milestones. Selection is lexicographic:

1. highest COMP_VAL family-macro MAA;
2. highest COMP_VAL MacroPSR;
3. highest ID_VAL MAA;
4. lowest COMP_VAL KID;
5. earlier milestone.

The role and order of COMP_VAL KID as the fourth tie-break are frozen. Its R2 pool, sampling rule, and seed mapping are frozen below. KID measures distributional image quality, not prompt adherence.

The primary COMP aggregation is family-macro. Frozen 10k COMP membership counts after precedence are CV1 = 66, CV2 = 139, CV3 = 52, and CV4 = 82 across 338 unique COMP_VAL records. One record satisfies more than one COMP definition, so the membership counts sum to 339. Record-micro aggregation would therefore weight families unequally, especially CV2. The primary aggregation is:

`MAA_COMP = (MAA_CV1 + MAA_CV2 + MAA_CV3 + MAA_CV4) / 4`.

## R2 frozen evaluation-panel design

All R2 and later development manifests may use only `ID_VAL` and `COMP_VAL`. `ID_TEST` and `FINAL_OOD` are denied by default and must be rejected by a future implementation allowlist; they are not inputs to any R2/R3 development panel.

### Stable record identity

The stable record key is `(dataset_variant, split_role, image_id)`, serialized canonically as `<dataset_variant>|<split_role>|<image_id>`. `source_image_path` is retained only as provenance. Artifact order, row position, and manifest position are not stable identity inputs.

### Robust conditioning and diversity panel

The robust COMP panel contains 100 unique prompts: 25 from each of CV1, CV2, CV3, and CV4. The one multi-definition `COMP_VAL` record is excluded from this panel before selection. For each family, eligible records are ordered by ascending `SHA256("2026|COMP_ROBUST|<family>|<stable_record_key>")`; the first 25 are selected. Expected eligible counts after the exclusion are CV1=65, CV2=139, CV3=52, and CV4=81. This procedure depends only on frozen data and annotations, never on model outputs.

The robust ID panel contains 100 `ID_VAL` prompts selected by a deterministic, ground-truth-only coverage-aware greedy procedure over all four attributes. Attribute contributions are normalized so that attributes contribute equally; at each selection the rule prioritizes the most undercovered equal-class target. Ties are broken by ascending `SHA256("2026|ID_ROBUST|<stable_record_key>")`. R2C must specify the exact deterministic formula and audit outputs, without random sampling, natural-distribution sampling, or model-performance inputs.

The robust 100 ID and 100 COMP prompts are also the diversity panel. For each prompt, reuse the same robust-conditioning samples generated with seeds 2026, 2027, 2028, 2029, and 2030: five images and all 10 unordered LPIPS pairs, with no additional diversity-only samples.

Diversity follows V1 LPIPS semantics: LPIPS `alex` / version `0.1` on RGB 32x32 images in `[-1, 1]`; compute the mean pairwise score per prompt. Report ID diversity as the macro mean over its 100 prompts. Report COMP diversity per family and then as an equal-family macro. Use a 10,000-replicate prompt-level percentile bootstrap with seed 202605, preserving COMP family structure. This is a matched ablation guardrail, not a natural-distribution estimate.

### Checkpoint-selection panel and KID tie-break

After removing the 100 robust ID records, the remaining 366 `ID_VAL` records form the ID selection candidate pool. Order them by ascending `SHA256("2026|ID_SELECTION|<stable_record_key>")` and select the first 256, leaving 110 unselected.

After removing the 100 robust COMP records and the one excluded multi-definition record, all remaining 237 `COMP_VAL` records form the COMP selection panel. This makes the conditioning-selection source sets strictly disjoint from the robust conditioning/diversity source sets.

Use these same 237 COMP sources for the checkpoint-selection KID tie-break. For each candidate milestone and model, generate one image per prompt using canonical manifest index `i` with seed `2026+i`. KID is the fourth checkpoint tie-breaker only and is an image-distribution quality measure, not a conditioning-adherence measure.

### Prospective quality guardrail

For every selected model, create a separate prospective quality-generation manifest covering all 466 `ID_VAL` and all 338 `COMP_VAL` prompts (804 prompts total), with one output per source record. Use canonical manifest index `i` and seed `2026+i`, identically across all six selected models. Report `KID_ID`, `KID_COMP`, and `KID_GLOBAL` on their respective ID, COMP, and union source sets; preserve FID as a secondary diagnostic if the existing infrastructure produces it unchanged.

Quality and conditioning evaluations use separate prospective generation manifests, although their validation source records may overlap. The quality guardrail therefore explicitly permits source-record overlap with the conditioning and checkpoint-selection panels while keeping the generated artifacts distinct. It must not use `ID_TEST` or `FINAL_OOD`.

### Frozen panel-purpose separation

| Purpose | Sources and size | Use |
| --- | --- | --- |
| Checkpoint selection | 256 ID + 237 COMP candidate-milestone prompts | MAA/MacroPSR selection; COMP KID as fourth tie-breaker |
| Robust conditioning and diversity | 100 ID + 100 COMP, with COMP 25 per family | MAA, MacroPSR, diagnostics, and matched LPIPS diversity |
| Prospective final quality | All 466 ID + all 338 COMP; one output per source | KID diagnostics and unchanged secondary FID when available |

## Reproducibility and retained scientific boundaries

All run-level provenance, checkpoint-selection inputs, prompt identities, generation seeds, aggregation outputs, and deviations must be retained so the pairing and selection rule are auditable.

Exact implementation manifests and the deterministic audit outputs for the frozen rules are R2C implementation deliverables; they do not reopen the panel-design decisions above. Exact parameter count, actual results, whether 100k steps, 64x64 resolution, CFG, larger models, a different objective, longer training, or a post-CA16 follow-up are warranted remain outside this focused ablation.
