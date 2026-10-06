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

The role and order of COMP_VAL KID as the fourth tie-break are frozen. Its exact prospective pool size, sampling rule, and operational manifest remain open and must be frozen in R2 before training; model outputs must not influence that choice. KID measures distributional image quality, not prompt adherence.

The primary COMP aggregation is family-macro. Frozen 10k COMP membership counts after precedence are CV1 = 66, CV2 = 139, CV3 = 52, and CV4 = 82 across 338 unique COMP_VAL records. One record satisfies more than one COMP definition, so the membership counts sum to 339. Record-micro aggregation would therefore weight families unequally, especially CV2. The primary aggregation is:

`MAA_COMP = (MAA_CV1 + MAA_CV2 + MAA_CV3 + MAA_CV4) / 4`.

The robust conditioning panel will use 100 ID_VAL prompts and 100 COMP_VAL prompts, with 25 prompts from each of CV1--CV4. A COMP_VAL example belonging to multiple COMP definitions is ineligible for a family-specific conditioning panel. Selected models use the same prompts, generation seeds `2026`--`2030`, sampler/settings, and aggregation rules.

Quality is a KID-based guardrail on a separate prospectively frozen, distribution-preserving manifest; the balanced/coverage-aware conditioning panel is not its sole representative pool. Diversity follows the V1 LPIPS-style prompt-macro philosophy where feasible, with no arbitrary diversity threshold. Efficiency reporting includes parameter count, sampling latency, images/sec, peak CUDA allocated memory, and peak CUDA reserved memory.

## Reproducibility and open protocol items

All run-level provenance, checkpoint-selection inputs, prompt identities, generation seeds, aggregation outputs, and deviations must be retained so the pairing and selection rule are auditable.

**OPEN — TO BE FROZEN IN R2 BEFORE TRAINING:** exact prospective COMP_VAL KID tie-break pool size, sampling rule, and manifest; and exact size and sampling rule for the separate distribution-preserving quality-guardrail pool. No model outputs may influence either choice.

**OPEN — TO BE FROZEN BEFORE FULL TRAINING:** exact diversity prompt-count/panel rule.
