# CA16 Ablation Preregistration

**Experiment:** focused CA16 ablation  
**Branch:** `exp/ca16-ablation`  
**Frozen starting point:** `v1.0.0` (`76a41e5742cf53f87eded1c49e9450bdd287469b`)  
**Status:** FROZEN preregistration. Scientifically reviewed and frozen before CA16 implementation, manifest construction, or training.

## Motivation and primary hypothesis

V1 established real but attribute-selective conditioning: ID MAA was approximately 0.522 and OOD MAA approximately 0.425. Glasses was the clearest weakness, followed by face shape and hair color. Some primitive weaknesses also occurred outside held-out combinations, while some held-out pairs exhibited further interaction/compositional degradation. The real-only evaluator was near-perfect at 32x32, so this is not evidence that the attributes are intrinsically unrecognizable at that resolution.

**H1.** The V1 bottleneck-only 8x8 image-to-text cross-attention may limit semantic conditioning. Extending that same mechanism with a second image-to-text cross-attention injection at decoder16 should improve conditioning adherence and may improve compositional behavior under an otherwise matched protocol.

The control is **CA8** (the unchanged V1 conditioning path). The treatment is **CA8 + CA16**. CA8 remains present and unchanged in the treatment.

## Outcomes and decision rule

### Primary outcome

The primary outcome is **COMP_VAL family-macro MAA**. For each training seed `s`:

`Delta_s = MAA_COMP(CA8 + CA16, s) - MAA_COMP(CA8, s)`.

Directional support for H1 requires mean `Delta > 0` and all three paired deltas to be positive. A positive mean without 3/3 positive deltas is inconclusive/mixed evidence. A non-positive mean or no systematic benefit does not support H1. This is not a statistical-significance claim.

### Secondary outcomes

Secondary outcomes are **COMP_VAL MacroPSR** and **ID_VAL MAA**.

### Diagnostics

Diagnostics are per-attribute accuracy, per-class accuracy, family-level results, and confusion/substitution analysis. Each delta, mean delta, and effect magnitude will be reported. A paired prompt-bootstrap confidence interval is planned with joint prompt resampling while preserving identical prompt/generation-seed structure. Its uncertainty concerns the evaluated prompts, not a population of training seeds.

### Guardrails

KID-based distributional image quality, diversity, and efficiency are guardrails and remain separate from conditioning adherence. KID measures distributional image quality, not prompt adherence.

Checkpoint selection is adherence-first and lexicographic: highest COMP_VAL family-macro MAA; then highest COMP_VAL MacroPSR; then highest ID_VAL MAA; then lowest COMP_VAL KID; then earlier milestone.

## Causal interpretation

A positive result supports only this statement:

> Extending bottleneck-only conditioning with a second image-to-text cross-attention injection at decoder16 improves conditioning/compositional behavior under the matched protocol.

The intervention adds both a second conditioning injection and approximately 25,152 expected additional parameters, with the exact treatment parameter count to be measured from the implemented model. It therefore does **not** isolate 16x16 spatial resolution alone and cannot establish that 16x16 is the causally optimal attention location.

Conditioning, quality, diversity, and efficiency will remain separate: higher MAA with stable quality is a clean-conditioning-improvement candidate; material KID degradation is a conditioning-quality trade-off; and diversity collapse is not a clean win.

- An MAA increase with MacroPSR approximately unchanged is evidence for improved attribute/primitive control, but not strong evidence for improved joint compositional satisfaction.
- An MAA increase together with a MacroPSR increase is stronger evidence that the conditioning extension also benefits compositional behavior.

Neither interpretation is a guarantee or automatically equivalent to statistical significance.

## Open alternative hypotheses

This experiment does not test the 100k-data regime, 64x64 resolution, classifier-free guidance, a larger U-Net, a larger text Transformer, a changed diffusion objective, or a longer training budget as the primary intervention. These remain open alternatives.

## FINAL_OOD epistemic status

`FINAL_OOD` is operationally sealed for this extension and will not be used for architecture development, checkpoint selection, manifest construction, tuning, or robust internal causal evaluation. V1 `FINAL_OOD` remains valid V1 final-test evidence. CA16 is a post-hoc, failure-driven focused extension because the V1 `FINAL_OOD` evidence and failure analysis informed H1. Accordingly, prospective `COMP_VAL` is the primary internal evaluation domain here. Any future CA16 evaluation on `FINAL_OOD` must disclose that prior V1 `FINAL_OOD` evidence informed hypothesis formation.
