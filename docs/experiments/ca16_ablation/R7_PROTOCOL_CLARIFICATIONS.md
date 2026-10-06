# R7 Protocol Clarifications

**Status:** FROZEN procedural addendum  
**Experiment:** focused CA16 ablation  
**Date:** 2026-10-06

## Scope and immutability

R1 and R2 remain immutable. This is a post-hoc procedural clarification made
before any CA16 full training, checkpoint selection, or scientific result
exists. It resolves two implementation-relevant ambiguities surfaced during
the R7 critical review.

This clarification does not alter dataset splits, the architecture, the
training budget, the outcome hierarchy, frozen R2 record membership, or
`FINAL_OOD` sealing. It does not modify the R2 manifests.

## Clarification A: ID checkpoint-selection seeds

For the 256 ID checkpoint-selection records in canonical
`checkpoint_selection_manifest.json` ID-entry order, freeze:

`generation_seed(i) = 2026 + i`

for `i = 0, ..., 255`. Therefore, the ID checkpoint-selection generation seeds
are `2026, ..., 2281`.

This mapping depends only on canonical ID manifest order. The same record uses
the same seed across architecture arms, training seeds, and candidate
milestones. The architecture arm does not affect the generation seed.

This rule applies only to the ID checkpoint-selection panel. The existing
frozen COMP rule remains unchanged: for COMP selection canonical index
`i = 0, ..., 236`,

`generation_seed(i) = 2026 + i`

as already frozen in R2 and the protocol.

The R2 manifest is not modified to serialize the new ID seeds. This addendum
is the frozen source for the ID seed rule.

## Clarification B: COMP MacroPSR

For each compositional family `f` in `CV1`, `CV2`, `CV3`, and `CV4`, compute:

`PSR_f = fraction of generated samples in family f for which all four requested attributes are predicted correctly`

Then freeze:

`COMP MacroPSR = (PSR_CV1 + PSR_CV2 + PSR_CV3 + PSR_CV4) / 4`

Each family has equal weight regardless of its number of selection records.
This is not pooled or sample-count-weighted COMP PSR. No family may be omitted.
MacroPSR uses the same equal-family rationale as the primary COMP family-macro
MAA. This clarification affects only the definition of selection criterion 2.

## Selection hierarchy unchanged

The already frozen checkpoint-selection hierarchy remains unchanged:

1. higher COMP family-macro MAA;
2. higher COMP MacroPSR;
3. higher ID MAA;
4. lower COMP KID;
5. earlier milestone.

No additional tie-break criterion is introduced. Epsilon-MSE is not a
selection criterion. No statistical significance test or approximate-tie
tolerance is introduced unless already frozen elsewhere.

## Epistemic status

These clarifications were made before observing any CA16 ablation training
result. No CA16 checkpoint-selection metric influenced their choice.
Therefore, they are procedural clarifications, not result-driven adaptation.

V1 `FINAL_OOD` evidence informed the original hypothesis, but `FINAL_OOD`
remains operationally sealed from CA16 development and evaluation.
