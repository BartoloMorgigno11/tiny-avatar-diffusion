# DR01 Final Compositional Split

## Protocol

This split is compositional rather than random-only: all examples matching any of eight frozen value-pair rules are removed before the remaining ID pool is split. COMP_VAL measures development-time generalization to four unseen combinations, while FINAL_OOD seals four different unseen combinations for final evaluation. FINAL_OOD has assignment precedence so an image matching both panels cannot leak into development-time compositional validation.

The exact same held-out value-pair definitions, precedence, 90/5/5 ID policy, and seed 2026 are reused independently for the official 10k and 100k datasets. Absolute counts intentionally differ.

All selected attribute values and every primitive used by a held-out definition remain represented in TRAIN. Vocabulary leakage is not assessed here because caption and tokenizer construction belongs to the next step.

## Final counts

| Dataset | Source | TRAIN | ID_VAL | ID_TEST | COMP_VAL | FINAL_OOD |
|---|---:|---:|---:|---:|---:|---:|
| 10k | 10000 | 8388 | 466 | 466 | 338 | 342 |
| 100k | 100000 | 84011 | 4667 | 4667 | 3311 | 3344 |

## Held-out pair support

### 10k

| Rule | Role | Before precedence | Effective after precedence |
|---|---|---:|---:|
| CV1: `face_shape=0 x glasses=0` | COMP_VAL | 68 | 66 |
| CV2: `face_shape=2 x hair_color=3` | COMP_VAL | 147 | 139 |
| CV3: `glasses=7 x face_color=2` | COMP_VAL | 54 | 52 |
| CV4: `hair_color=2 x face_color=9` | COMP_VAL | 82 | 82 |
| OOD1: `face_shape=2 x glasses=3` | FINAL_OOD | 72 | 72 |
| OOD2: `face_shape=4 x hair_color=6` | FINAL_OOD | 137 | 137 |
| OOD3: `glasses=10 x face_color=10` | FINAL_OOD | 47 | 47 |
| OOD4: `hair_color=0 x face_color=0` | FINAL_OOD | 86 | 86 |

Overlap counts: multiple FINAL_OOD = 0; multiple COMP_VAL = 1; both panels = 12.

Post-holdout marginal minimum = 366; post-holdout partner-diversity minimum = 6; unintended absent pair cells = 0.

### 100k

| Rule | Role | Before precedence | Effective after precedence |
|---|---|---:|---:|
| CV1: `face_shape=0 x glasses=0` | COMP_VAL | 659 | 653 |
| CV2: `face_shape=2 x hair_color=3` | COMP_VAL | 1448 | 1386 |
| CV3: `glasses=7 x face_color=2` | COMP_VAL | 420 | 413 |
| CV4: `hair_color=2 x face_color=9` | COMP_VAL | 873 | 869 |
| OOD1: `face_shape=2 x glasses=3` | FINAL_OOD | 665 | 665 |
| OOD2: `face_shape=4 x hair_color=6` | FINAL_OOD | 1373 | 1373 |
| OOD3: `glasses=10 x face_color=10` | FINAL_OOD | 411 | 411 |
| OOD4: `hair_color=0 x face_color=0` | FINAL_OOD | 900 | 900 |

Overlap counts: multiple FINAL_OOD = 5; multiple COMP_VAL = 10; both panels = 79.

Post-holdout marginal minimum = 3723; post-holdout partner-diversity minimum = 6; unintended absent pair cells = 0.

## Integrity result

For both datasets, the five manifests are pairwise image-ID disjoint, their union equals the source set, every source image is assigned once, all eight held-out pairs have zero TRAIN support, held-out primitives retain positive TRAIN support, alternate-partner diversity is greater than one, and no exact image hash crosses final split roles.
