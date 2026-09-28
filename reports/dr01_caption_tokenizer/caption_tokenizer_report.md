# DR01 Caption and Tokenizer Finalization

## Canonical grammar

`avatar <face_shape_token> <glasses_token> <hair_color_token> <face_color_token>`

Every caption has exactly five lowercase whitespace-delimited lexical tokens. Attribute-value tokens are atomic and underscores are never split.

## Tokenizer

The tokenizer is deterministic whitespace word-level. IDs 0–3 are `<pad>`, `<bos>`, `<eos>`, `<unk>`; lexical tokens follow in lexicographic order. Serialized length is fixed at 7, so canonical captions need no padding.

## TRAIN-only vocabulary and leakage results

| Dataset | Vocabulary size | Lexical size | ID_VAL OOV | ID_TEST OOV | COMP_VAL OOV | FINAL_OOD OOV | Total `<unk>` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10k | 45 | 41 | 0 | 0 | 0 | 0 | 0 |
| 100k | 45 | 41 | 0 | 0 | 0 | 0 | 0 |

Each vocabulary was constructed independently from its TRAIN captions only. Non-train captions were generated and checked only after vocabulary construction. No validation or test token was inserted into either vocabulary.

## Cross-dataset comparison

Lexical-token intersection size: **41**.
Tokens unique to 10k TRAIN: **none**.
Tokens unique to 100k TRAIN: **none**.

The vocabularies are equal because every primitive value remains represented in both TRAIN splits; equality was verified, not forced.

## Semantic provenance

`glasses=11 -> no_glasses` is the only official Google semantic mapping. Other readable labels are project-defined from the visual audit. `face_shape=1` remains neutral. Face-color labels are graphical color descriptors and do not represent race, ethnicity, or demographic categories.

## Assertions

All caption-order, one-caption-per-row, deterministic regeneration, deterministic vocabulary/ID, same-metadata/same-text, zero-OOV, zero-`<unk>`, and raw-manifest immutability assertions passed for both datasets.
