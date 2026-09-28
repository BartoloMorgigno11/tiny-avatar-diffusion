# DR01 Semantic Mapping Audit — S4_FC

## A. Primary-source search

The search was restricted to primary Google Cartoon Set materials. The [official Google Cartoon Set data documentation](https://google.github.io/cartoonset/download.html) states that each per-image CSV stores the selected variant as a numeric value and the total number of variants. It documents the generic attribute meanings/cardinalities: `face_shape` (7), `glasses` (12), `hair_color` (10), and `face_color` (11). It explicitly states that glasses type 11 means no glasses.

The [official `google/cartoonset` repository](https://github.com/google/cartoonset) is an archived repository containing the project webpage code and assets; inspection found no complete ID-to-semantic-name table for the four attributes. Direct read-only inspection of both distributed archives found only PNG and per-image CSV files, with numeric IDs and cardinalities but no schema or label file.

No third-party source was used to establish an official label.

## B. Official mapping availability

| Attribute | Complete official ID mapping found | Exact primary-source evidence |
|---|---|---|
| `face_shape` | no | Official documentation gives 7 variants and the generic description “overall shape of the face”; no names by ID. |
| `glasses` | no | Official documentation gives 12 variants and explicitly maps only type 11 to no glasses. |
| `hair_color` | no | Official documentation gives 10 variants and a generic color description; no names by ID. |
| `face_color` | no | Official documentation gives 11 variants and a generic face-fill color description; no names by ID. |

Accordingly, only `glasses=11` is marked `OFFICIAL`. Every other natural-language descriptor is marked as a `PROJECT-DEFINED VISUAL LABEL CANDIDATE`, or replaced by a neutral token when a concise descriptor is not prudent.

## C. Face-shape mapping candidates

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | pointed | VISUALLY_CLEAR | Stable tapered jaw and pointed chin across both datasets. |
| 1 | face_shape_1 | NEUTRAL_TOKEN_REQUIRED | Stable outline, but short labels such as angular or tapered are not sufficiently unambiguous. |
| 2 | oval | VISUALLY_CLEAR | Stable rounded oval outline across both datasets. |
| 3 | long | VISUALLY_CLEAR | Stable vertically elongated outline across both datasets. |
| 4 | round | VISUALLY_CLEAR | Stable compact near-round outline across both datasets. |
| 5 | wide | VISUALLY_CLEAR | Stable broad cheek and jaw outline relative to height. |
| 6 | rectangular | VISUALLY_CLEAR | Stable longer outline with comparatively straight sides. |

## D. Glasses mapping candidates

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | oversized glasses | VISUALLY_CLEAR | Large rounded frames are stable in both datasets. |
| 1 | aviator sunglasses | VISUALLY_CLEAR | Dark aviator-style lenses are stable in both datasets. |
| 2 | cat eye glasses | VISUALLY_CLEAR | Upswept cat-eye frame is stable in both datasets. |
| 3 | rimless glasses | VISUALLY_CLEAR | Minimal narrow rim appearance is stable in both datasets. |
| 4 | rectangular glasses | VISUALLY_CLEAR | Thin rectangular frame is stable in both datasets. |
| 5 | round glasses | VISUALLY_CLEAR | Small round frame is stable in both datasets. |
| 6 | thick rectangular glasses | VISUALLY_CLEAR | Thick rectangular frame is stable in both datasets. |
| 7 | oval glasses | VISUALLY_CLEAR | Small oval frame is stable in both datasets. |
| 8 | narrow sunglasses | VISUALLY_CLEAR | Narrow dark lenses are stable in both datasets. |
| 9 | round sunglasses | VISUALLY_CLEAR | Round dark lenses are stable in both datasets. |
| 10 | heart glasses | VISUALLY_CLEAR | Heart-shaped lenses are stable in both datasets. |
| 11 | no glasses | OFFICIAL | Official Google documentation explicitly states that type 11 is no glasses. |

## E. Hair-color mapping candidates

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | light blond | VISUALLY_CLEAR | Stable light blond fill; observed base RGB 255,222,138. |
| 1 | golden blond | VISUALLY_CLEAR | Stable golden blond fill; observed base RGB 255,187,77. |
| 2 | orange | VISUALLY_CLEAR | Stable orange fill; observed base RGB 237,111,47. |
| 3 | red | VISUALLY_CLEAR | Stable red fill; observed base RGB 192,74,42. |
| 4 | golden brown | VISUALLY_CLEAR | Stable golden-brown fill; observed base RGB 216,154,50. |
| 5 | medium brown | VISUALLY_CLEAR | Stable medium-brown fill; observed base RGB 161,83,50. |
| 6 | dark brown | VISUALLY_CLEAR | Stable dark-brown fill; observed base RGB 91,51,28. |
| 7 | black | VISUALLY_CLEAR | Stable near-black fill; observed base RGB 30,21,20. |
| 8 | blue gray | VISUALLY_CLEAR | Stable blue-gray fill; observed base RGB 144,164,174. |
| 9 | silver | VISUALLY_CLEAR | Stable light-gray/silver fill; observed base RGB 224,224,224. |

## F. Face-color mapping candidates

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | deep brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 96,61,48. |
| 1 | dark brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 136,89,75. |
| 2 | medium brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 170,116,84. |
| 3 | warm brown | VISUALLY_AMBIGUOUS | Stable RGB 199,139,93, but the short color name overlaps nearby palette terms. |
| 4 | golden tan | VISUALLY_AMBIGUOUS | Stable RGB 217,161,110, but the boundary between tan and light brown is naming-dependent. |
| 5 | light tan | VISUALLY_CLEAR | Stable light-tan graphic fill; observed base RGB 227,180,126. |
| 6 | rose tan | VISUALLY_AMBIGUOUS | Stable RGB 238,175,148, but the concise descriptor is not uniquely determined. |
| 7 | peach tan | VISUALLY_AMBIGUOUS | Stable RGB 240,192,146, but peach/tan wording requires human confirmation. |
| 8 | light beige | VISUALLY_CLEAR | Stable light-beige graphic fill; observed base RGB 246,216,193. |
| 9 | light peach | VISUALLY_AMBIGUOUS | Stable RGB 251,205,182, but it is close to adjacent pale warm colors. |
| 10 | pale pink | VISUALLY_CLEAR | Stable pale-pink graphic fill; observed base RGB 251,219,209. |

`face_color` labels describe only the observed graphic fill colors of cartoon faces. They do not denote or infer ethnicity, race, gender, age, or any demographic category.

## G. Ambiguous values

`face_color` IDs 3, 4, 6, 7, and 9 are visually stable across the reviewed samples, but their short English color names overlap neighboring palette terms and therefore remain `VISUALLY_AMBIGUOUS`. `face_shape=1` has a stable outline, but no short descriptor was judged sufficiently unambiguous, so its candidate representation is the neutral token `face_shape_1`.

## H. Cross-dataset consistency

For every one of the 40 value IDs, 6 deterministic examples from the official 10k and 6 from the official 100k were reviewed. The observed rendering identity was consistent across datasets, with no visual evidence of ID remapping. This is evidence from the displayed samples, not a proof over every image.

## I. Values requiring neutral tokens

`face_shape=1` requires the neutral token `face_shape_1` in the current candidate mapping. Human review may also elect to replace any visually ambiguous `face_color` candidate with its corresponding neutral token.

## J. Implications for deterministic captions

A future deterministic caption system can use an attribute-scoped canonical mapping and mix approved semantic labels with neutral tokens. This audit does not define the caption grammar, aliases, ordering, or final vocabulary.

## K. Implications for Guided UI

The four sheets provide an ID-indexed visual reference suitable for human approval of Guided Mode display labels. The UI should preserve attribute scope and retain the numeric ID internally; no label here is approved automatically.

## L. Implications for controlled natural-language UI

Only approved canonical labels should be accepted as controlled-language values. Aliases, free-form interpretation, parsing behavior, and synonym policy remain unresolved and are outside this audit.

## M. Unresolved human decisions

- Approve, revise, or replace each project-defined candidate label.
- Decide whether the five ambiguous `face_color` labels should become neutral tokens.
- Decide whether `face_shape=1` remains a neutral token or receives a human-approved descriptor.
- Decide future caption grammar, canonical ordering, aliases, tokenizer, and UI parsing separately.

## Visual-evidence protocol

Candidates were ordered lexicographically by basename. For each attribute/value and dataset, six positions were selected uniformly from the first through last compatible candidate using deterministic rounded indices. `face_shape` samples use `glasses=11` and `facial_hair=14`; `glasses` samples use `facial_hair=14`; color samples use `glasses=11` and `facial_hair=14` to reduce obstruction. These filters are audit-display controls, not future preprocessing or dataset rules.

Raw RGBA images were alpha-composited on pure white and converted to RGB. Sheets use 128×128 LANCZOS thumbnails only for readable display; colors, contrast, and geometry were otherwise unchanged. This is not frozen model preprocessing.

## Human-review handoff — all 40 values

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | pointed | VISUALLY_CLEAR | Stable tapered jaw and pointed chin across both datasets. |
| 1 | face_shape_1 | NEUTRAL_TOKEN_REQUIRED | Stable outline, but short labels such as angular or tapered are not sufficiently unambiguous. |
| 2 | oval | VISUALLY_CLEAR | Stable rounded oval outline across both datasets. |
| 3 | long | VISUALLY_CLEAR | Stable vertically elongated outline across both datasets. |
| 4 | round | VISUALLY_CLEAR | Stable compact near-round outline across both datasets. |
| 5 | wide | VISUALLY_CLEAR | Stable broad cheek and jaw outline relative to height. |
| 6 | rectangular | VISUALLY_CLEAR | Stable longer outline with comparatively straight sides. |

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | oversized glasses | VISUALLY_CLEAR | Large rounded frames are stable in both datasets. |
| 1 | aviator sunglasses | VISUALLY_CLEAR | Dark aviator-style lenses are stable in both datasets. |
| 2 | cat eye glasses | VISUALLY_CLEAR | Upswept cat-eye frame is stable in both datasets. |
| 3 | rimless glasses | VISUALLY_CLEAR | Minimal narrow rim appearance is stable in both datasets. |
| 4 | rectangular glasses | VISUALLY_CLEAR | Thin rectangular frame is stable in both datasets. |
| 5 | round glasses | VISUALLY_CLEAR | Small round frame is stable in both datasets. |
| 6 | thick rectangular glasses | VISUALLY_CLEAR | Thick rectangular frame is stable in both datasets. |
| 7 | oval glasses | VISUALLY_CLEAR | Small oval frame is stable in both datasets. |
| 8 | narrow sunglasses | VISUALLY_CLEAR | Narrow dark lenses are stable in both datasets. |
| 9 | round sunglasses | VISUALLY_CLEAR | Round dark lenses are stable in both datasets. |
| 10 | heart glasses | VISUALLY_CLEAR | Heart-shaped lenses are stable in both datasets. |
| 11 | no glasses | OFFICIAL | Official Google documentation explicitly states that type 11 is no glasses. |

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | light blond | VISUALLY_CLEAR | Stable light blond fill; observed base RGB 255,222,138. |
| 1 | golden blond | VISUALLY_CLEAR | Stable golden blond fill; observed base RGB 255,187,77. |
| 2 | orange | VISUALLY_CLEAR | Stable orange fill; observed base RGB 237,111,47. |
| 3 | red | VISUALLY_CLEAR | Stable red fill; observed base RGB 192,74,42. |
| 4 | golden brown | VISUALLY_CLEAR | Stable golden-brown fill; observed base RGB 216,154,50. |
| 5 | medium brown | VISUALLY_CLEAR | Stable medium-brown fill; observed base RGB 161,83,50. |
| 6 | dark brown | VISUALLY_CLEAR | Stable dark-brown fill; observed base RGB 91,51,28. |
| 7 | black | VISUALLY_CLEAR | Stable near-black fill; observed base RGB 30,21,20. |
| 8 | blue gray | VISUALLY_CLEAR | Stable blue-gray fill; observed base RGB 144,164,174. |
| 9 | silver | VISUALLY_CLEAR | Stable light-gray/silver fill; observed base RGB 224,224,224. |

| ID | Candidate label | Status | One-line rationale |
|---:|---|---|---|
| 0 | deep brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 96,61,48. |
| 1 | dark brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 136,89,75. |
| 2 | medium brown | VISUALLY_CLEAR | Stable graphic face fill; observed base RGB 170,116,84. |
| 3 | warm brown | VISUALLY_AMBIGUOUS | Stable RGB 199,139,93, but the short color name overlaps nearby palette terms. |
| 4 | golden tan | VISUALLY_AMBIGUOUS | Stable RGB 217,161,110, but the boundary between tan and light brown is naming-dependent. |
| 5 | light tan | VISUALLY_CLEAR | Stable light-tan graphic fill; observed base RGB 227,180,126. |
| 6 | rose tan | VISUALLY_AMBIGUOUS | Stable RGB 238,175,148, but the concise descriptor is not uniquely determined. |
| 7 | peach tan | VISUALLY_AMBIGUOUS | Stable RGB 240,192,146, but peach/tan wording requires human confirmation. |
| 8 | light beige | VISUALLY_CLEAR | Stable light-beige graphic fill; observed base RGB 246,216,193. |
| 9 | light peach | VISUALLY_AMBIGUOUS | Stable RGB 251,205,182, but it is close to adjacent pale warm colors. |
| 10 | pale pink | VISUALLY_CLEAR | Stable pale-pink graphic fill; observed base RGB 251,219,209. |

**SEMANTIC MAPPING EVIDENCE READY FOR HUMAN REVIEW**
