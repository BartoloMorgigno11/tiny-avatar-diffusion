# Four-model benchmark input contract

> **Historical protocol notice.** This document preserves the DR07 planning
> contract and its historical CA8/H7/V2-A/V2-B model set; it is not the public
> inference or benchmark contract for this release. The current official
> public models are **DDPM-U32, DDPM-C32, DDPM-U64, and DDPM-C64**, identified
> by `src/tiny_avatar_diffusion/public_model_manifest.json`. Historical
> benchmark execution also requires assets that are not distributed with the
> four public checkpoints.

`tiny_avatar_diffusion.four_model_benchmark` is a CPU-only planning and
validation module. It does not open protected manifests, load models, generate
images, or calculate metrics.

Version 2 is the preserved DR07 protocol. Version 1 (`8e5eab23ae8eec213443baa4d1637cb666e6ffe65081edc7573627cf91932ae`) remains immutable provenance for historical DR07 artifacts and must not be silently remapped.
Version 2 binds the public frozen-H7 geometry and the four-model native-resolution
rules, not a sealed H7 plan file. The manifest hash for a production run is
computed only after a separately authorized case/reference plan is supplied.

The four-model primary contract fixes these model identities: CA8-40k/native32;
H7 CA8+CA16/native32/40k; V2-A CA8+CA16/native64/40k; V2-B
CA8+CA16+CA32/native64/40k. Every model receives identical
`evaluation_case_id` values for a given configuration, prompt ID, and seed.
CA8-20k is a separately named baseline for the unconditional-20k comparison.

The full configuration fixes the public H7 population shape of 252 prompts and
seeds 2026--2030. It requires an independently authorized sealed case list;
this module never discovers protected membership. Development configurations
are caller-supplied and deterministic.

The quality contract separately fixes the public H7 envelope: 10,000 generated
slots (5,767 ID and 4,233 OOD), 808 real references (466 ID and 342 OOD),
16-image batches, and batch seed `502026 + batch_index`. It preserves the
authorized plan's prompt/reference/namespace fields verbatim and hashes that
sealed input. The source code available here does not expose the authoritative
quality prompt distribution or per-slot seed composition, so exact H7-quality
reconstruction is intentionally blocked rather than substituted.

Real reference records require both 32px and 64px versions for every original
source image ID, sourced from the same archive and split. Both use direct RGB
conversion and one original-to-target Pillow bilinear resize. V2 native64 is
not primarily downsampled to 32px.

The shared frozen dual-view evaluator is
`results/runs/dr04_cartoonset100k_dualview_full_seed2026_50ep_r1/best.pt`
(`963a8c3592a91c3b43928cfd0728fdd6e9113a78136ebd01276478e01561d075`).
It scores V1/H7 through official RGB32→RGB64 bilinear view and V2-A/V2-B through
direct native RGB64 view, using the same CNN weights. Legacy Evaluator64 is not
valid for final benchmark conditioning metrics.

Metric result namespaces are conditioning (`maa`, per-attribute accuracy,
`psr`, `macro_psr`), native-resolution quality (`fid`, `kid`), diversity
(`lpips`), and efficiency. FID/KID32 and FID/KID64 are distinct metric spaces
and must not be numerically ranked against each other. Training budget remains
a V1-versus-40k confounder.

LPIPS uses the frozen conditioning population: 252 prompts x five seeds, every
unordered seed pair per prompt (10 pairs/prompt; 2,520 pairs total), with 152
ID prompts, 100 OOD prompts, and 25 prompts in each OOD family. Pair IDs are
identical across all four models while input images remain at each model's
native resolution. The sealed adapter refuses any absent, H7-only, wrong-model,
or wrong-protocol authorization before it invokes its loader.

## Native-quality execution

`tiny_avatar_diffusion.native_quality_metrics` executes Clean-FID `0.1.35`,
KID (seed `2026`, 100 subsets), and LPIPS AlexNet `0.1.4` from validated native
PNG pools. It reuses the existing H7/DR05 Clean-FID and LPIPS logic. Generated
V1/H7 files must be RGB 32x32; V2-A/V2-B files must be RGB 64x64. The executor
never resizes generated samples.

Real references are materialized through an injected original-source loader:
RGBA/RGB source to RGB, followed by one direct Pillow bilinear resize to 32px
and 64px. Each output stores source-file, pixel, PNG, archive, split, and
preprocessing provenance. Before any metric constructor is called, the executor
requires local Clean-FID Inception, ImageNet AlexNet, and LPIPS linear weights;
it fails closed rather than downloading them.

The executable output namespaces are `native32.quality`, `native64.quality`,
`native32.diversity`, and `native64.diversity`. Values across the two native
resolutions are not direct numerical rankings.
