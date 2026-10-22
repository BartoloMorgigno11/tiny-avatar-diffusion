# Report figures

This directory contains the report-ready derived figures and the preserved
authenticated historical assets. Their source paths, hashes, captions and use
limitations are listed in `../FINAL_FIGURE_TABLE_INVENTORY.md`.

Publication-ready Notebook 03 exports are:

- `conditional_ablation_ood_metrics.{png,pdf}` — frozen STEP 10 / STEP 9B OOD
  adherence values; E1--E5 key and source provenance are printed in the figure.
- `conditional_ablation_common32_quality.{png,pdf}` — frozen Common-32 KID/FID
  values; it is categorical, not a continuous training curve or native-quality
  comparison.
- `conditional_ablation_ood_4x5_seed2026.{png,pdf}` — a qualitative 4×5
  composition made solely from the twenty hash-recorded PNGs in
  `results/qualitative/conditional_ablation_ood_4x5_seed2026/`. It is not a
  benchmark and does not replace frozen metrics.

Notebook 01 exports `dataset_split_composition.{png,pdf}` and
`compositional_ood_holdout.{png,pdf}` from the frozen DR01 aggregate manifests.
Notebook 02 exports `ddpm_linear_schedule.{png,pdf}` and
`ddpm_architecture_pathway.{png,pdf}` from the implemented schedule and model
contracts; neither is a measured scientific result.

Notebook 04 exports `final_conditional_adherence.{png,pdf}` and
`final_distributional_quality.{png,pdf}` from the named frozen STEP 9B/STEP 13
records. Its `final_four_model_{unconditional,conditional}_gallery.{png,pdf}`
panels are compositions of the hash-recorded native PNGs and manifest in
`results/qualitative/final_four_model_gallery_seed2026/`. All four notebooks
can display their packaged assets on CPU; optional regeneration/inference cells
remain disabled and require the documented GPU/checkpoint environment.

The STEP 9A development diagnostic remains development-only. The STEP 9B
conditional and native64 panels remain authenticated historical illustrations;
their original PNG bytes are not edited here.
