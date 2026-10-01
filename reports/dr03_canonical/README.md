# DR03 canonical evidence

This directory contains the compact frozen DR03 evidence used by the notebook,
report, later evaluation, and checkpoint provenance. Large local experiment
outputs and checkpoints are deliberately excluded.

## Files

- `dr03_summary.json`: frozen metadata and provenance.
- `training_validation_metrics.csv`: raw training epsilon-MSE, trailing-100
  smoothing, and RAW validation diagnostics.
- `checkpoint_selection_kid.csv`: validation-only EMA checkpoint-selection KID.
- `training_validation_epsilon_mse.png`: canonical optimization/validation curve.
- `validation_kid_by_milestone.png`: validation-only checkpoint-selection curve.
- `fixed_ema_preview_milestones.png`: fixed-prompt/fixed-seed qualitative panel.

## Reproduction

From the repository root run:

```bash
python scripts/render_dr03_figures.py
```

The renderer only renders figures from frozen artifacts; it does not train,
sample, or recompute KID.

## Scientific boundaries

- KID uses 256 samples per split on ID_VAL and COMP_VAL only.
- ID_TEST and FINAL_OOD were not used.
- This KID is not final project KID; previews are qualitative diagnostics only.
- Conditioning adherence has not yet been evaluated.
- The selected checkpoint is 20k EMA.

## Validated local environment and Clean-FID note

The local experiments used Python 3.12.10, PyTorch 2.5.1+cu121, torchvision
0.20.1+cu121, torchaudio 2.5.1+cu121, Clean-FID 0.1.35, and an NVIDIA RTX 4060
Laptop GPU. On Windows, Clean-FID checkpoint evaluation required
`num_workers=0`: multiprocessing in the resizing pipeline raised
`Can't pickle local object 'make_resizer.<locals>.func'`.

An unrestricted `clean-fid==0.1.35` install previously replaced the validated
PyTorch stack and caused a Windows CUDA/DLL failure. Preserve the tested
PyTorch/torchvision stack and review dependency changes explicitly rather than
blindly reinstalling Clean-FID.
