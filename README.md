# Tiny Text-Conditioned Avatar Diffusion from Scratch

Progetto universitario dedicato allo sviluppo da zero di un modello di diffusione condizionato dal testo per avatar. L'obiettivo generale è esplorare in modo riproducibile i componenti fondamentali del progetto.

Stato: under development.

Installazione e riproduzione saranno documentate progressivamente.

## DR03 — Text Conditioning

Stato: scientificamente congelato; closeout del repository in corso.

Il notebook canonico è `notebooks/DR03_Text_Conditioning.ipynb`. Le evidenze
compatte e versionabili sono in `reports/dr03_canonical/`:

- `dr03_summary.json`, `training_validation_metrics.csv` e
  `checkpoint_selection_kid.csv`;
- `training_validation_epsilon_mse.png`, `validation_kid_by_milestone.png` e
  `fixed_ema_preview_milestones.png`.

Per rigenerare soltanto le figure dai dati congelati, dalla root eseguire:

```bash
python scripts/render_dr03_figures.py
```

Il checkpoint selezionato è il 20k EMA, identificato da SHA256
`0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884`.
Checkpoint ed evidenze sperimentali di grandi dimensioni restano locali e non
sono normali sorgenti versionate.
