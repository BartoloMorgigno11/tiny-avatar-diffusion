# H7: 20k-to-40k continuation results

This release candidate reports a three-seed paired comparison from the versioned
snapshot [`data/h7_results_v1.json`](data/h7_results_v1.json). The snapshot was
strictly checked against frozen CA16 and H7 completion markers before publication.

## Results

The primary endpoint, COMP_VAL equal-family macro MAA, increased for all seeds:
2026: 0.670254 to 0.967153; 2027: 0.791942 to 0.973510; and 2028: 0.750148 to
0.960966. The mean paired change was +0.229762 (sample SD 0.059954).

COMP_VAL MacroPSR and ID_VAL MAA also increased for every seed. KID and the
secondary FID decreased in the favourable direction for ID, COMP, and GLOBAL in
all three seeds. Complete per-seed values, paired differences, marker hashes,
and protocol references are in the JSON snapshot. Figures are in
[`figures/`](figures/).

## Interpretation and limits

COMP_VAL contributed to historical checkpoint selection, so this is not
independent held-out confirmation. Three training seeds provide limited
replication evidence. KID and FID are distributional proxies, not perceptual
ratings or a diversity guarantee; FID is secondary and sample-size sensitive.

Historical 20k ID_VAL PSR and CV1-CV4 family summaries have no currently
traceable authoritative persisted source. They are represented as `null` in the
snapshot; no paired comparison is made. The CV1-CV4 figure therefore presents
verified 40k diagnostics only.

The R13 calculation uses the validated batch-8 compatibility correction. An
earlier interrupted batch-1 H7 pool was isolated and never used for KID/FID.
The correction demonstrated algorithmic compatibility, not byte identity across
different checkpoints.

## Reproduction

Snapshot-only figure reproduction is CPU-only and needs only Python plus
Matplotlib:

```powershell
& <python-with-matplotlib> scripts/generate_h7_report.py `
  --snapshot docs/experiments/data/h7_results_v1.json `
  --figures docs/experiments/figures
```

It does not reproduce training or evaluation; it renders the packaged snapshot.
To verify the snapshot read-only against external frozen artifacts:

```powershell
& <python-with-matplotlib> scripts/generate_h7_report.py `
  --snapshot docs/experiments/data/h7_results_v1.json `
  --figures docs/experiments/figures `
  --verify --ca16-root <CA16-root> --h7-root <H7-root>
```

The verification mode fails rather than falling back to snapshot data if the
frozen sources do not agree.
