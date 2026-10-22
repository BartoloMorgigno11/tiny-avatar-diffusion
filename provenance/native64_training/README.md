# Recovered Native64 Training Sources

This directory is a byte-preserving historical snapshot recovered on 2026-10-22 from local development worktrees.  It is evidence for the provenance of the released DDPM-C64 and DDPM-U64 inference checkpoints; it is **not** an integrated or supported public training interface.

The recovered files were untracked in their source worktrees.  Consequently, no retrospective Git commit history can be established for these exact files.  `INVENTORY.md` records their original absolute paths, bytes, SHA-256 digests, worktree state, and source/destination verification.

The C64 path is the matched-budget native64 trainer (microbatch 8 × accumulation 4, effective batch 32), followed by its 40k-to-60k continuation.  The recorded U64 run uses microbatch 16 × accumulation 2 (effective batch 32).  Both use AdamW (LR 2e-4, zero weight decay), FP16 AMP, EMA 0.9999, and a 1,000-step linear DDPM schedule.

No checkpoint, cache, dataset, manifest, result, or executable public entry point was copied or changed.  In particular, these scripts may retain historical absolute paths and should not be presented as standalone commands without a separately approved integration and validation task.
