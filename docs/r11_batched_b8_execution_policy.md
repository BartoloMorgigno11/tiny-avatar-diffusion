# R11 fixed-B8 execution policy

R11 uses batch size 8 for both arms with FP32, 1,000-step ancestral pixel-space
DDPM, frozen canonical seeds/tokens, EMA checkpoints, evaluator, KID, and ranking.
Canonical batches are deterministic and resume retains their original boundaries.

This is a distinct deterministic execution profile. It does not claim bitwise
equivalence to legacy serial sampling: strict serial-parity thresholds were not
met. The bounded frozen-evaluator check found zero attribute disagreements over
16 serial/B8 pairs. Historical R11 artifacts remain separate and immutable.
