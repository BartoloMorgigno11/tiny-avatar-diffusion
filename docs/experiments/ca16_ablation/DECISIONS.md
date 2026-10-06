# CA16 Ablation Decision Register

| ID | Decision | Status | Rationale / context | Date |
|---|---|---|---|---|
| CA16-D01 | Compare CA8 control against CA8 + CA16 treatment only. | FROZEN | One focused intervention: retain V1 CA8 and add decoder16 image-to-text cross-attention. | 2026-10-06 |
| CA16-D02 | Treat conditioning adherence as primary; use COMP_VAL family-macro MAA as the primary outcome. | FROZEN | Targets the hypothesized semantic/compositional limitation without pooling families as raw records. | 2026-10-06 |
| CA16-D03 | Use seeds 2026, 2027, and 2028 in paired CA8/CA8+CA16 runs; require identical shared initialization and same-commit code. | FROZEN | Supports matched, seed-paired causal comparison. | 2026-10-06 |
| CA16-D04 | Select among 5k, 10k, 15k, and 20k milestones by the preregistered adherence-first lexicographic rule. | FROZEN | Prevents post-result checkpoint selection. | 2026-10-06 |
| CA16-D05 | Seal FINAL_OOD from all CA16 development and internal causal evaluation. | FROZEN | V1 FINAL_OOD informed the post-hoc hypothesis; COMP_VAL is the prospective internal domain. | 2026-10-06 |
| CA16-D06 | Do not attribute any effect to 16x16 location alone or claim its optimality. | FROZEN | The treatment jointly adds a second injection and approximately 25k parameters. | 2026-10-06 |
| CA16-D07 | Keep data regime, resolution, U-Net backbone width/depth, text encoder, objective, and primary training budget unchanged apart from the added CA16 module. | FROZEN | Keeps stated alternatives outside this focused ablation. | 2026-10-06 |
| CA16-D08 | Freeze a separate distribution-preserving quality-guardrail pool before training. | OPEN — TO BE FROZEN IN R2 BEFORE TRAINING | No exact size or sampling rule is established here. | 2026-10-06 |
| CA16-D09 | Freeze the diversity prompt-count/panel rule before full training. | OPEN — TO BE FROZEN BEFORE FULL TRAINING | No exact CA16 diversity-panel rule is established here. | 2026-10-06 |
