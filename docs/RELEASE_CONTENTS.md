# Professor-facing release contents

This release contains exactly four public EMA endpoints: DDPM-U32, DDPM-C32,
DDPM-U64 and DDPM-C64. Their immutable identities are recorded in
`src/tiny_avatar_diffusion/public_model_manifest.json`.

## Included

- package, CLI, four-model Streamlit UI, public adapters and selected CPU tests;
- four notebooks in `notebooks/final/` and their compact saved inputs;
- DR01 tokenizer/split metadata and frozen DR04/DR05 summaries;
- the compact STEP 9B endpoint comparison and STEP 13 unconditional summary read by notebooks 03--04;
- definitive PNG/PDF exports from all four final notebooks;
- the 20-image conditional 4x5 gallery and its provenance manifest, plus the
  eight-image final four-model gallery and its provenance manifest, solely for
  CPU notebook visualization and hash verification;
- report LaTeX source and figure assets, plus selected release documentation;
- four public checkpoint packages, each verified against its manifest SHA-256.

Public checkpoints are named `ddpm-u32-ema-40k.pt`, `ddpm-c32-ema-40k.pt`,
`ddpm-u64-ema-60k.pt`, and `ddpm-c64-ema-60k.pt`; historical local filenames
are not part of the release index.

## Excluded

- raw CartoonSet archives, caches, generated populations and training outputs,
  except for the 28 explicitly listed provenance-recorded qualitative PNGs;
- blind/internal review packages, agent scaffolding and local test caches;
- historical scripts with absolute paths or requiring sibling worktrees;
- `scorer_sources.zip` and all unreviewed third-party/source archives.

## External delivery item

No compiled final report PDF is present in this working directory. If required,
add a reviewed PDF to the delivery package without replacing the report source
or any frozen scientific record.

## Notebook execution boundary

All four notebooks execute their default path on CPU from packaged manifests,
frozen summaries, figures and qualitative PNGs; they do not train, sample or
rerun evaluators. Optional inference/regeneration cells are disabled. They need
the documented CUDA environment and, for historical intermediate endpoints,
experimental checkpoints that are not among the four public EMA packages.
