# Academic LaTeX report

> **Repository-source notice.** This directory is a historical, non-authoritative
> LaTeX source snapshot retained with the release. It is not the separately
> finalized scientific report PDF and must not be represented as that PDF.
> The current official public models are **DDPM-U32, DDPM-C32, DDPM-U64, and
> DDPM-C64**; see `src/tiny_avatar_diffusion/public_model_manifest.json` from
> the repository root for their immutable release identities.

`main.tex` is the main report source. The project is modular: metadata and the
preamble are at the root, chapters are in `sections/`, appendices are in
`appendices/`, and provenance-verified images belong in `figures/`.

The intended compilation command, from this directory, is:

```powershell
latexmk -pdf -use-biber main.tex
```

Alternatively:

```powershell
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The final report uses only authenticated frozen records.  The Overleaf package
is self-contained: its figures are local copies whose hashes are recorded in
`FINAL_FIGURE_TABLE_INVENTORY.md`.
