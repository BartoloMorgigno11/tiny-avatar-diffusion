# Academic LaTeX report

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
