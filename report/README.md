# Report Build Guide

## Files

- `main.tex`: full report manuscript
- `references.bib`: bibliography database

## Compile locally

Use XeLaTeX + BibTeX:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## Notes

- `main.tex` uses Unicode Vietnamese text and `fontspec`, so XeLaTeX is recommended.
- Figures are loaded from `../outputs/figures/` and `../outputs/figures/baseline_robustness/`.
- If compiling in Overleaf, upload the whole `report/` folder and the relevant `outputs/figures` images (or adjust paths).

