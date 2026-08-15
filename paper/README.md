# Note DIC plein champ - local/global et identification

## Compilation

Prérequis : XeLaTeX et Biber.

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Ou, si `latexmk` est disponible :

```bash
latexmk -xelatex -use-biber main.tex
```

Le document utilise uniquement des figures TikZ intégrées : aucun asset externe n'est nécessaire.
