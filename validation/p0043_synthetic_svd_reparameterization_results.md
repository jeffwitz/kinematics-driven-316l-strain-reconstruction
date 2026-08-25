# E-SRIX-P43-SYNTH-SVD-001 — résultats

Le rapport primaire est produit par
`scripts/qualify_srix_p0043_synthetic_svd_reparameterization.py` dans
`validation/reference_data/p0043_synthetic_svd_reparameterization_v1/`.

Le gate reste synthétique uniquement. C1 est positif sur les quatre départs :
les RMS finaux sont `8.65e-18`, `3.32e-16`, `9.56e-18` et `2.45e-17`, avec une
erreur rang-3 maximale de `1.61e-9` en coordonnées log. Ni `Q` ni `b` ne sont
fixés : les quatre paramètres reconstruits restent variables pendant
l'optimisation.

La base M20 a pour spectre `(1,0.135725,0.036116,1.031e-4)`. L'alignement de
`v4` avec `log(Q)-log(b)` vaut `0.999977`, et l'alignement de `v3` avec
`log(Q)+log(b)` vaut `0.998844`.

C3, avec `z1..z3` fixés à la meilleure solution C1, donne :

| z4 | RMS |
|---:|---:|
| -0.30 | `4.53e-11` |
| -0.20 | `2.84e-11` |
| -0.10 | `1.35e-11` |
| 0.00 | `8.65e-18` |
| +0.10 | `1.26e-11` |
| +0.20 | `2.47e-11` |
| +0.30 | `3.68e-11` |

La direction faible modifie fortement `Q` et `b` (par exemple `Q=8.10,
b=3.71` à `z4=-0.3`, `Q=12.35,b=2.42` à `z4=+0.3`) sans dégrader
significativement le champ synthétique. C1 est donc qualifié ; C3 confirme
que `Q` et `b` ne sont pas séparément identifiables sur cette observation.

Les détails sont dans `report.json`, `weak_mode_profile.csv` et
`weak_mode_profile.png`.
