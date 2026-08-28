# P43 — crossover `state + coupled block`

Les microbenchmarks préchauffés comparent le chemin vectorisé (`state` NumPy
puis `numba-fused` block) au kernel `numba-fused-state`, sur les mêmes données.

| points | ratio fused-state / référence |
|---:|---:|
| 800 | 1.528 |
| 2 000 | 1.765 |
| 5 000 | 1.538 |
| 10 000 | 0.796 |
| 20 000 | 0.781 |

Le premier balayage (v1) indiquait un gain de 20–22 % à 10k–20k points. Un
second balayage autour du crossover (v2/v3) est plus variable : ratios 1.43,
1.37, 1.38 à 6k, 8k, 10k, puis 0.65, 0.94, 0.92 à 12k, 16k, 20k. Le
crossover est donc réel mais dépend fortement de la charge machine et du
nombre de threads du kernel fused existant.

Le forward M100 correspondant donne `211.687 s`, contre `243.721 s` pour le
run fused-tangent précédent, avec un RAW RMS identique à `3.0e-15` près. Les
écarts de champs maximaux sont `8.5e-13 mm` en déplacement et `6.4e-10` en
déformation ; les trajectoires globales ne sont toutefois pas identiques.

Conclusion : un dispatch `auto` est disponible avec seuil configurable, par
défaut `12000` points actifs. Ce seuil est expérimental et doit rester
explicitement contrôlable ; `numba-fused` demeure la référence pour les petits
batches et les contrôles A/B.
