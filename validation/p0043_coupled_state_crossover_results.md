# P43 — crossover `state + coupled block`

Le microbenchmark préchauffé compare le chemin vectorisé (`state` NumPy puis
`numba-fused` block) au kernel `numba-fused-state`, sur les mêmes données.

| points | ratio fused-state / référence |
|---:|---:|
| 800 | 1.528 |
| 2 000 | 1.765 |
| 5 000 | 1.538 |
| 10 000 | 0.796 |
| 20 000 | 0.781 |

Le kernel est défavorable sur M20 mais gagne environ 20–22 % à la taille
M100 représentative. Le forward M100 correspondant donne `211.687 s`, contre
`243.721 s` pour le run fused-tangent précédent, avec un RAW RMS identique à
`3.0e-15` près. Les écarts de champs maximaux sont `8.5e-13 mm` en
déplacement et `6.4e-10` en déformation.

Conclusion : conserver `numba-fused-state` comme chemin explicitement
activable pour les grands batches ; conserver `numba-fused` comme référence
rapide pour les petits batches et pour les contrôles A/B.
