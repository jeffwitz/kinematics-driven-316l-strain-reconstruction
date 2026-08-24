# E-SRIX-FEMU-BRANCH-002A — diagnostic local autour de `f ≈ 0.237`

## Résultat

Le diagnostic local M8 a été exécuté autour de l'intervalle parent
`[0.234375, 0.23828125]`, sans identification ni calcul P43. Les cinq
partitions locales (`alpha = 0.25, 0.40, 0.50, 0.60, 0.75`) convergent, y
compris `alpha = 0.50`, qui correspond au second demi-pas ayant bloqué le
raffinement global à 114 pas.

Cela montre que l'échec du chemin 114 n'est pas expliqué par ce seul demi-pas
local. Il dépend aussi de l'état atteint après le raffinement des autres
intervalles du chemin.

| alpha | déplacement vs chemin 57 | contrainte vs chemin 57 | déformation plastique vs chemin 57 |
| ---: | ---: | ---: | ---: |
| 0.25 | `8.09e-6` | `7.30e-5` | `4.97e-4` |
| 0.40 | `1.04e-5` | `9.36e-5` | `6.38e-4` |
| 0.50 | `1.09e-5` | `9.77e-5` | `6.66e-4` |
| 0.60 | `1.05e-5` | `9.41e-5` | `6.42e-4` |
| 0.75 | `8.30e-6` | `7.42e-5` | `5.07e-4` |

La dispersion entre ces cinq partitions reste faible mais non nulle :
`2.80e-6` pour le déplacement, `2.47e-5` pour la contrainte et `1.70e-4`
pour la déformation plastique. Elle ne démontre donc pas une seconde branche
constitutive distincte, mais elle interdit encore de traiter le chemin 57
comme une limite convergée par rapport au pas de chargement.

Les deux prédicteurs globaux testés n'ont pas rétabli la convergence du chemin
local : l'extrapolation échoue à l'incrément 44 et l'état final grossier comme
prédicteur échoue à l'incrément 18. Ces essais ne copient volontairement pas
l'état constitutif du calcul grossier ; ils ne permettent donc pas de conclure
sur une différence de branche matérielle.

## Décision

`BRANCH-002A` est classé **diagnostic de continuation numérique incomplet** :
les partitions locales convergent vers des états proches, mais la convergence
par raffinement global n'est pas établie. Le chemin 114 reste bloqué et aucun
calcul 228 pas, aucune identification et aucun calcul P43 ne sont autorisés.

Le rapport machine-readable et la figure sont dans
[`srix_femu_branch_local_v2`](reference_data/srix_femu_branch_local_v2/).
La figure compare les écarts d'endpoint aux cinq partitions et ne remplace
pas une qualification de convergence.

## Limites

L'observateur actuel expose contrainte, déformation et tenseur de déformation
plastique, mais pas les tableaux internes SRIX bruts `g`, `p` et `a`. La
comparaison des prédicteurs porte uniquement sur le déplacement initial global
; l'état constitutif du calcul grossier n'est jamais injecté dans le calcul
raffiné.
