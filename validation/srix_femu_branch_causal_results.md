# E-SRIX-FEMU-BRANCH-002B — localisation causale

## Correction préalable

Le diagnostic a révélé un défaut dans l'initialisation des chemins fixes : un
`initial_displacement` plein champ annulait aussi les valeurs de bord imposées
par le premier pas. Le solveur ne doit modifier que les inconnues intérieures.
La correction est testée par
`test_initial_displacement_guess_does_not_cancel_first_boundary_step`.
Les campagnes historiques ne sont pas réécrites.

## Résultat corrigé

Avec cette correction, le chemin 57 de référence échoue déjà à l'incrément 5
(`f=0.15625`) et le chemin entièrement raffiné échoue à l'incrément 28
(`f=0.2421875`). Les deux histoires ne peuvent donc pas être comparées jusqu'à
`f≈0.237` comme si le niveau 57 était un forward convergé.

Aux quatre endpoints communs disponibles, la divergence est d'abord faible,
puis devient constitutive :

| fraction | erreur contrainte | erreur `g` | changements d'activité |
| ---: | ---: | ---: | ---: |
| 0.03125 | `2.89e-10` | `0` | `0` |
| 0.06250 | `1.45e-10` | `0` | `0` |
| 0.09375 | `1.90e-3` | `3.31e-1` | `0` |
| 0.12500 | `7.17e-3` | `2.01e-1` | `1` |

Le premier signal n'est donc pas une dérive régulière minuscule : une
différence de l'histoire constitutive apparaît vers `f=0.09375`, puis un
changement d'activité est observé à `f=0.125`. Cela constitue un **candidat**
à une transition active, pas encore une démonstration de bifurcation physique.

Les chemins hybrides raffinant les `k` premiers intervalles échouent tous dans
la configuration corrigée ; le point d'échec se déplace selon `k` mais aucun
préfixe ne fournit un chemin validé. La classification reste donc
`unresolved`, et non `active_set_transition` confirmée.

Les observables brutes et les états communs sont archivés dans
[`srix_femu_branch_causal_v2`](reference_data/srix_femu_branch_causal_v2/),
notamment `common_endpoint_states.npz`. Aucune identification, aucun calcul
P43 et aucune reprise de PATH-002 ne sont autorisés.
