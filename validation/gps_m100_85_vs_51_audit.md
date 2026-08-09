# Audit du passage GPS M100 : 85 → 51 Newton

## Conclusion

Le résultat récent à 51 Newton ne constitue pas encore une comparaison de
performance avec l'artefact historique à 85 Newton : les deux exécutions ne
portent pas sur le même crop.

| Exécution | Crop | Newton | Sous-pas |
|---|---|---:|---:|
| historique (`261434455daaa1e6b69ae89fe434fac89cef5449`) | `[1570,1670,1035,1135]` | 85 | 400 points, 56 épisodes |
| test FD récent | `[1610,1710,1075,1175]` | 51 | 0 point |

Le test FD récent ne peut donc pas démontrer un gain : son option FD était
inactive dans le baseline et aucun point n'a déclenché de reconstruction.

## Différence de formulation identifiée

Le code MFront a changé entre l'artefact historique et l'état actuel.

Le comportement historique reposait sur une formulation augmentée avec les
déformations transverses comme inconnues d'état et une fermeture explicite.
Le commit `f14d87e` a remplacé cette formulation par un système de 18
inconnues :

- les lignes dans le plan portent la cinématique ;
- les trois lignes transverses portent directement les contraintes planes ;
- les déformations transverses sont reconstruites comme sorties.

Cette modification précède les ajouts FD et peut modifier directement le
critère d'échec d'une intégration complète, donc la décision de sous-pas.
Le commit `1df14a6` a ensuite également modifié `@Epsilon` de `1e-12` à
`1e-14`. Ces changements doivent être isolés avant toute conclusion sur le
FD composite.

## Options communes vérifiées

L'artefact historique indique :

- 8 incréments ;
- 4 threads MFront ;
- prédicteur transverse tangent ;
- LGMRES recyclé ;
- Eisenstat–Walker ;
- 40 Newton maximum ;
- résidu final `5.1381159314083e-09` ;
- `iterations_per_increment = [6,6,7,7,11,13,17,18]`.

Le rapport FD récent n'enregistre pas ces options et n'archive pas de champ
`.npz`. Il ne permet donc pas une comparaison de champs ni une preuve de
neutralité sur le même cas.

## État de l'enquête

Le FD composite n'est pas responsable du passage 85 → 51 dans le baseline :
il est désactivé et son compteur vaut zéro. La cause à tester en priorité est
la refonte MFront `f14d87e`, puis le changement de tolérance `@Epsilon`.

La prochaine comparaison propre doit rejouer, sans FD, le crop historique
`[1570,1670,1035,1135]`, avec archivage systématique des champs et des
diagnostics de sous-pas par incrément. Aucun nouveau résultat de performance
ne doit être annoncé avant ce contrôle.

## Rejeu exact effectué

Le rejeu exact a été réalisé sur le crop historique, sans FD, avec le même
backend GPS. Il donne :

- `85` Newton ;
- `[6,6,7,7,11,13,17,18]` par incrément ;
- `400` points sous-pasés ;
- `51` hits du cache et `5` misses ;
- résidu final `5.138178245452621e-09`.

La comparaison des champs avec l'artefact `261434455d...` donne :

| Champ | erreur relative L2 |
|---|---:|
| déplacement | `6.03e-13` |
| réaction | `6.97e-10` |
| contrainte en plan | `8.89e-10` |
| glissement signé par système | `1.51e-9` |
| glissement accumulé par système | `1.51e-9` |
| glissement accumulé agrégé | `1.40e-9` |

Le rejeu confirme donc que le mécanisme de sous-pas n'a pas disparu. Le
résultat à `51` Newton provenait uniquement du crop décalé
`[1610,1710,1075,1175]`. Le FD composite reste non testé sur ce rejeu.

## Test décisif du FD composite

Le même crop historique a ensuite été exécuté avec le FD composite activé
uniquement sur les points sous-pasés :

| Grandeur | Baseline | FD composite |
|---|---:|---:|
| Newton total | 85 | **58** |
| Newton par incrément | `[6,6,7,7,11,13,17,18]` | `[6,6,7,7,7,8,8,9]` |
| points FD | 0 | 192 |
| trajectoires FD | 0 | 1152 |
| changements de partition | 0 | 19 |
| résidu final | `5.138e-9` | `5.341e-9` |

Les différences relatives entre les champs finaux sont :

- déplacement : `2.19e-12` ;
- réactions : `6.27e-10` ;
- contraintes : `5.99e-9` ;
- glissement signé et équivalent par système : `2.17e-8` ;
- glissement accumulé agrégé : `1.70e-8`.

Le temps de cette exécution A/B était de `210.19 s` pour le baseline et
`135.92 s` avec FD. Ce ratio ne doit pas encore être publié comme gain
définitif : les temps M100 sont fortement variables selon la charge machine.
Le résultat robuste est la réduction de `85` à `58` Newton, obtenue avec
`192` points FD effectivement traités.

Artefact :
`validation/_generated/performance/gps_composite_fd_vs_gps_m100_same_crop.json`.

Interprétation : le critère `N <= 60` est atteint. Le tangent du dernier
sous-pas expliquait donc une part majeure de la pénalité Newton M100. Les
changements de partition FD et l'écart des glissements à `2.2e-8` nécessitent
encore une qualification de robustesse avant de sélectionner cette option par
défaut.

Une répétition complète indépendante a confirmé exactement `85 → 58` Newton,
avec `[6,6,7,7,7,8,8,9]` pour la voie FD, `192` points et `1152`
trajectoires. Son temps était de `293.64 s` pour le baseline et `220.92 s`
pour le FD. Les deux autres répétitions prévues n'ont pas produit de mesures
valides et ne sont pas incluses dans une statistique temporelle.
