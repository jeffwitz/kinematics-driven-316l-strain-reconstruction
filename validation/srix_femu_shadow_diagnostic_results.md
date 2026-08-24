# E-SRIX-FEMU-SHADOW-003 — résultat

Le forward L3 strict converge sur 809 incréments. La Jacobienne directe ne
peut toutefois pas être évaluée : le premier échec est localisé sans
ambiguïté à l’incrément accepté 271, soit
`[0.232177734375, 0.2322998046875]`, pour l’ombre `tau0−`, dans la phase
`fixed_current_strain`. MFront retourne `status -1`.

La phase `history_advance` n’est donc pas atteinte dans ce run. Il ne s’agit
pas d’un échec du forward principal.

Sur L2 (392 incréments), les trois pas logarithmiques pré-enregistrés passent :

| h | statut | temps (s) | résolutions GMRES |
|---:|:---:|---:|---:|
| 0.003 | OK | 123.87 | 1568 |
| 0.0015 | OK | 123.76 | 1568 |
| 0.001 | OK | 121.37 | 1568 |

La réduction de `h` ne restaure donc pas un échec sur L2. Aucun changement de
`h` n’est adopté et aucune conclusion de convergence L2→L3 des sensibilités
n’est tirée ici ; ce diagnostic compare les statuts et les coûts, pas les
matrices de Jacobienne.

## Décision

Le blocage est classé **shadow FD localisé sur chemin très raffiné**, et non
comme échec du forward mécanique. Aucun niveau L4, aucune identification et
aucun calcul P43 ne sont autorisés. La suite rationnelle est soit un rejeu
local de `tau0−` autour de l’incrément 271 avec télémétrie MFront, soit
l’ouverture du provider de sensibilité constitutive analytique ; il n’est pas
justifié de modifier les tolérances ou de relâcher l’oracle.
