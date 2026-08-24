# E-SRIX-FEMU-SHADOW-003 — résultat

Le forward L3 strict converge sur 809 incréments. La Jacobienne directe ne
peut toutefois pas être évaluée : le premier échec est localisé sans
ambiguïté à l’incrément accepté 271, soit
`[0.232177734375, 0.2322998046875]`, pour l’ombre `tau0−`, dans la phase
`fixed_current_strain`. MFront retourne `status -1`.

La phase `history_advance` n’est donc pas atteinte dans ce run. Il ne s’agit
pas d’un échec du forward principal.

## SHADOW-003B — préfixe L3

Le chemin a été rejoué uniquement jusqu’à l’incrément 271 :

| h | résultat au pas 271 |
|---:|:---|
| 0.003 | échec `tau0−`, `fixed_current_strain`, status -1 |
| 0.0015 | passage du pas 271 |
| 0.001 | passage du pas 271 |

La réduction de `h` déplace donc effectivement le shadow hors de l’échec local
observé à `tau0−`. Cela ne prouve pas encore que le chemin L3 complet passe.

Sur L2 (392 incréments), les trois pas logarithmiques pré-enregistrés passent :

| h | statut | temps (s) | résolutions GMRES |
|---:|:---:|---:|---:|
| 0.003 | OK | 123.87 | 1568 |
| 0.0015 | OK | 123.76 | 1568 |
| 0.001 | OK | 121.37 | 1568 |

La réduction de `h` ne restaure donc pas un échec sur L2. Aucun changement de
`h` n’est adopté et aucune conclusion de convergence L2→L3 des sensibilités
n’est tirée ici. En revanche, la comparaison des matrices L2 est favorable :

| h | erreur maximale de colonne vs h=0.003 | cosinus minimal | conditionnement |
|---:|---:|---:|---:|
| 0.0015 | 0.204 % | 0.9999981 | 16043 |
| 0.001 | 0.246 % | 0.9999973 | 16047 |

Les spectres normalisés restent `0.18245/0.04358/6.23e-5` à moins de
quelques centièmes de pourcent. Les matrices sont archivées dans
`validation/reference_data/srix_femu_shadow_diagnostic_v1/`.

## Décision

Le blocage est classé **shadow FD localisé sur chemin très raffiné**. Les
résultats L2 rendent `h=0.001` ou `h=0.0015` défendables comme candidats, mais
aucun n’est adopté avant un rejeu L3 complet. Aucun niveau L4, aucune
identification et aucun calcul P43 ne sont autorisés. La prochaine action est
un L3 complet avec ces deux valeurs, puis un choix fondé sur convergence et
stabilité ; si elles échouent plus loin, on ouvre le diagnostic constitutif
analytique.
