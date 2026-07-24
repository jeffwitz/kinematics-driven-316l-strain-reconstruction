# Sous-domaine de l'article — 100 partitions, padding 150, partition 0

Statut : **calcul convergé et contrôlé le 2026-07-24**.

Ce répertoire conserve un calcul réel effectué depuis le ROI DIC complet
versionné. Il ne s'agit ni d'un cas synthétique ni d'un benchmark homogène.

Configuration géométrique :

- ROI global : `3600 × 3100` éléments ;
- grille : `10 × 10`, soit 100 partitions ;
- partition : `0`, coin inférieur suivant la convention du dépôt ;
- cœur conservé : `360 × 310` éléments ;
- padding : `150` éléments ;
- zone effectivement résolue : `510 × 460`, soit `234 600` éléments.

Tous les champs finaux du solveur doivent être conservés :

- `U` : déplacements nodaux ;
- `S` : contraintes ;
- `E` : déformations totales ;
- `PE` : déformations plastiques ;
- `PEEQ` : déformation plastique équivalente ;
- `RF` : réactions nodales.

Le manifeste du workflow est écrit avant la résolution. Les tableaux sont
écrits atomiquement et leur SHA-256 est enregistré dans `status.json`.
`run.log` conserve le journal applicatif et `resource-usage.txt` la mesure
`/usr/bin/time -v`.

## Résultat

- 20 incréments convergés sur 20, sans cutback ;
- 113 itérations de Newton, au plus 6 par incrément ;
- résidu relatif final : `3,183e-9` ;
- temps solveur : `1088,13 s` ;
- temps mur processus : `1089,80 s` ;
- pic RSS : `3 768 132 KiB` ;
- équilibre relatif des réactions : `4,395e-14` ;
- erreur maximale des déplacements prescrits sur le bord :
  `4,163e-17 mm`.

Les contrôles d'intégrité, statistiques et métriques scientifiques sont
enregistrés dans `validation-report.json`. Les quatre cartes dérivées sous
`derived/` conservent les déformations équivalentes DIC et EF, leur différence
signée et la contrainte de von Mises.

## Comparaison exploratoire avec les valeurs de l'article

La déformation équivalente est reconstruite à partir des déplacements DIC et EF
avec le même opérateur de petites déformations, un pas de `1,84 µm`, la
contrainte plane et `nu = 0,3`.

Sur toute la zone résolue :

- RMSE : `0,253` point de pourcentage ;
- MAE : `0,185` point de pourcentage ;
- corrélation spatiale : `0,016`.

L'article donne respectivement `0,220` et `0,156` pour le ROI complet raccordé
avec 100 partitions et un padding de 150. La proximité des erreurs est
encourageante, mais la faible corrélation spatiale et le fait qu'une seule
partition de coin soit calculée interdisent de conclure à une reproduction.
Il faut exécuter et raccorder les 100 partitions, puis appliquer exactement le
masque et les conventions métriques de l'article.

`preview.png` permet l'inspection visuelle immédiate des cartes. Les pointillés
délimitent le cœur `360 × 310` qui serait conservé au raccordement.

## Fichiers conservés

- `run-request.json` : intention, commit et paramètres avant calcul ;
- `manifest.json` : disposition complète des 100 partitions et empreintes ;
- `preflight.log` : contrôle avant lancement ;
- `run.log` : journal du solveur ;
- `resource-usage.txt` : temps, CPU et mémoire mesurés ;
- `partitions/0000/status.json` : convergence et SHA-256 des six champs ;
- `partitions/0000/*.npy` : résultats bruts exhaustifs ;
- `derived/*.npy` : cartes scientifiques dérivées ;
- `validation-report.json` : rapport machine-readable ;
- `preview.png` : synthèse visuelle.
