# P43-SYNTH-002B — résultat multi-départs M20

Les quatre départs éloignés ont tous convergé (`least_squares success=true`).
Les RMS finaux sont compris entre `4.8e-17` et `5.0e-13`.

| départ | tau0 (MPa) | R (MPa) | Q (MPa) | b | RMS final | nfev |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 40.000000 | 18.781910 | 10.000003 | 2.999999 | 5.26e-17 | 9 |
| B2 | 39.999986 | 18.781901 | 9.973054 | 3.008216 | 4.98e-13 | 9 |
| B3 | 40.000000 | 18.781910 | 10.000001 | 3.000000 | 4.84e-17 | 7 |
| B4 | 40.000000 | 18.781910 | 9.999997 | 3.000001 | 7.25e-17 | 7 |

`B2` illustre la vallée faible `Q-b` : il s'écarte davantage dans cette
combinaison tout en conservant un résidu pratiquement nul. La projection de
l'erreur sur le quatrième vecteur singulier vaut `-3.84e-3`, alors que les
trois directions fortes restent proches de zéro.

Conclusion : B est positif pour le bassin d'attraction et confirme la
dégénérescence `Q/b`. La récupération individuelle de `Q` et `b` n'est pas
revendiquée.

Artefact primaire :
[`report.json`](reference_data/p0043_synthetic_multistart_v1/report.json).
