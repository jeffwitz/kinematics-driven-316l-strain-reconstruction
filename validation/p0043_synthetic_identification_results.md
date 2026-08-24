# P43-SYNTH-001 — résultat du smoke test synthétique

## Statut

Le premier smoke test sans bruit sur un crop P43 M20 est **numériquement
positif**, mais ne constitue pas encore une qualification complète de
l'identification. L'optimiseur a atteint le plancher du résidu avant la limite
de six évaluations demandées (`success=false` uniquement parce que cette limite
a été atteinte). Les tests de départs éloignés, de réduction à trois
paramètres, de mismatch de chemin et de bruit ne sont pas lancés.

## Configuration

- crop absolu `[1610:1630, 1075:1095]`, 400 nœuds et 800 points matériau ;
- EBSD réel P43, histoire DIC réparée, 9 états macro ;
- 32 incréments fixes (4 par segment macro) ;
- observation identité, sans bruit ;
- paramètres `eta=log(tau0,R,Q,b)`, bornes `[1/4,4]` autour de la vérité ;
- Jacobienne FEMU directe avec shadow FD `h=0.0015` ;
- 6 évaluations au maximum, `scipy.optimize.least_squares`.

Les détails complets et les empreintes sont dans
[`report.json`](reference_data/p0043_synthetic_identification_v1/report.json).

## Résultats

| quantité | vérité | départ | identifié après 6 évaluations |
|---|---:|---:|---:|
| `tau0` (MPa) | 40.000000 | 42.000000 | 40.000004 |
| `R` (MPa) | 18.781910 | 17.842815 | 18.781912 |
| `Q` (MPa) | 10.000000 | 10.800000 | 10.006888 |
| `b` | 3.000000 | 2.760000 | 2.997907 |

Le RMS du résidu whitened passe de `8.8502e-8` à `1.2693e-13` (vérité :
`0`). Les erreurs logarithmiques finales sont `9.0e-8`, `1.2e-7`, `6.89e-4`
et `-6.98e-4` pour `(tau0,R,Q,b)`.

Le calcul a nécessité 7 forwards et 7 Jacobiennes, soit `438.9 s` au total
(environ 50 s par Jacobienne directe et 14 s par forward). Ce coût est celui
du smoke M20 avec 32 incréments ; il ne doit pas être extrapolé directement à
M100.

## Observabilité

La SVD de la Jacobienne finale donne :

```text
(1, 0.135725, 0.036116, 1.0313e-4)
conditionnement = 9696.6
```

Le quatrième vecteur est dominé par le contraste `Q-b` (`0.7015, -0.7119`),
avec corrélation locale `rho(Q,b)=0.999999997`. Le smoke récupère donc la
réponse et les trois directions robustes, mais il ne démontre pas une
identification indépendante de `Q` et `b`.

## Décision

`P43-SYNTH-001/A` est **GO pour poursuivre la validation synthétique
progressive**, pas pour le P43 expérimental. La prochaine étape autorisée est
le test B (départs plus éloignés) puis le test C à trois paramètres. Aucun
bruit, transfert DIC ou campagne expérimentale ne doit être ajouté avant ces
contrôles.

