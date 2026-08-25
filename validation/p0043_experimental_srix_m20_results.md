# P43-EXP-001 — résultat M20 expérimental

## Statut

Le calcul M20 est terminé, mais le gate M20→M100 est **NON-GO**. Aucun calcul
M100 expérimental n'a été lancé.

Le rapport machine-readable est
`validation/reference_data/p0043_experimental_srix_m20_v1/report.json`.

## Résultat numérique

Le prior `(tau0,R,Q,b)=(40,18.7819100705,10,3)` donne un RMS whitened de
`0.05026295`. Les trois départs rang-3 diminuent le coût, mais atteignent la
limite volontaire `max_nfev=8` :

| départ | RMS initial | RMS final | statut |
|---|---:|---:|---|
| E1 | 0.05014629 | 0.04904369 | limite nfev |
| E2 | 0.05037622 | 0.04904377 | limite nfev |
| E3 | 0.05022758 | 0.04904358 | limite nfev |

Les trois trajectoires poussent les coordonnées réduites associées à `R` et à
la combinaison `Q/b` vers la borne inférieure (`z≈-log(4)`). Les solutions
physiques correspondantes sont environ :

```text
tau0 = 8.6–8.7 MPa
R    = 12.7–12.9 MPa
Q    = 3.86 MPa
b    = 1.17
```

Ce comportement est pathologique pour le gate : il ne démontre ni convergence
vers une solution expérimentale, ni stabilité des paramètres observables.

La SVD au prior vaut environ
`(1, 0.1357, 0.0361, 1.03e-4)` ; à la meilleure solution elle devient
`(1, 0.1431, 0.0165, 1.55e-5)`, ce qui indique en outre une rotation et un
affaiblissement de la géométrie locale pendant la tentative.

## Décision

```text
M20 completed: true
M20 gate: false
M100 authorized: false
experimental parameters identified: false
```

Le résultat est un diagnostic de non-adéquation du prior/modèle à cette chaîne
d'observation M20, pas une identification du lot P43. Avant toute relance, il
faudra décider d'un diagnostic ciblé (bornes/prior, observation et états DIC,
ou modèle constitutif) ; il est interdit de lancer M100 en l'état.
