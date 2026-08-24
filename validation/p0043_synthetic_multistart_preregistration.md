# P43-SYNTH-002B — pré-enregistrement multi-départs

Ce gate étend le smoke test `P43-SYNTH-001/A` sur le même crop P43 M20, la
même EBSD, la même histoire synthétique, la même partition de 32 incréments et
la même observation identité sans bruit.

Départs log-déterministes :

```text
B1 = (+20 %, -20 %, +25 %, -25 %)
B2 = (-20 %, +20 %, -25 %, +25 %)
B3 = (+25 %, +25 %, -20 %, -20 %)
B4 = (-25 %, -25 %, +20 %, +20 %)
```

Les quatre paramètres restent libres, avec les bornes `[1/4,4]` autour de la
vérité. La limite artificielle `nfev=6` du smoke est supprimée ; la limite
opérationnelle est `max_nfev=15`. Les calculs utilisent la Jacobienne FEMU
directe et `h=0.0015`.

Pour chaque départ sont archivés : paramètres initiaux/finals, coûts, statut
`least_squares`, nombre de forwards/Jacobiennes, SVD finale et projection de
l'erreur log-paramétrique sur les vecteurs singuliers. Aucun critère n'exige la
récupération individuelle de `Q` et `b` : leur direction faible est analysée
séparément. Ce gate ne concerne pas le P43 expérimental.

