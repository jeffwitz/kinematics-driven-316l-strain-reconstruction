# P43-SYNTH-003 — résultat de montée M100

Le calcul M100 est terminé. L'artefact primaire est
`validation/reference_data/p0043_synthetic_scaleup_v1/report.json`.

Le rapport doit être lu avec le statut de `P43-SYNTH-002B` : l'initialisation
M100 est le meilleur jeu identifié sur le crop M20, et la limite de quatre
évaluations est un diagnostic de transfert d'échelle, pas une qualification
finale multi-départs.

Le crop `[1580:1680,1030:1130]` (100×100) a convergé en 3 évaluations,
avec une durée de `18282.4 s` (~5.08 h). L'initialisation issue de M20 avait
un RMS de `3.38e-16`; le RMS final est `3.25e-18`. Les paramètres finaux sont
`tau0=40.000000`, `R=18.781910`, `Q=10.000000`, `b=3.000000`.

La SVD M100 vaut `(1, 0.41608, 0.05539, 1.4307e-4)` et la corrélation `Q/b`
est `0.9999999999`. La montée d'échelle est réussie numériquement, mais ne
constitue pas une identification expérimentale ni une preuve de séparation
de `Q` et `b`.
