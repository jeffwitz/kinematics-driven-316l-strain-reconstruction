# P43-EXP-RAW-001 — résultat M20 brut

## Décision

Le calcul FEMU brut est terminé, mais le gate M20→M100 est **NO-GO**. Aucun
M100 expérimental n'est lancé.

Artefact principal :
`validation/reference_data/p0043_experimental_raw_femu_m20_v1/report.json`.

## Fonction objectif

Le résidu a été optimisé directement en mm, sans pondération :

```text
observation_weighting = none
noise_model_used      = false
covariance_used       = false
```

Le prior donne `RMS_u = 4.7247169e-6 mm`. Les quatre départs réduisent le
résidu vers `4.6100966e-6 mm`, soit une baisse d'environ `2.43 %` pour le
nominal et des valeurs finales pratiquement identiques entre départs.

| départ | RMS initial (mm) | RMS final (mm) | arrêt |
|---|---:|---:|---|
| nominal | 4.7247169e-6 | 4.6100966e-6 | xtol, succès |
| E1 | 4.7137514e-6 | 4.6100967e-6 | max_nfev=24 |
| E2 | 4.7353651e-6 | 4.6100967e-6 | gtol, succès |
| E3 | 4.7213924e-6 | 4.6100967e-6 | gtol, succès |

Les solutions convergent toutes vers la même vallée :

```text
tau0 ≈ 8.70 MPa
R    ≈ 12.76 MPa
Q    ≈ 3.862 MPa
b    ≈ 1.173
```

Mais `R`, `Q` et `b` sont actifs sur la borne inférieure dans les coordonnées
réduites. La convergence apparente est donc un optimum contraint, pas une
identification libre robuste. Le gate reste volontairement négatif.

## SVD

Le spectre brut au prior est :

```text
(1, 0.135725, 0.036116, 1.031e-4)
```

À la meilleure solution :

```text
(1, 0.143084, 0.016465, 1.553e-5)
```

La troisième direction s'affaiblit fortement pendant l'ajustement ; la base
SVD initiale n'est donc pas stable sur cette tentative expérimentale.

## Conclusion autorisée

```text
raw-displacement objective decreased: true
raw-displacement multistart valley: reproducible
raw-displacement unconstrained identification: not demonstrated
M100 authorized: false
experimental parameters identified: false
```

Ce résultat confirme que la suppression du scalaire de bruit ne change pas la
vallée atteinte ; elle rend simplement le coût lisible en unités physiques.
La cause du comportement aux bornes doit être diagnostiquée avant tout
scale-up : adéquation du prior/modèle à l'observation brute, borne imposée ou
rotation de la base observable. Aucune conclusion statistique n'est tirée ici.
