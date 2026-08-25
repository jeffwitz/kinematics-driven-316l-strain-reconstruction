# E-SRIX-P43-F-REIDENTIFICATION-003

## Résultat

Le re-forward strict du point trouvé par SLSQP reproduit `3.5764e-6 mm`, mais son résidu d'équilibre vaut `5.80e-7`; ce point est donc mécaniquement invalide et n'est pas un optimum.

La Jacobienne F complète à neuf paramètres a été calculée. La qualification shadow F n'est pas passée : comparée à la FD centrée F (`h=1.5e-3`), l'erreur maximale de colonne est `116.9 %` et le cosinus minimal `-0.406`. La FD centrée est donc l'oracle F actuel; la cause résiduelle du shadow F doit être traitée séparément.

Avec cet oracle FD, le spectre normalisé F au prior est :

```text
(1, 0.35566, 0.33810, 0.10197, 0.07196, 0.05697, 0.03687,
 4.12e-4, 7.70e-5)
```

Le sous-espace observable de rang 7 reste très proche de l'ancien espace C (angle principal maximal `0.095°`). Les deux directions faibles restent proches de la jauge d'échelle de contrainte et de `log(Q)-log(b)`.

Un pas Gauss–Newton contraint utilisant la FD F et un polytope physique explicite est admissible et équilibré :

```text
RMS prior       4.06717e-6 mm
RMS après 1 pas 3.98367e-6 mm
réduction       2.053 %
résidu équilibre 1.39e-10
```

L'optimisation F complète n'est donc pas encore déclarée stationnaire. Le dossier ne justifie pas encore M100.

Les cartes correspondantes sont dans `validation/reference_data/p0043_f_mapping_reidentification_v1/fd_gn_one_step_maps/`.
