# E-SRIX-FEMU-SHADOW-003C — résultat L3 complet

Les deux Jacobiennes directes ont convergé sur le chemin L3 complet de 809
incréments :

| h | temps direct | GMRES |
|---:|---:|---:|
| 0.0015 | 227.44 s | 3236 |
| 0.001 | 210.25 s | 3236 |

La stabilité entre les deux valeurs est validée : erreurs de colonnes
`[0.0532 %, 0.1020 %, 0.0026 %, 0.0025 %]`, cosinus minimaux `0.99999957`
et angle maximal du sous-espace de rang 3 `0.136°`. Le mode faible reste le
contraste `Q-b`.

Le calcul L2→L3 reste cependant négatif selon le seuil pré-enregistré
PATH-002S, uniquement à cause de l’angle de rang 3 :

| h | forward | erreurs colonnes 1–3 | cosinus min. | angle rang 3 |
|---:|---:|---:|---:|---:|
| 0.0015 | `4.71e-5` | `0.984 %, 1.875 %, 0.870 %` | `0.999825` | `2.290°` |
| 0.001 | `4.71e-5` | `0.973 %, 1.799 %, 0.869 %` | `0.999839` | `2.195°` |

Les autres seuils passent, notamment les trois premiers rapports singuliers,
qui varient de moins de `1.1 %`. La stabilité du pas shadow est donc acquise,
mais la convergence de la géométrie L2→L3 ne l’est pas encore. Le gate
PATH-002S reste négatif sans modification des seuils ; l’identification et P43
restent interdits.

Artefact primaire :
`validation/reference_data/srix_femu_shadow_h_l3_v1/report.json`.
