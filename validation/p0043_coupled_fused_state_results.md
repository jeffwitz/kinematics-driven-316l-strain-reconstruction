# P43 — expérimentation `state + coupled block` fused

## Résultat

Un kernel Numba point-local expérimental a été ajouté sous
`coupled_block_solver="numba-fused-state"`. Il calcule dans le même kernel
l'état SRIX, le résidu, `A/B`, le Schur et la correction coupled. Les chemins
NumPy et `numba-fused` restent inchangés et constituent les références.

Le premier run a inclus la compilation JIT et a échoué lors d'une line-search.
Après correction et préchauffage, le run M20 converge avec les mêmes 119
Newton globaux, 2265 GMRES et le même RAW RMS que le chemin de référence.

## M20 préchauffé

| chemin | temps (s) | état (s) | bloc (s) | tangente (s) |
|---|---:|---:|---:|---:|
| `numba-fused` | 9.03 | 0.89 | 1.21 | 1.92 |
| `numba-fused-state` | 10.79 | 3.94 | 0.00* | 1.82 |

`*` le temps du bloc est inclus dans le kernel et compté dans `state`.

Les champs restent proches : déplacement maximal `1.94e-12 mm`, déformation
`1.07e-9`, contrainte `1.32e-4 MPa`, EVM `5.58e-10`.

## Décision

Le kernel est fonctionnel mais ne constitue pas encore une optimisation sur
M20. Il reste activable explicitement pour de futurs benchmarks de grande
taille ; il ne change aucun défaut de production et aucun M100 n'est lancé
sur la base de ce résultat seul.
