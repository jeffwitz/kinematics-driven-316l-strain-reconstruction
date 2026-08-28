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
M20. Il reste activable explicitement pour les benchmarks de grande taille et
ne change aucun défaut de production.

## Microbenchmark de crossover

Sur des états identiques préchauffés, le ratio `fused-state / vectorisé +
fused-block` vaut `1.53`, `1.77`, `1.54`, `0.80` et `0.78` pour `N = 800`,
`2000`, `5000`, `10000` et `20000`. Le crossover apparaît donc entre 5000 et
10000 points.

## M100

Le seuil de gain étant franchi à la taille M100, un unique forward a été
exécuté : `211.69 s`, 124 Newton globaux et 3390 GMRES. Comparé au run
`numba-fused` tangent précédent (`243.72 s`, 140 Newton, 3926 GMRES), le RAW
RMS diffère de `3.0e-15` et les champs diffèrent de `8.5e-13 mm` au maximum.
La comparaison wall-time reste une mesure de runs distincts ; elle confirme
toutefois que la fusion devient pertinente sur le grand batch.
