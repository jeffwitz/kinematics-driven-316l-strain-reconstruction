# Premières mesures de performance

Ces mesures dimensionnent le solveur Python/PyPardiso ; elles ne constituent
pas encore la qualification du cas de production hétérogène.

## Protocole

- machine : 8 processeurs logiques, 30 GiB de RAM ;
- backend : `pypardiso (MKL, multithreaded)` ;
- environnement : versions exactes de `requirements-lock.txt` ;
- cas : traction équibiaxiale homogène, loi tabulée, 20 incréments ;
- commande : `/usr/bin/time -v fem-inhouse validate --nx N --ny N` ;
- mesure mémoire : maximum resident set size du processus complet.

| Maillage | Éléments | Temps mur | CPU cumulé | CPU moyen | Pic RSS |
|---:|---:|---:|---:|---:|---:|
| 100×100 | 10 000 | 5,01 s | 27,69 s | 552 % | 163 MiB |
| 224×224 | 50 176 | 10,60 s | 55,34 s | 521 % | 557 MiB |
| 316×316 | 99 856 | 21,87 s | 76,37 s | 349 % | 1,04 GiB |

Les taux CPU supérieurs à 100 % confirment l'utilisation de plusieurs cœurs.
Le nombre de threads n'était pas forcé par `MKL_NUM_THREADS` ou
`OMP_NUM_THREADS`; il reste à fixer explicitement pour obtenir des comparaisons
reproductibles.

Les trois calculs satisfont les seuils analytiques. À 100k éléments :

- erreur relative SVM : `5,84e-6` ;
- erreur relative PEEQ : `2,17e-6` ;
- déséquilibre relatif des réactions : `1,25e-12`.

## Limite de la campagne actuelle

Le point à environ 350k éléments n'a pas été lancé : au moment de la mesure, la
machine ne disposait que de 3,7 GiB de mémoire disponible et ses 8 GiB de swap
étaient déjà saturés. Les mesures précédentes suggèrent un pic de plusieurs GiB,
mais une extrapolation d'un solveur direct sparse n'est pas une preuve de
compatibilité.

Le benchmark 350k doit être repris lorsque :

- au moins 8 GiB sont disponibles sans swap ;
- le nombre de threads MKL est fixé et enregistré ;
- aucun autre calcul lourd n'est actif ;
- le cas homogène et un cas hétérogène représentatif sont tous deux mesurés.

## Interprétation

Le temps observé couvre la génération du cas, l'assemblage, toutes les
résolutions PyPardiso, Newton et le post-traitement final.

Une instrumentation interne a ensuite été ajoutée à `SolverDiagnostics`. Sur
un cas hétérogène déterministe de 100×100 éléments (gradients et perturbations
sinusoïdales des deux cartes matériau), elle donne :

| Poste | Temps cumulé |
|---|---:|
| Initialisation du maillage et des opérateurs | 0,019 s |
| Assemblage élastique initial | 0,054 s |
| Intégration constitutive | 8,744 s |
| Tangentes et assemblages non linéaires | 10,095 s |
| Résolutions linéaires PyPardiso | 10,592 s |
| Construction des sorties | 0,021 s |
| Total mur | 31,948 s |

Ce calcul a convergé sans cutback en 20 incréments, 78 itérations de Newton au
total et au plus 5 itérations par incrément. Les temps sont cumulatifs par
phase, mesurés avec `perf_counter`; leur somme n'est pas exactement le temps
mur, car les calculs de résidu et les allocations intermédiaires ne forment pas
encore une catégorie dédiée. PyPardiso regroupe actuellement analyse,
factorisation et substitutions dans l'appel chronométré `linear_solve_seconds`.

Le statut de chaque partition enregistre aussi `write_seconds`, séparément du
temps du solveur.

Avant le ROI complet, il reste donc à :

- séparer, si l'API PyPardiso le permet sans modifier le résultat, analyse,
  factorisation et substitutions ;
- mesurer 350k éléments ;
- définir un budget par job à partir du cas hétérogène ;
- tester l'effet d'un nombre de threads fixé ;
- contrôler les copies lors de l'assemblage des tangentes.

SciPy/SuperLU n'est pas une option de production : conformément au cahier des
charges, son repli reste limité au diagnostic de petits systèmes lorsque l'API
publique est appelée avec `require_pypardiso=False`.
