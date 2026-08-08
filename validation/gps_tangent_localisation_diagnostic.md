# Diagnostic — localisation causale de la pénalité 52 vs 46 Newton (GPS vs référence, M20)

Date : 2026-08-08
Code : `scripts/diagnose_gps_tangent_localisation.py`.
Données : `validation/_generated/performance/gps_tangent_localisation_m20.json`,
`gps_tangent_localisation_points.csv`, `gps_tangent_substitution_m20.json`.

## Question

Toutes les hypothèses globales sont réfutées (pureté bit à bit, Jv exact à
`1e-9`, forcing quasi neutre, spectres `BᵀCB` identiques) ; le test croisé a
montré que le compte de Newton suit le **champ de tangentes**. Où, sur quels
points matériels, le champ tangent GPS diffère-t-il assez pour coûter les 6
itérations ? Et cette différence est-elle une différence d'état ou une
différence de formulation ?

## Protocole

Checkpoint : incrément 6 du run M20 EBSD (premier incrément profond), première
itération Newton. Les deux backends sont conduits sur leur trajectoire
complète (52 vs 46 Newton), puis :

- **Test A** — directions Newton : matrices pleines `J_G = BᵀC_G B` et
  `J_R = BᵀC_R B` assemblées depuis les tangentes réellement appliquées au
  premier appel de l'incrément 6 ; les DEUX résolues par solveur direct sur le
  MÊME résidu GPS `r = R_GPS(u_k)` ; comparaison des directions et des
  réductions non linéaires du résidu GPS sous chaque correction (norme
  intérieure, la mesure du solveur).
- **Test B** — classement spatial : par point, `ΔC = C_GPS − C_REF`,
  `δε = B δu` (direction Newton réelle du run GPS), `δσ = ΔC δε`, score
  `s_i = |δσ_i|` ; classement décroissant, fractions cumulées, nombre de
  points pour 50/80/90/95 % de l'action.
- **Test C** — substitution chirurgicale : runs M20 complets avec stress GPS,
  état GPS, loi GPS, substepping GPS, mais tangente RÉFÉRENCE sur les top-k
  points du classement, k ∈ {0, 1, 5, 10, 25, 50, 100, 800}.
- **Test D** — caractérisation constitutive des top-20 points (glissements,
  stress, position).
- **Test E** — transplant d'état : sur les top-5 points, exporter l'état du
  GPS **par nom de variable** (elastic strain, glissements, glissements
  équivalents, back strain) et l'importer dans la référence, évaluer les deux
  backends au même strain imposé, comparer les tangentes ; et la paire
  symétrique (état référence importé dans le GPS).

## Résultats

### Test A — les directions Newton au checkpoint sont identiques à `8,4e-5`

| quantité | valeur |
|---|---|
| `|δu_G − δu_R| / |δu_R|` | `8,44e-5` |
| `cos θ` | `0,999999997` |
| `ρ_GPS = |R_GPS(u+δu_G)|/|R_GPS(u)|` | `0,219` |
| `ρ_REF = |R_GPS(u+δu_R)|/|R_GPS(u)|` | `0,219` |

Les deux matrices donnent la même direction Newton (cos θ = 1 à `3e-9` près)
et la même réduction du résidu GPS (`0,219` — qui reproduit exactement la
réduction `4,86e-2 → 1,07e-2` du solveur réel). **Au checkpoint, la pénalité
n'est pas dans la direction du premier pas** : le mécanisme est une
amplification le long de la trajectoire, pas une mauvaise direction locale.

### Test B — l'action est très concentrée : 6 points font 50 %

| fraction de l'action | points nécessaires (sur 800) |
|---|---|
| 50 % | **6** |
| 80 % | 54 |
| 90 % | 142 |
| 95 % | 253 |

Le point n° 1 (point 96, pixel (8, 2), subcell 0) porte à lui seul **32 %**
de l'action totale. Classement complet : `gps_tangent_localisation_points.csv`.

### Test C — UN point suffit : top-1 fait tomber 52 → 47

| tangentes remplacées | Newton M20 |
|---|---|
| 0 (GPS pur) | 52 |
| **1** | **47** |
| 5, 10, 25, 50, 100, 800 | 47 |

Substituer la tangente de la référence sur le **seul point 96** récupère la
totalité de la pénalité (52 → 47, la valeur du run croisé complet et le
critère de succès du CdC §11 : « 52 → 46–48 »). **La pénalité est localisée
sur un point matériel unique** (et sa sous-cellule), pas distribuée.

### Test D — le point 96

Position : pixel (8, 2), subcell 0. Glissements GPS (par système) :
`[7e-5, 4,3e-4, 0, 6,0e-4, 0, −1,2e-3, 0, 0, 0, 3,6e-4, −1,10e-2, −2,9e-4]` —
système 11 dominant (`−1,1e-2`), 6 systèmes actifs. Stress au checkpoint :
GPS `[−152,6, 119,8, −40,6]` MPa, référence `[−152,6, 119,9, −40,5]` MPa —
les états sont proches, les tangentes ne le sont pas.

### Test E — à même état (ISV par nom), les tangentes diffèrent de `1e-4` à `4e-3`

| point | score | `|C_GPS − C_REF|/|C_REF|` sur S_G | sur S_R |
|---|---|---|---|
| 96 | 4,70e-2 | `1,86e-3` | `1,86e-3` |
| 95 | 1,12e-2 | `3,59e-3` | `3,60e-3` |
| 59 | 7,02e-3 | `1,89e-4` | `1,89e-4` |
| 20 | 4,31e-3 | `1,97e-4` | `1,97e-4` |
| 21 | 3,04e-3 | `8,75e-5` | `8,75e-5` |

Les deux directions de transplant (état GPS dans la référence, état référence
dans le GPS) donnent la même différence, au signe de l'état près — le résultat
est symétrique et reproductible. **Le seuil `1e-10` du CdC §9 n'est pas
atteint de sept ordres de grandeur.**

Limite du test E, documentée : le transplant est partiel. Il copie les
variables internes **par nom** (elastic strain, glissements, glissements
équivalents, back strain) mais pas le gradient committé : celui de la
référence est stocké dans le repère cristal (rotation appliquée avant
l'écriture), celui du GPS en repère global, et imposer le gradient global du
GPS à la référence fait échouer son Newton local (vérifié). Les déformations
transverses committées restent donc celles de chaque backend. La différence
mesurée à même ISV peut venir (a) des transverses committées, (b) d'une
différence algébrique réelle de la tangente entre les deux formulations.

## Conclusion causale

La chaîne du CdC §11 est établie avec localisation :

```
ΔC_i (point 96, à même ISV : 1,9e-3)
→ ΔJ (matrice assemblée, action concentrée à 32 % sur ce point)
→ Δδu par itération (8,4e-5 au checkpoint — petite mais systématique)
→ amplification le long de la trajectoire (ρ identiques localement,
   trajectoires divergentes globalement)
→ 52 Newton au lieu de 46
```

- **La pénalité est portée par un seul point matériel** (point 96, pixel
  (8,2), subcell 0, système de glissement 11 dominant) : substituer sa
  tangente suffit à faire 52 → 47.
- **La différence de tangente n'est pas qu'un écart d'état** : à variables
  internes identiques (export/import par nom) et même strain imposé, les
  tangentes diffèrent encore de `1,9e-3` sur ce point — le seuil de
  formulation identique (`1e-10`) est manqué de 7 ordres de grandeur.
- Le test A nuance : au checkpoint, les deux matrices donnent la même
  direction Newton (cos θ = 1 à `3e-9`). La différence de tangente est donc
  **locale au point 96** mais s'amplifie le long des 46–52 itérations, ce qui
  réconcilie toutes les mesures globales (Jv exact, spectres identiques) avec
  le compte d'itérations.

### Test F — bloc par bloc au même état complet : la formulation diffère de `3e-3`

Étape §12 du CdC, `scripts/diagnose_gps_tangent_blocks.py`. Pour les points
96, 95, 59 au checkpoint inc 6, avec un transplant COMPLET (ISV par nom +
gradient tourné dans le repère du receveur + transverse convergé imposé) :

- la tangente 3D de la référence est reconstruite à sa fermeture convergée
  (évaluation du bridge au transverse `_latest_transverse`) — validé :
  son Schur redonne la tangente retournée à `0.000e+00` ;
- la tangente GPS est la tangente DSL globale (post-multiplication par le
  projecteur in-plane, rotations de sortie répliquées), réévaluée au même
  état.

| point | Schur(réf) vs projetée(GPS) même état | retournées même état |
|---|---|---|
| 96 | `3,13e-3` | `2,69e-3` |
| 95 | `3,75e-3` | `3,45e-3` |
| 59 | `1,42e-4` | `1,37e-4` |

**À même état complet, les deux formulations ne produisent pas la même
tangente** : `3e-3` sur les points responsables, contre `1,4e-4` sur le point
au score faible. L'écart croît avec le score de responsabilité — c'est le
chaînon manquant. La comparaison 6×6 brute n'est pas pertinente (la tangente
DSL GPS a les lignes transverses nulles par construction — Cbb = 0, déjà
documenté dans `mfront.py`) ; c'est la projection in-plane qui décide du
Newton, et elle diffère.

**Conclusion causale finale** : la tangente GPS projetée (DSL + projecteur
in-plane) diffère du Schur de la condensation référence de `~3e-3` à l'état
réel du point 96 — une différence de FORMULATION (dérivée), pas d'état. Le
solveur GPS applique cette matrice le long de la trajectoire ; la direction
diffère de `8,4e-5` par itération (test A) et s'amplifie en écart de compte
(52 vs 46). La localisation, la causalité et l'échelle sont établies.

**Piste pour la correction** (non modifiée — démonstration d'abord) : la
post-multiplication par le projecteur in-plane `P` est l'endroit où la
tangente DSL GPS devrait coïncider avec le Schur — elle n'y arrive qu'à
`3e-3`. La route 2 du handoff (tangente condensée via le Schur de la loi
brute, `shadow_condensed_tangent`) produit exactement le Schur ; elle est
désactivée par défaut car « le shadow ne suit pas la branche » — à
réexaminer à la lumière de cette mesure.

### Test G — sensibilité directe du système GPS : le DSL est exact, le SYSTÈME diffère

Étape 1-2 du CdC de l'utilisateur, `scripts/diagnose_gps_direct_sensitivity.py`.
Réimplémentation numpy complète des 18 équations de
`Fcc316LForestRubinSrixGps.mfront` (résidu ET Jacobien, avec les blocs
déclarés `dfeel_ddeel`, `dfeel_ddg`, `dfg_ddeel`, `dfg_ddg`), validée sur le
point 96 au checkpoint inc 6 :

- **`|F(x*)| = 1e-15`** à l'état convergé — la réimplémentation du résidu est
  exacte (conventions : rotation `Qᵀ` dans `gpsRotate`, `deto` = incrément
  `s1.gradients − s0.gradients`, tenseurs de Schmid recopiés du `.ixx`
  généré) ;
- **`C_sens vs C_DSL = 1e-15`** — la sensibilité directe `C_sens =
  (∂σ_a/∂x)·(−A⁻¹B)` avec `B = ∂F/∂ε_a = [−I₃; 0; 0]` reproduit EXACTEMENT
  la tangente que le bridge retourne. **Le DSL ne fait pas d'erreur de
  calcul** : il assemble précisément la dérivée du système qu'il résout ;
- **`C_sens vs C_shadow = 3,1e-3`** (point 96), `3,7e-3` (95), `1,4e-4` (59)
  — la sensibilité du système GPS diffère du Schur de la référence de la
  même quantité que la tangente DSL. **Le jalon du CdC
  (`|C_sens − C_shadow|/|C_shadow| ≤ 1e-10`) est réfuté par la mesure.**

**Interprétation — c'est le SYSTÈME, pas la dérivée.** Le DSL calcule
exactement la dérivée du système GPS tel qu'il est formulé ; ce système
diffère du problème condensé de la référence par la STRUCTURE de
l'élimination transverse : la fermeture GPS (`σ_b = 0` dans les lignes
2,4,5 de `feel`) ne voit que `deel` (∂σ_b/∂dg = 0), alors que le Schur de la
référence élimine les transverses à travers la cinématique complète
(`deel_b + plastic_b − deto_b`, qui dépend de `dg`). Les deux systèmes
donnent les mêmes valeurs (stress à `1e-11`) mais des dérivées différentes
à `3e-3` — une différence de formulation, pas de calcul. Le shadow
(`2da6268`, 52 → 47) ne « corrige » donc pas une dérivée mal calculée : il
remplace la matrice du système GPS par celle d'un autre système (la
référence), dont la convergence est meilleure. La leçon : améliorer la
convergence du GPS exige de rapprocher la FORMULATION du système de la
condensation (la dépendance transverse complète), pas de recalculer la
dérivée du système actuel — celle-ci est déjà exacte.

La vérification FD du Jacobien est cohérente avec cette lecture : l'écart
`|A_an − A_FD|/|A_FD| ≈ 2,5e-3` ne diminue pas quand h décroît (`1e-7` →
`1e-9`) et se concentre sur les lignes `fg` des systèmes inactifs — la
non-différentiabilité du crochet de Macaulay (la perturbation traverse le
coin `overstress = 0`), pas une erreur de transcription : la validation
décisive est l'égalité `C_sens = C_DSL` à `1e-15`, où `C_DSL` est la
tangente réellement retournée par MFront.
