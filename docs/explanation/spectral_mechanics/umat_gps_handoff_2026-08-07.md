# Synthèse de reprise — UMAT GPS (fermeture de contrainte plane dans MFront)

Date : 2026-08-07. Branche : `codex/native-generalised-plane-stress`.
Auteur du travail : session Claude (avec l'utilisateur). Ce document est écrit
pour qu'un nouveau modèle reprenne le travail sans contexte.

## 1. Le problème

Le solveur spectral 2D (contrainte plane) a besoin, pour chaque point de
Gauss, de la réponse dans le plan de la loi cristalline SRIX 3D
(`Fcc316LForestRubinSrix`, MFront, DSL Implicit, 18 inconnues : `deel` + `dg[12]`)
**avec la fermeture de contrainte plane** :

```
sigma_zz = sigma_xz = sigma_yz = 0   (repère STRUCTURAL / global)
```

La contrainte plane est une condition **globale** : les surfaces libres de la
tôle sont normales à z. Les déformations transverses (ε_zz, ε_xz, ε_yz) sont
les inconnues de fermeture. La loi est intégrée dans le repère **cristal**
(rotations EBSD appliquées autour de l'intégration).

Objectif : déplacer cette fermeture **dans le comportement MFront** (côté
C++), au lieu de la boucle Python actuelle — pour (a) la vitesse, (b) forcer
la racine qui satisfait la contrainte plane (voir §4).

## 2. Ce qui marche en Python (la référence, à ne pas casser)

`MFront3DCondensedPlaneStressBatch` (`src/fem_inhouse/core/mfront.py`) :
- le pont tourne la déformation globale → repère cristal (`rotateGradients`),
  intègre la loi 3D, retransforme contrainte et tangente → global ;
- le Newton de fermeture (Python) itère les déformations transverses
  **globales** en résolvant `Cbb Δε_b = −σ_b` (Schur), avec le prédicteur
  transverse (committed/tangent), les contrôles de conditionnement de `Cbb` ;
- c'est la **référence de production** (qualifiée ~`1e-11` sur P43 M100 EBSD,
  ~56,88 s) et les campagnes archivées reposent dessus.

**Conventions de stockage (cruciales)** : les tableaux MGIS (gradients,
forces thermodynamiques, variables internes) stockent le cisaillement en
**Kelvin** : `shear = gamma/sqrt(2)` (pas l'ingénieur `gamma`). L'ordonnancement
des 6 composantes est standard `[11, 22, 33, 12, 13, 23]` (confirmé par
`kelvin_3d_to_tensor` dans `src/fem_inhouse/core/tensor_reconstruction.py`).
Les rotations MGIS (`rotateGradients`, `rotateThermodynamicForces`) opèrent
sur ce stockage Kelvin. Toute formule de rotation écrite à la main doit
utiliser le stockage Kelvin (voir §3, bug 1).

**Découverte scientifique (2026-08-07)** : la loi SRIX admet **plusieurs
racines** au premier incrément plastique (crochet de Macaulay, comportement
déjà consigné au journal du 2026-08-03 : « une autre solution »). Le Newton
imbriqué Python et le Newton conjoint sélectionnent des branches différentes
(mêmes systèmes actifs `[1,2,4,5,7,8,10,11]`, amplitudes de glissement
~2-3× différentes, `eps_zz = -0,00165` vs `-0,00262`). La racine « naturelle »
à un ε_zz donné peut violer la fermeture (`sigma_zz = -154,7 MPa` → **pas une
solution de contrainte plane**). Voir
`validation/srix_plane_stress_branch_diagnostic.md`.

## 3. Ce qui a été déplacé vers MFront/C++ (la voie « UMAT »), et comment

**Principe** : passer la rotation `Q_global_to_material` à la loi comme **9
propriétés matériau par point** (`Q11`..`Q33`, dimensionnées `real`), la loi
applique elle-même la rotation du gradient imposé, et porte la fermeture dans
son propre Newton implicite. Le pont devient passif (pas de rotation
d'entrée, pas de fermeture Python).

**La loi** : `mfront/Fcc316LForestRubinSrixGps.mfront` (corps SRIX identique) :
- 3 variables d'état supplémentaires : `ezz`, `eyz`, `exz` (déformations
  transverses **globales**) — le système local passe à **21 inconnues** ;
- dans l'`@Integrator` : le gradient imposé `deto` est recopié, ses
  composantes transverses (2, 4, 5) remplacées par les inconnues de
  fermeture, puis tourné par `Q` (`gpsRotate`, formule stockage Kelvin) ;
  `feel = deel - deto_m + sum(dg m)` (on écrase le `feel = deel - deto` du
  brick `StandardElasticity`) ; blocs de Jacobien `dfeel_ddezz` etc. ;
- 3 résidus de fermeture : `fezz = (Qᵀ sigma Q)_zz / G_ref` (idem xz, yz),
  normalisés par un module de référence (`122000`) ; les dérivées
  `dfezz_ddeel` = rangées de la raideur tournée ;
- paramètre de diagnostic `GpsClosureFrame` (1 = global, 0 = matériau) —
  diagnostic seulement ;
- `@Algorithm NewtonRaphson`, `@Epsilon 1.e-12` (relâché depuis 1e-14),
  `@IterMax 200` (testé, sans effet) ;
- **le solveur `PowellDogLeg_NewtonRaphson` (région de confiance) échoue
  partout** (incompatible avec le crochet de Macaulay non lisse) — ne pas
  l'utiliser.

**Limite du DSL Implicit 5.1 découverte en route** : pas de blocs de
dérivées utilisateur par rapport au gradient (`dfeel_ddeto` non déclaré). La
tangente automatique différencie le système comme si `dF/ddeto = -I` sur le
bloc élastique. La vraie tangente est donc celle retournée par le DSL
post-multipliée par l'opérateur de rotation dans le plan
`d(deto_m)/ddeto` restreint aux colonnes dans le plan — correction appliquée
**côté pont** (voir §4, mécanique de tangente).

**Le pont** : `MFrontNativeGeneralisedPlaneStressBatch`
(`src/fem_inhouse/core/mfront.py`, réécrit sur le pattern `MaterialDataManager`) :
- pose `Q11`..`Q33` par point (`setMaterialProperty`, stockage
  `ExternalStorage`), ne tourne PAS le gradient d'entrée ;
- intègre (`integrate(manager, IntegrationWithConsistentTangentOperator, ...)`) ;
- rotations de sortie : contrainte et déformation élastique → global
  (`rotateThermodynamicForces`) ; tangente : post-multiplication par
  l'opérateur dans le plan + rotation **unilatérale** (colonne par colonne via
  `rotateThermodynamicForces`) — la tangente retournée par la loi est mixte
  (σ matériau, ε global) ;
- lit les variables de fermeture (offsets `ezz/eyz/exz`) pour reconstruire la
  déformation totale ; `plane_stress_residual` = contrainte transverse
  globale (critère A3) ; discard du bloc hors plan ;
- la **sortie `rotateGradients` sur le bloc identité est transposée** par
  rapport à l'opérateur dérivé (les vecteurs unités rotés sont en LIGNES, les
  colonnes de l'opérateur les veulent en COLONNES) → `.transpose(0, 2, 1)`
  (voir §4, bug 2).

**Câblage** : `create_plane_stress_material_batch("mfront-native-generalised-plane-stress", ...)`
route automatiquement vers le spec `fcc_forest_rubin_srix_gps`
(`src/fem_inhouse/core/mfront_behaviours.py`) ; la loi est dans la liste
explicite de `scripts/build_mfront_behaviour.sh`.

**La sonde C++** : `/tmp/gps_probe.cxx` (compilée avec `-std=c++20 -O2` +
l'objet généré `build/mfront/src/Fcc316LForestRubinSrixGps.o` + les libs TFEL)
pilote la loi générée directement via `mfront::gb::integrate` et imprime
l'état convergé — utilisée pour vérifier que l'état UMAT est une vraie racine.

## 4. Ce qui marche (vérifié)

- **La fermeture converge** : `sigma_transverse = 0` à `1e-14 MPa` à
  l'identité ET aux orientations tournées (`[35,20,15]`, `[54.7,45,10]`),
  à travers la plage plastique modérée (7 des 12 incréments de l'historique
  gelé) ;
- **La tangente est correcte** : différences finies à `1.3e-7` relatif au
  point tourné plastique (et `2.3e-8` à l'identité) ;
- **Performance** : `~6,7× plus rapide` que la référence Python par
  évaluation matériau (`0,42 ms` vs `2,81 ms`, un point, 4 threads) — un
  Newton au lieu d'un Newton imbriqué ;
- l'incrément élastique s'accorde à la référence à `7e-15` (contrainte).

## 5. Ce qui ne marche pas (les deux blocages)

**5.1 Accord avec la référence (F1) — tranché par décision utilisateur.**
Dès le premier incrément plastique, la solution UMAT diffère de la référence
(les deux sont des racines du même système discret — loi multivaluée). La
décision de l'utilisateur : **seule la racine qui satisfait la contrainte
plane est valide — on la force** ; la racine naturelle qui viole la fermeture
n'est pas une solution de contrainte plane. La préinscription est amendée :
A1' = la solution UMAT est une racine du système fermé (Newton convergé +
fermeture ≤ `1e-6 MPa` + tangente FD) ; l'écart à la référence est rapporté
comme écart de branche. À terme, les campagnes condensées archivées (branche
naturelle) devront être refaites — pour l'instant : documenter + implémenter.
`validation/srix_umat_gps_closure_preregistration.md` (amendement 1),
`validation/srix_umat_gps_closure_results.md`.

**5.2 Mur de robustesse (ouvert, non résolu).** Le Newton conjoint 21
inconnues **diverge aux états plastiques profonds** : incrément 8 de
l'historique gelé à 12 incréments ; le P43 20×20 à 8 incréments échoue aux
points profonds. Faits établis :
- indépendant du **départ** (échoue même depuis l'état committé de la
  référence, branche naturelle injectée) ;
- indépendant du **Jacobien** (analytique ou solveur
  `NewtonRaphson_NumericalJacobian`) ;
- indépendant d'**`@IterMax`** (200) et de la **normalisation** de la
  fermeture (module `1e6` testé) ;
- la **référence imbriquée converge** au même état (le Newton 3D seul
  converge ; c'est le couplage conjoint qui casse) ;
- les deux fermetures (globale et matériau, flag `GpsClosureFrame=0`)
  échouent au même endroit.
Conclusion : limite de la **structure du Newton conjoint dans le DSL
Implicit** (pas de line search, pas de contrôle du pas), pas de la
formulation de la fermeture. Voies possibles : accepter le mur (backend
qualifié sur la plage modérée) ou investiguer le bassin du Newton à l'inc 8
(la sonde C++ peut imprimer la trajectoire du résidu).

## 6. Fichiers et état

- Loi : `mfront/Fcc316LForestRubinSrixGps.mfront` (expérimentale).
- Pont : `MFrontNativeGeneralisedPlaneStressBatch` dans
  `src/fem_inhouse/core/mfront.py`.
- Factory/spec : `src/fem_inhouse/core/plane_stress_material.py`,
  `src/fem_inhouse/core/mfront_behaviours.py`.
- Préinscription amendée : `validation/srix_umat_gps_closure_preregistration.md`.
- Résultats (négatif + fixes + mur) : `validation/srix_umat_gps_closure_results.md`.
- Diagnostic des branches : `validation/srix_plane_stress_branch_diagnostic.md`.
- Scripts : `scripts/qualify_srix_umat_gps_closure.py`,
  `scripts/diagnose_srix_plane_stress_branches.py`,
  `scripts/benchmark_srix_umat_gps_p43.py`.
- Commits de la branche : `1c7b091` (backend), `0c793b5` (diagnostic),
  `6e9a423` (fixes de rotation), `81562cf` (Claude.md).
- **Non commités** (consigne utilisateur — ne pas les inclure) : la campagne
  de condensation en blocs (`scripts/benchmark_srix_ebsd_condensation_blocks.py`,
  `validation/_generated/performance/srix_p43_m100_ebsd_condensation_blocks*`,
  `docs/explanation/spectral_mechanics/srix_p43_performance_and_step_control.md`
  modifié, `tests/unit/core/test_mfront.py` modifié).
- Échec de test **préexistant** (pas une régression) :
  `test_3d_condensed_backend_matches_native_plane_stress_fem` (KeyError
  `plastic_strain_2d`).

## 7. Environnement

TFEL/MFront 5.1.0, MGIS 3.1 (`/home/jeff/.local`). Shell :
`sourcer /home/jeff/.local/share/tfel/env/env.sh` + `PYTHONPATH` vers
`/home/jeff/.local/lib/python3.12/site-packages` + `MFRONT_BEHAVIOUR_LIBRARY`
vers `build/mfront/src/libBehaviour.so`. Recompiler : `bash
scripts/build_mfront_behaviour.sh`. Le fork TFEL
`jeffwitz/tfel-generalised-plane-stress` (prototype générateur) est **parker**
— la voie UMAT ne le nécessite pas.

## 8. Reprise du 2026-08-07 (suite) — le mur, mesuré puis déplacé

### 8.1 Il n'y a pas de repli de branche (hypothèse testée et réfutée)

`scripts/diagnose_srix_closure_root_sweep.py` balaie `sigma_zz(eps_zz)` avec la
loi 3D brute à chaque état committé de l'historique gelé, à incrément dans le
plan figé. Résultat : **exactement une racine à chacun des douze incréments**,
l'incrément 8 compris, et les racines avancent régulièrement de `-1,0e-3` par
incrément — soit exactement `-(eps_xx + eps_yy)`, l'incompressibilité plastique.

L'hypothèse d'un point limite (deux racines qui fusionnent) est donc **réfutée** :
la racine de contrainte plane existe, elle est unique et bien séparée là où le
Newton conjoint meurt. Le §5.2 n'est pas une limite du problème, c'est une
limite de l'itération. Données :
`validation/_generated/performance/srix_closure_root_sweep.json`.

### 8.2 Le sous-pas franchit le mur

`MFrontNativeGeneralisedPlaneStressBatch` réduit l'incrément par moitiés
successives (jusqu'à 1/256) quand le Newton conjoint échoue, en avançant `s0`
en interne puis en le remettant en place — `commit()` et `revert()` gardent leur
sens pour l'appelant. Mesure : **sans sous-pas, la qualification ne dépasse pas
l'incrément 3** ; avec, les trois cas parcourent les douze incréments, fermeture
à `4e-14 MPa`. Le mur du §5.2 est franchi.

Coût : 9 à 10 incréments sur 12 nécessitent le sous-pas, jusqu'à 1/256. Le gain
de 6,7× n'est donc plus acquis et doit être remesuré.

Un `@Predictor` transverse (élastique puis isochore) a été ajouté à la loi. Il
est correct et il ne change rien au comptage de sous-pas : **l'échec n'est pas
un problème de point de départ**, ce qui confirme le §5.2.

### 8.3 Le vrai blocage : A6 était vacue, et la tangente est fausse

`_finite_difference_tangent_check` était appelé **après** `_run_history`, donc
sur un incrément de déformation **exactement nul** : la loi prenait sa branche
élastique gardée et la différence finie s'accordait à `1e-9` avec la tangente
élastique. **A6 ne testait pas la tangente plastique** et rapportait un succès.

Corrigé : le contrôle tourne maintenant à **chaque** incrément, depuis l'état
committé qui le précède, et le critère est le pire des douze. Ce que cela
révèle :

| cas | A6 (ancien, vacu) | A6 (réel) |
|---|---|---|
| C1 identité | — | `7,4e-01` |
| C2 Bunge 35/20/15 | `2,0e-09` | `5,2e+00` |
| C3 Bunge 54,7/45/10 | `1,2e-09` | `3,2e-07` |

Par incrément, sans sous-pas, là où le Newton conjoint converge (incréments 1 à
3) la tangente est excellente : `7e-08`, `1e-07`, `1,8e-06`. **La formulation de
la tangente est juste ; c'est le sous-pas qui la détruit**, parce que la matrice
retournée est celle du dernier sous-pas et non celle de l'incrément complet.

Le mur de convergence a donc été échangé contre un mur de tangente, et le second
était masqué par un test vacu. La qualification **rejette** maintenant C1 et C2.

### 8.4 Voie identifiée pour la tangente

Le sous-pas donne le `Δezz` exact. Il reste à obtenir la tangente cohérente de
l'incrément entier. Deux routes, non implémentées :

1. **Sous-pas pour localiser, puis un pas complet exact** : réinjecter le
   `Δezz` trouvé comme point de départ du Newton conjoint sur l'incrément
   entier (via une variable externe lue par le `@Predictor`). Un Newton parti
   sur la racine converge, et la tangente est exacte. Préserve la vitesse.
2. **Condenser** : à `Δezz` connu, intégrer la loi 3D **brute** sur l'incrément
   entier (elle converge, c'est ce que fait la référence) et condenser
   `C^ps = Caa − Cab Cbb⁻¹ Cba`. Exact, mais une intégration 3D de plus par
   évaluation.

La route 1 est la bonne si le prédicteur peut recevoir une valeur imposée.

### 8.5 Route 1 testée — le Newton complet ne peut pas converger aux états profonds

La route 1 (sous-pas pour localiser `Δeps_zz`, puis Newton complet sur
l'incrément entier depuis la racine localisée, injectée au `@Predictor` via
une variable externe) a été implémentée : 3 variables externes
`GpsPredictorEzz/Eyz/Exz` dans la loi, posées par le pont après le sous-pas,
le prédicteur les lit. Résultat mesuré : le re-run **réussit** (status 1)
mais converge vers **le même point que le sous-pas** (stress et tangente
identiques, diff 0) — le sous-pas accumulé est déjà une racine du problème
complet (les équations sont additives pour ce chargement monotone). La
tangente du DSL à l'état profond est donc intrinsèquement fausse : à
l'identité (où la correction `ROT_inplane` est un no-op), elle diffère de la
différence finie d'un facteur `~1,5-2,2` (mesuré à l'incrément 8, vraie
incrément : `1,48 / 1,74 / 1,72 / 1,74 / 2,21` par entrée). Le problème n'est
pas le chemin du Newton, c'est la **tangente cohérente du DSL Implicit à
l'état plastique profond** (le Jacobien 21×21 ou la mécanique `D_tdt·Je`).

La route 2 devient la voie : avec la racine localisée (le sous-pas donne le
`Δeps_zz` exact), intégrer la **loi 3D brute** sur l'incrément entier depuis
le même état committé (elle converge — c'est la mécanique de la référence)
et condenser sa tangente : `C^ps = Caa − Cab Cbb⁻¹ Cba`. La tangente de la
loi 3D brute est celle qualifiée (celle de la référence) ; le Schur est la
tangente de contrainte plane exacte. La convention des variables de fermeture
(ingénieur vs Kelvin) doit être tranchée empiriquement avant (comparer les
ISV de fermeture aux composantes du tenseur de déformation de la référence au
point tourné). Le pont doit maintenir un scratch de la loi 3D brute (état
committé synchronisé depuis celui du GPS, mapping ISV 45→42).

### 8.6 Reprise : la route 1 n'a jamais tiré, la route 2 est réfutée

Trois mesures, dans l'ordre où elles ont été faites.

**La tangente 3D brute est exacte.** Différences finies centrées sur la loi
`Fcc316LForestRubinSrix` seule, aux incréments 2, 4, 8 et 12 d'un trajet
plastique : erreur relative `4,2e-11`, `2,7e-11`, `3,3e-11`, `1,7e-10`, rapport
numérique/analytique `[1,000 ; 1,000]` partout. **La machinerie de tangente du
DSL Implicit n'est donc pas en cause**, contrairement à ce que conclut le §8.5.
Et la tangente GPS est elle aussi exacte partout où le sous-pas ne se déclenche
pas : `7e-08` à l'incrément 1, `1e-07` au 2, `1,8e-06` au 3. La chaîne
—jacobien, `J^-1`, post-multiplication par `ROT·P`, rotation de sortie— est
juste. Seule la matrice issue du sous-pas est fausse.

**La route 1 ne s'exerce pas.** En instrumentant l'appel :
`_rerun_full_increment_from_located_root` est bien appelée (6 fois sur 8
incréments), renvoie statut 1, et laisse `manager.K` **bit à bit identique**
(`|ΔK| = 0,0`). Le « le rerun converge au même point » du §8.5 n'est donc pas
une propriété de la physique : le mécanisme n'a produit aucun effet, et la
conclusion « c'est la tangente du DSL le blocage » ne repose sur rien. Le
pourquoi du no-op reste ouvert (variable externe `ExternalStorage` non
maintenue en vie côté Python ? sortie anticipée du Newton au point de départ ?).

**La route 2 est réfutée.** Elle a été implémentée : un jumeau
`MFront3DMaterialPointBatch` piloté en phase, à qui l'on impose exactement la
déformation transverse localisée par la fermeture, et dont la tangente 3D est
condensée par le Schur `C^ps = Caa − Cab Cbb⁻¹ Cba`. Mesure de la contrainte
du jumeau contre celle du GPS, même état committé, même déformation totale :

| incrément | écart relatif | `sigma_zz` du jumeau |
|---|---|---|
| 1 | `1,6e-11` | `0,000` |
| 2 | `1,06` | `-152,1` |
| 4 | `5,37` | `-915,7` |
| 8 | `22,0` | `-4221,4` |

**Imposer la déformation ne sélectionne pas la branche.** Le problème à 18
inconnues est lui-même multivalué, et le Newton brut retombe sur la racine
« naturelle » — les `-152 MPa` sont les `-154,7 MPa` du diagnostic de branches.
La prémisse « la loi 3D brute converge, c'est la mécanique de la référence »
est fausse pour la sélection de branche. A6 avec le jumeau : `9,84`, `3,81`,
`0,78` — pire que sans. Le code est conservé, `shadow_tangent=False` par
défaut, comme trace du résultat négatif.

**Ce qui reste.** Toute route future doit transporter la **branche**, pas
seulement la déformation : par exemple partir le Newton 3D depuis l'état
interne convergé du GPS (glissements compris) et non depuis l'état committé, ou
obtenir la tangente sans réintégration, en assemblant `J^-1` du système
augmenté au point convergé du sous-pas mais avec les incréments de l'incrément
entier. Rien de cela n'est mesuré.

Note annexe : `@CompareToNumericalJacobian` est **inutilisable** sur cette loi.
TFEL 5.1 génère le bloc de comparaison avec de mauvais décalages de colonnes —
l'intégrateur mappe `dfeel_ddezz` en colonne `StensorSize+12`, le comparateur
en colonne `StensorSize` — et signale donc des blocs faux qui ne le sont pas.

### 8.7 Pourquoi cela n'arrive pas en Python — et ce que ça dit de la « branche »

La question est la bonne, et sa réponse retourne le diagnostic.

**En Python la déformation transverse est IMPOSÉE.** La loi reçoit un gradient
complet et ses cinématiques sont cohérentes par construction : il n'y a aucun
degré de liberté par lequel l'état pourrait dériver. Dans l'UMAT la déformation
transverse est une inconnue **du même système**, et c'est là que le mode de
défaillance devient possible.

**L'identité qui tranche.** `feel = deel − deto_m + Σ dg m`, les tenseurs de
Schmid sont déviatoriques (`tr(m) = 0`) et une rotation conserve la trace. Donc
tout état convergé doit vérifier

```
tr(eel) = tr(eps_total)
```

Ce n'est pas une affirmation physique, c'est la trace du résidu : un état qui la
viole n'est pas une solution du système que la loi prétend avoir résolu. Aucune
référence n'est nécessaire pour l'évaluer.

Mesure (`scripts/diagnose_gps_kinematic_consistency.py`) :

| orientation | inc 1 | inc 2 | inc 3 | inc 4 |
|---|---|---|---|---|
| identité | `4,2e-16` | **`1,52`** | `1,15` | `1,07` |
| Bunge 35/20/15 | `4,6e-15` | **`1,04`** | `1,00` | `0,99` |

Exact à l'incrément 1, violé de 100 % dès le 2, sur les deux orientations.
**Indépendant du sous-pas** (identique avec `_maximum_substeps = 1`) et
**indépendant du prédicteur** (identique après suppression du bloc
`@Predictor` — qui s'avère au passage totalement inerte, et a donc été retiré).
La violation est intrinsèque à la loi GPS telle qu'écrite, depuis son premier
commit.

**Et le chiffre qui conclut.** À l'incrément 2, identité, la valeur de `eps_zz`
qui satisfait l'identité vaut `tr(eel) − trace imposée = 3,40e-4 − 2,0e-3
= -1,66e-3`. La référence donne `-1,65e-3`. L'UMAT donne `-2,66e-3`.

**Il n'y a donc pas deux racines : il y a la solution, et un état incohérent.**
La « multiplicité » du §5.1 et du diagnostic de branches, le refus de la loi 3D
brute de reproduire l'état UMAT (§8.6), et la tangente fausse sont tous des
conséquences du même défaut. La décision utilisateur « on force la racine de
contrainte plane » n'a plus d'objet : la racine de contrainte plane est celle
que la référence calcule déjà.

**Ce qui reste à trouver** est le point précis où l'écart entre `dezz` utilisé
et `dezz` stocké apparaît. Il vaut, à l'incrément 2, exactement l'incrément de
trace imposé dans le plan (`+1,0e-3`) : la loi se comporte comme si elle avait
intégré avec `dezz ≈ -0,92e-3` (cohérent) tout en enregistrant `-1,92e-3`. Les
offsets de variables internes du pont sont vérifiés corrects (`eel` 0-5,
`PlasticSlip` 6-17, `ezz` 18, `eyz` 19, `exz` 20), et aucune autre écriture sur
`ezz` n'existe dans la loi. La piste suivante est l'ordre d'application entre
l'affectation utilisateur `feel = deel - deto_m` et ce que le brick
`StandardElasticity` écrit lui-même dans `feel`.

Tant que ce point n'est pas résolu, **la référence condensée reste la seule
solution correcte**, et le backend UMAT ne doit pas être utilisé.

### 8.8 RÉSOLU — le pont appliquait la déformation totale comme un incrément

**Tout ce qui précède avait une cause unique, et ce n'était ni la loi, ni le
DSL, ni une multiplicité de racines.**

`evaluate(in_plane_strain, ...)` reçoit la déformation **totale** — c'est le
contrat de tous les backends de contrainte plane de ce module, et la référence
écrit son gradient de façon absolue. Le pont GPS faisait :

```python
self._manager.s1.gradients[:, :] = s0.gradients + gradient
```

Il appliquait donc le total comme un **incrément**, et la déformation imposée
s'accumulait en `1+2+3+...` au lieu de `1,2,3`. À l'incrément 2, la trace en
plan valait `3,0e-3` là où `2,0e-3` était demandé.

L'incrément 1 n'était pas affecté — total et incrément y coïncident. **C'est
exactement pourquoi toutes les comparaisons contre la référence s'accordaient à
l'incrément 1 et divergeaient à partir du 2.**

Corrigé : le gradient en plan est écrit **absolu**, la transverse reste celle de
l'état committé et la fermeture y ajoute son incrément, et les sous-pas
interpolent le total entre l'état committé et la cible (une fraction d'un total
n'est pas une déformation).

#### Ce que la correction emporte avec elle

| | avant | après |
|---|---|---|
| A3 fermeture (MPa) | `4e-14` | `3e-14` |
| A6 tangente FD, C1 | `7,4e-01` | **`1,6e-07`** |
| A6 tangente FD, C2 | `5,2e+00` | **`1,2e-07`** |
| A6 tangente FD, C3 | `3,2e-07` | **`1,4e-07`** |
| écart à la référence, C1 | `2,7e-01` | **`1,1e-11`** |
| écart à la référence, C2 | `4,4e-01` | **`7,4e-11`** |
| écart à la référence, C3 | `1,7e-01` | **`4,8e-11`** |
| verdict | REJETÉ | **ACCEPTÉ** |

**Il n'y avait pas deux branches.** Le F1 du §5.1, la « multiplicité » du
diagnostic de branches, le refus de la loi 3D brute de reproduire l'état UMAT
(§8.6), le mur de robustesse du §5.2 (la déformation croissait
quadratiquement !) et la tangente fausse du §8.3 sont **tous** des conséquences
de cette seule ligne. La décision « on force la racine de contrainte plane »
n'a plus d'objet, et les campagnes condensées archivées **n'ont pas à être
refaites** : la référence avait raison depuis le début.

`eps_zz` à l'incrément 2, identité : `-1,684e-3` contre `-1,65e-3` pour la
référence, et la marche est désormais linéaire.

#### Ce qui reste vrai des sections précédentes

- Le sous-pas (§8.2) reste utile mais marginal : `max_div` tombe de 256 à 32, et
  C3 n'en a plus besoin du tout. Conservé comme filet.
- La réparation d'A6 (§8.3) reste indispensable : le contrôle vacu aurait
  rapporté un succès sur une tangente fausse, et c'est lui qui a rendu la
  correction vérifiable.
- Le §8.6 (la route 1 ne tire pas, le bloc `@CompareToNumericalJacobian` de
  TFEL 5.1 lit de mauvaises colonnes) reste valable.
- Le §8.7 est **rétracté** : l'identité `tr(eel) = tr(eps_total)` est satisfaite
  à `1e-13`; le défaut de 100 % que j'y rapportais venait de ma propre
  reconstruction du total à partir des variables internes, pas de la loi.
- La route 2 reste éteinte : elle n'a plus d'objet.

### 8.9 P43 20x20 : il tourne — et il est plus lent

Le cas polycristallin échouait à l'**incrément 1**, résidu bloqué à `0,68`,
alors que la qualification au point matériel passait à `1e-11`. La différence
entre les deux : mes cas de qualification sont mono-orientation et **à un seul
point matériel**.

**Deuxième défaut trouvé, dans le même bloc que Q.** Le pont écrit les neuf
composantes de la rotation en propriétés matériau avec `ExternalStorage` —
MGIS conserve donc un *pointeur* et lit la mémoire comme contiguë. Or :

- `rotations[:, row, column]` est une **vue à pas de neuf doubles**. Passée
  comme span, chaque point lit les composantes `Q` d'un autre point. **Avec un
  seul point il n'y a rien à enjamber** : c'est précisément pourquoi toutes les
  qualifications mono-orientation passaient pendant que les 400 points EBSD
  démarraient leur premier résidu à `0,835` contre `0,178` pour la référence.
- les tampons sont des **temporaires**, libérés à la sortie de la boucle, MGIS
  pointant ensuite sur de la mémoire recyclée.

Corrigé : copies contiguës, conservées sur l'instance (`_property_buffers`).
Le même risque existait sur `Temperature` et sur les variables externes de
prédiction ; traité au même endroit.

**Résultat sur P43 20x20, huit incréments, quatre threads :**

| | référence condensée | UMAT GPS |
|---|---|---|
| incréments | 8 | **8** |
| itérations de Newton | 46 | 69 |
| temps total | `2,53 s` | `8,77 s` |
| temps matériau | `1,88 s` | `8,05 s` |
| **gain matériau** | — | **`0,23×`** |

| champ | écart relatif L2 |
|---|---|
| déplacement | `4,2e-07` |
| forces de réaction | `3,1e-03` |
| contrainte en plan | `3,2e-03` |
| glissements | `2,5e-03` |
| glissement cumulé | `9,5e-04` |

**Le cas converge, et l'argument de vitesse tombe.** Les `6,7×` annoncés au §4
étaient mesurés sur **un point**, mono-orientation, sans sous-pas — et sous le
chargement quadratique du défaut du §8.8. À l'échelle du polycristal réel le
backend UMAT est **4,3× plus lent** que la condensation Python : le sous-pas,
qui reste nécessaire, multiplie le coût d'un Newton conjoint par le nombre de
divisions et détruit l'avantage du « un Newton au lieu d'un Newton imbriqué ».

L'écart de champ de `3e-3` n'est pas expliqué. Il est du bon ordre de grandeur
pour de l'erreur de discrétisation constitutive due au sous-pas (mesurée à 3 %
entre 1 et 256 sous-pas au §8.8), mais cela n'est pas démontré.

**Le levier suivant est donc la robustesse du Newton conjoint, pas la
fermeture** : tant qu'il faut sous-passer, l'UMAT coûte plus cher qu'il ne
rapporte. La référence condensée reste le backend de production.

### 8.10 Peut-on éviter le sous-pas ? Ce que la référence a, et ce qui se transfère

Trois stratégies de `MFront3DCondensedPlaneStressBatch` ont été examinées et
essayées. Une seule a payé, et ce n'est pas celle qu'on attendait.

**1. Le prédicteur transverse — non transférable tel quel.** La référence part
sa fermeture de `eps_b_accepté - Cbb^-1 Cba (eps_a - eps_a_accepté)`, une
extrapolation exacte au premier ordre. Toute la machinerie
(`_accepted_transverse`, `_accepted_cbb`, `_accepted_cba`,
`accept_global_trial`) **était déjà dans le pont GPS, jamais utilisée**. Une
fois branchée, elle ne produit rien : **`Cbb` est identiquement nul**, parce que
la fermeture impose `sigma_transverse = 0` et que les lignes transverses de la
tangente GPS sont donc nulles par construction. Le `Cbb` de la référence est un
bloc de la tangente 3D **non contrainte**, que la loi GPS n'expose pas.

Un prédicteur de remplacement a été implémenté — l'incrément transverse accepté
précédent, remis à l'échelle de l'incrément en plan. Il laisse `3e-05` à
corriger au lieu de `1e-03`, soit un départ trente fois plus proche. **Il ne
change pas d'une unité le nombre de sous-pas.**

**2. Un critère de convergence à l'échelle de la contrainte — perte sèche.**
La référence converge sur `1e-8 MPa + 1e-10 |sigma|`; la loi utilise
`@Epsilon 1e-12` sur la norme du résidu mixte à 21 composantes. Relâcher à
`1e-9` : sous-pas **inchangé**, et A6 se dégrade de `1,2e-07` à `3,4e-05`,
au-delà de la tolérance. Rétabli à `1e-12`.

**3. Supprimer le travail inutile — le seul gain réel.** La route 1 (rerun de
l'incrément complet depuis la racine localisée) avait été mesurée bit à bit
inerte au §8.6. Pire : quand son propre essai échouait, elle **refaisait tout
le sous-pas**. Chaque incrément sous-passé payait donc le sous-pas deux fois,
plus un essai complet voué à l'échec. Retirée.

| | avant | après |
|---|---|---|
| appels sous-passés (12 incréments) | 18 / 22 | **9 / 11** |
| P43 20x20, temps matériau | `8,05 s` | **`4,94 s`** |
| gain matériau contre la référence | `0,23×` | **`0,39×`** |

Champs et qualification inchangés au bit près, ce qui confirme au passage que
la route 1 et les prédicteurs étaient numériquement sans effet.

**Conclusion : le sous-pas n'est pas évitable par ces leviers.** Il n'est causé
ni par un mauvais point de départ, ni par un critère trop serré. C'est le
Newton conjoint à 21 inconnues lui-même qui échoue, sur le chemin de
déformation désormais correct — sans recherche linéaire ni contrôle de pas, ce
que le DSL Implicit n'offre pas.

Ce qui reste, non essayé : `@Algorithm LevenbergMarquardt` (amortissement, une
ligne — `PowellDogLeg` est réputé échouer partout, LM ne l'a jamais été), ou
sortir la fermeture du Newton conjoint pour en faire une boucle externe en C++
— ce qui est la structure de la référence, et retire à l'UMAT sa seule raison
d'être. À `0,39×`, l'UMAT reste **2,6× plus lent** que la condensation Python :
le backend de production ne change pas.

### 8.11 Où passe le temps, et ce qui reste comme levier

Trois mesures, et le suspect principal est innocenté.

**Le sous-pas n'est pas le coût.** Avec le sous-pas **désactivé**, aux mêmes
états, sur 400 points et 4 threads
(`scripts/diagnose_gps_local_solve_cost.py`) :

| | par point |
|---|---|
| loi brute, 18 inconnues | `10 – 14 µs` |
| loi GPS, 21 inconnues, **sans sous-pas** | `79 – 113 µs` |
| **rapport** | **`7,7 – 8,1×`** |

Le facteur ~8 du benchmark P43 est donc le prix intrinsèque du Newton conjoint,
pas le sous-pas.

**Ce n'est pas non plus le nombre d'itérations.** Compteur `LocalIterations`
ajouté à la loi : **4, 5, 8** itérations locales aux incréments 1 à 3 — des
valeurs normales pour un return mapping. `21³/18³ = 1,6` fois deux fois plus
d'itérations donne `3,2`, pas `8`. **Il reste un facteur ~2,5 inexpliqué.**

**Ce n'est pas le recalcul de constantes.** Neuf `gpsRotate` sur des tenseurs
constants et six produits `D·u_j` constants étaient refaits à chaque itération.
Hissés dans `@InitLocalVariables` : rapport `7,74 → 8,09`, c'est-à-dire dans le
bruit. Le hissage est conservé — c'est du travail provablement redondant en
moins — mais il ne rapporte rien de mesurable.

**`LevenbergMarquardt` confirme la diagnose et échoue sur le coût.** Le sous-pas
s'effondre — identité `9 → 0`, `bunge_54` reste à `0`, `bunge_35` `11 → 8` :
l'échec du Newton conjoint **est** bien un défaut de globalisation, et un
amortissement le corrige. Mais le coût explose à `828 µs/point`, soit `73×` la
loi brute. Rétabli à `NewtonRaphson`.

#### Ce que cela laisse

L'avantage structurel est réel et acquis : **3,3× moins d'appels constitutifs**
(56 000 contre 121 600). Il faut passer sous `41 µs` par intégration pour que
l'UMAT gagne, contre `79 – 113 µs` aujourd'hui.

La piste la plus prometteuse reste la **réduction de la taille du système
local**, parce que `sigma = D : eps_el` est exactement linéaire : la contrainte
plane est donc une relation linéaire **constante** entre composantes de la
déformation élastique,

```
eps_el,g,b = − D_g,bb^-1 D_g,ba · eps_el,g,a ,   D_g = R D R^T constante
```

ce qui ramène la loi à **15 inconnues** — trois déformations élastiques dans le
plan global et douze glissements — sans lignes de fermeture, sans bloc diagonal
nul, et **plus petite que la loi brute**.

**Mais il faut d'abord expliquer le facteur 2,5 restant.** Si le surcoût du
solve local ne vient pas de sa taille, passer de 21 à 15 inconnues n'y changera
rien non plus. Le profilage du solve local (assemblage du jacobien contre
factorisation contre inversion pour la tangente cohérente) est le prochain pas,
et il doit précéder toute réécriture.

### 8.12 Le facteur 2,5 : le pont GPS n'était pas parallélisé

Intuition de l'utilisateur, vérifiée en un grep, et c'était bien elle.

`MFront3DMaterialPointBatch` — la loi brute, et donc la référence — a **deux**
chemins d'intégration :

```python
if self._thread_pool is None:
    status = self._mgis.integrate(manager, type, dt, 0, point_count)   # série
else:
    status = self._mgis.integrate(self._thread_pool, manager, type, dt)  # 4 fils
```

`MFrontNativeGeneralisedPlaneStressBatch` n'avait **que la surcharge série**, et
ne construisait même pas de `ThreadPool`. Il intégrait sur un fil pendant que la
référence en utilisait quatre. **Les deux backends n'étaient pas comparés sur le
même nombre de cœurs.**

| coût par point, sans sous-pas | avant | après |
|---|---|---|
| loi brute 18 inconnues | `13,0 µs` | `13,0 µs` |
| loi GPS 21 inconnues | `75 – 113 µs` | **`26,2 µs`** |
| rapport | `5,7 – 8,1×` | **`2,02×`** |

`2,02` est exactement ce que prédit l'arithmétique : `21³/18³ = 1,59` pour la
factorisation, fois un peu plus d'itérations locales (4 à 8 contre 3 à 5).
**Le facteur inexpliqué n'existe plus.**

Sur P43 20x20, le gain matériau contre la référence :

| | temps matériau | gain |
|---|---|---|
| avant la route 1 retirée | `8,05 s` | `0,23×` |
| après | `4,94 s` | `0,39×` |
| **après la parallélisation** | **`2,38 s`** | **`0,89×`** |

contre `2,12 s` pour la référence : **la parité est atteinte**, à 12 % près.

#### Ce qui sépare encore de la victoire

Il reste **69 itérations globales contre 46**, soit `1,5×`. Ce n'est pas la
tangente : A6 vaut `1,2e-07` à *chaque* incrément, y compris les neuf sur douze
qui sous-passent — la conclusion du §8.3 selon laquelle « le sous-pas détruit la
tangente » était elle-même un artefact du chargement quadratique du §8.8, et
elle est **rétractée**.

Avec `3,3×` moins d'appels, un surcoût par appel de `2,0×` et une pénalité
d'itérations globales de `1,5×`, le compte est `3,3 / (2,0 × 1,5) = 1,1` — la
parité mesurée. Fermer l'écart des itérations globales, ou ramener le coût par
appel sous `2×` par la réduction à 15 inconnues, ferait basculer le bilan.
La réécriture à 15 inconnues garde donc tout son sens, et pour la première fois
elle vise un objectif atteignable et non un facteur inexpliqué.

### 8.13 Réduction du système local : 21 -> 18 inconnues

**Le principe.** Le résidu élastique est écrit dans le repère **global** :

```
feel_g = rot(deel + Σ dg·m, Qᵀ) − deto
```

Ses trois lignes **dans le plan** sont la cinématique. Ses trois lignes
**transverses** ne sont pas des équations du problème — la surface libre laisse
la déformation transverse totale indéterminée — donc elles sont libres, et on y
met `sigma_g,b / G = 0`. La contrainte plane devient une *ligne de résidu*, plus
une inconnue.

Conséquences dans la loi :

- `ezz, eyz, exz` passent de `@StateVariable` à `@AuxiliaryStateVariable` :
  ce sont des **sorties**, calculées après coup par `eps = eel + Σ g·m` ;
- **plus de lignes de fermeture, plus de bloc diagonal nul** ;
- `∂feel/∂deto = −P` avec `P = diag(1,1,0,1,0,0)`, un projecteur **constant** :
  la correction de tangente côté pont n'est plus une rotation, et l'aller-retour
  `rotateGradients` qui la construisait disparaît ;
- tout le bloc élastique du jacobien est constant, assemblé une fois dans
  `@InitLocalVariables` (colonnes de la rotation, de la raideur tournée et des
  douze tenseurs de Schmid tournés).

Système local vérifié dans le code généré : `tmatrix<StensorSize+12,...>`,
soit **18×18** au lieu de 21×21.

**Résultat.** Qualification **ACCEPTÉE**, aux mêmes chiffres au bit près :
fermeture `2e-14 MPa`, tangente FD `1,2 – 1,6e-07`, écart à la référence
`1e-11`. P43 20x20 :

| | temps matériau | gain |
|---|---|---|
| 21 inconnues, parallélisé | `2,38 s` | `0,89×` |
| **18 inconnues** | **`2,13 s`** | **`0,98×`** |

contre `2,09 s` pour la référence : **parité exacte**.

**Et une prédiction fausse, à consigner.** J'attendais `21³/18³ = 1,59` sur le
coût par intégration. Mesuré : `26,2 → 26,0 µs/point`, **inchangé**. La
factorisation du système local n'est donc pas dominante non plus. Après le
sous-pas, le nombre d'itérations, le recalcul de constantes, la parallélisation
et maintenant la taille du système, le surcoût de `2,1×` par appel reste sans
explication identifiée.

Le bilan se lit : `3,3 / (2,1 × 1,5) = 1,05`, contre `0,98` mesuré. Les deux
leviers restants sont donc **l'écart d'itérations globales** (69 contre 46, le
plus gros des deux) et ce `2,1×` résiduel. La descente à 15 inconnues — qui
exige d'abandonner le brick `StandardElasticity` — n'a plus de justification
tant que la taille du système est mesurée sans effet.

La reformulation est conservée malgré le gain modeste : elle supprime le
point-selle, trois inconnues, trois équations, une rotation par évaluation de
tangente, et rend le jacobien élastique constant. Le code est plus simple et
plus proche de la loi brute.
