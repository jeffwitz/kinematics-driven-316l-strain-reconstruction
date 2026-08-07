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
