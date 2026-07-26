# Plan de mise à niveau de `fem_inhouse`

Dernière mise à jour : 2026-07-26
Statut global : **pipeline autonome DIC → entrées canoniques → calcul
partitionné validé sur une partition article de 234 600 éléments ; backend
MFront/MGIS branché dans Newton et validé sur le crop DIC réel 10×10 ;
tenseurs 3D complets reconstruits en post-traitement du solveur 2D et validés
sur les deux backends sans modification des sorties historiques ;
loi J2 tridimensionnelle condensée localement en contraintes planes et validée
contre le backend MFront natif, les trois backends mesurés sur un crop DIC
100×100 avec neuf exécutions intégralement sauvegardées, interface prête à
recevoir une loi 3D ; diagnostic de largeur spatiale par filtre de Helmholtz
implémenté avec sélection pré-enregistrée sur P48 et confirmation sans
ajustement sur P42, avec hypothèse de largeur spatiale soutenue mais aucune
longueur matérielle identifiée ;
deux comportements MFront micromorphiques ajoutés, point fixe
`p ↔ chi` transactionnel branché dans chaque Newton mécanique, sorties et
diagnostics non locaux sauvegardés ; campagne P154 padding 128 terminée pour
`alpha=0,5`, `1` et `2`, avec interaction spatiale partiellement soutenue mais
aucun `Hchi` admissible à figer selon tous les critères pré-enregistrés ;
P43 retenue après inspection visuelle comme prochaine ROI scientifique à deux
bandes ; chemin constitutif micromorphique allégé et validé avant campagne,
avec `1,89×` de gain sur le cœur P43 et `1,45×` sur un calcul EF complet
intermédiaire sans changement scientifique ; CSR triangulaire et PARDISO
symétrique `mtype=2` activés par défaut pour les comportements J2 vérifiés,
avec `mtype=11` conservé pour les comportements non classifiés ;
exécution et raccordement des 100 partitions du ROI complet à planifier**
Objectif de maturité : **au moins 4/5 sur tous les axes**

Jalon atteint au 2026-07-25 : **couplage constitutif micromorphique J2 sur la
partition P154 d'un découpage 20×20**. Le protocole et les résultats sont
respectivement dans `validation/nonlocal_p154_preregistration.md` et
`validation/nonlocal_p154_validation_results.md`. Le meilleur point testé
(`alpha=2`) passe sept critères sur huit, mais l'aire active q90 reste à
`21,85 %` pour une borne pré-enregistrée de `20 %`. La conclusion est donc
« interaction spatiale partiellement soutenue » et aucun `Hchi` n'est figé.

## 1. Rôle de ce document

Ce fichier est la feuille de route vivante du projet. Il doit être mis à jour à
chaque jalon avec :

- l'état réel des tâches ;
- les décisions scientifiques et techniques ;
- les commandes de validation exécutées ;
- les résultats mesurés ;
- les écarts restant à traiter ;
- la date et, si disponible, le commit correspondant.

Une tâche ne doit être marquée terminée que si son critère d'acceptation est
vérifié par un test, un rapport ou une mesure reproductible.

### Convention d'état

- `[ ]` : à faire
- `[~]` : en cours
- `[x]` : terminé et vérifié
- `[!]` : bloqué

## 2. Sources de vérité

1. `ArticleSource/ArticleAdil.pdf`
2. Les données DIC et cartes de paramètres versionnées dans
   `data/raw/case_study`
3. Le manifeste de provenance et le profil de préparation produits par le
   présent projet
4. Les tests automatisés du présent projet
5. À titre de validation différée : les fichiers d'entrée Abaqus, ODB et
   scripts d'extraction ayant produit les résultats historiques

Toute contradiction entre l'article, les entrées Abaqus et le code doit être
documentée et résolue explicitement. Le comportement courant du code n'est pas
considéré comme une spécification par défaut.

### Priorité de développement

Le chemin critique est la reproduction du calcul **à partir des données DIC**.
Le dépôt doit permettre, depuis un clone neuf :

1. de récupérer et vérifier les données scientifiques brutes versionnées ;
2. de les convertir sans opération implicite vers le contrat canonique ;
3. de lancer ou reprendre les partitions indépendamment ;
4. de raccorder les champs globaux ;
5. de reconstruire les grandeurs de l'article depuis les déplacements ;
6. d'obtenir des manifestes et rapports contenant les paramètres, empreintes et
   versions du code.

La comparaison Abaqus reste souhaitable, mais elle n'est plus un prérequis au
développement ni à l'exécution de ce pipeline principal. Elle constitue une
campagne de validation externe ultérieure.

## 3. Objectif scientifique

Le projet ne cherche pas à reproduire Abaqus de manière générale.

Il doit fournir, pour le cas d'étude de l'article, un moteur de reconstruction
cinématique :

- mécaniquement admissible ;
- limité aux petites déformations et aux contraintes planes ;
- fondé sur un maillage CPS4 rectangulaire structuré ;
- piloté par les déplacements DIC prescrits aux frontières ;
- utilisant des descripteurs élastoplastiques effectifs identifiés à l'échelle
  du pixel ;
- capable de reconstruire l'organisation spatiale des bandes de localisation ;
- capable de traiter le domaine complet par sous-domaines avec recouvrement et
  raccordement ;
- reproductible depuis les quatre tableaux bruts versionnés, sans chemin
  personnel ni donnée cachée.

Les paramètres locaux sont des **descripteurs effectifs dépendant du chargement,
de la résolution DIC et des hypothèses constitutives**. Ils ne doivent pas être
présentés comme des propriétés intrinsèques des grains.

## 4. Périmètre supporté

### Inclus

- matériau 316L du cas d'étude ;
- élasticité homogène : `E = 205 GPa`, `nu = 0.30` ;
- plasticité J2/von Mises ;
- loi de Ludwik-Hollomon ;
- exposant `n = 0.245` pour le cas nominal ;
- cartes spatiales de limite d'élasticité et de coefficient d'écrouissage ;
- contrainte plane ;
- éléments quadrilatéraux bilinéaires CPS4, intégration 2×2 ;
- maillage régulier à un élément par pixel ;
- déplacements mesurés imposés sur les frontières des sous-domaines ;
- résolution sparse avec PyPardiso/MKL ;
- partitionnement sans recouvrement et avec padding ;
- raccordement des cœurs des partitions ;
- post-traitement DIC/EF commun à partir des déplacements ;
- comparaison avec les champs expérimentaux ;
- préparation traçable des données DIC historiques.

### Hors périmètre

- maillages non structurés ;
- éléments autres que CPS4 ;
- grandes transformations ;
- contact ;
- endommagement et rupture ;
- dynamique ;
- 3D ;
- plasticité cristalline ;
- chargements généraux sans rapport avec le cas d'étude ;
- solveur EF généraliste ou remplacement global d'Abaqus.

La comparaison avec Abaqus appartient au périmètre de validation, mais peut
être réalisée après la mise en service du calcul autonome depuis la DIC.

## 5. Échelle du problème de production

- ROI : `3600 × 3100` pixels
- Nombre d'éléments : environ `11,16 millions`
- Domaine physique : `6,624 × 5,704 mm²`
- Taille de pixel : `1,84 µm`
- Schémas étudiés dans l'article :
  - 25 partitions sans recouvrement ;
  - 25 partitions avec padding ;
  - 100 partitions avec padding ;
- Padding de production mentionné dans l'article : environ 150 éléments

Le solveur ne doit pas tenter de charger ou résoudre le domaine complet de
manière monolithique. Les entrées, sorties et raccordements doivent être conçus
pour fonctionner hors mémoire.

## 6. État initial vérifié

### Points positifs

- [x] Environnement `.venv` créé
- [x] NumPy, SciPy et Matplotlib installés
- [x] PyPardiso 0.4.7 et MKL 2026.1.0 installés
- [x] Backend réellement sélectionné :
  `pypardiso (MKL, multithreaded)`
- [x] Test biaxial homogène 20×20 exécuté avec succès
- [x] Erreur affichée sur la contrainte de von Mises : 0 %
- [x] Tangente cohérente vérifiée par différences finies
- [x] Erreur relative de tangente observée : `1e-10` à `7e-9`
- [x] Matrice élémentaire symétrique avec trois modes rigides
- [x] Cas hétérogène convergeant en quatre itérations de Newton par incrément
- [x] Équilibre global observé de l'ordre de `1e-14`

### Blocages et défauts initiaux

Une case cochée dans cette liste signifie que le défaut initial a été corrigé
et vérifié ; une case vide indique qu'il reste à traiter.

- [x] `test_config.py` est absent du projet livré
- [~] Les scripts de validation ne sont pas exécutables de manière autonome
- [x] Des chemins Windows absolus sont présents
- [x] La courbe étiquetée « FEM stress » remplace la contrainte EF directe par
      une reconstruction de Ludwik après plastification
- [x] Les quatre courbes scientifiques de l'article ne sont pas séparées
- [x] La table plastique par défaut utilise 50 points, contre 1000 points dans
      l'article
- [x] Les conventions d'axes DIC ne sont pas cohérentes dans tous les scripts
- [x] Les conventions cisaillement tensoriel/ingénieur ne sont pas garanties
- [x] Le seul test intégré n'asserte pas la valeur de PEEQ
- [x] Aucun moteur de partitionnement/raccordement n'existe
- [x] Aucun traitement hors mémoire du ROI complet n'existe
- [x] Aucun manifeste de dépendances ou verrouillage des versions n'existe
- [x] Aucun historique Git exploitable n'est présent dans le dossier
- [ ] Aucun seuil automatique de parité Abaqus n'est défini
- [x] Les quatre tableaux scientifiques du ROI sont absents du dépôt
- [~] Aucun pipeline autonome ne transforme les noms, unités et conventions
      historiques vers les quatre entrées canoniques
- [ ] La règle de complétion nodale `3600×3100 → 3601×3101` n'est pas encore
      ratifiée scientifiquement
- [ ] L'écart entre le facteur d'écrouissage `380 MPa` de l'article et
      `396 MPa` du générateur historique doit rester explicite et paramétrable

## 7. Grandeurs scientifiques à maintenir séparées

Le logiciel doit produire et nommer sans ambiguïté :

1. la courbe macroscopique mesurée ;
2. la contrainte reconstruite depuis la déformation DIC ;
3. la contrainte reconstruite depuis la déformation EF ;
4. la contrainte EF directe calculée depuis `S11`, `S22`, `S12`.

Les courbes 2 et 3 sont des contrôles de cohérence obtenus en réappliquant la loi
constitutive à une mesure de déformation. Elles ne constituent pas des
prédictions indépendantes de contrainte.

L'écart entre la contrainte EF directe et la courbe macroscopique mesurée doit
être conservé et analysé. Il ne doit pas être corrigé par le post-traitement.

## 8. Planification révisée

Durée prévisionnelle : **12 semaines pour une personne à temps plein**, avec
revue scientifique régulière.

### Phase prioritaire A — Dépôt autonome depuis la DIC

- [x] Versionner sous Git LFS les quatre tableaux bruts sans les modifier
- [x] Enregistrer forme, type, taille, rôle et SHA-256 de chaque tableau
- [x] Conserver les générateurs Abaqus reçus uniquement comme provenance
- [x] Ajouter `fem-inhouse prepare-case`
- [x] Vérifier les empreintes avant toute transformation
- [x] Convertir `V → u_x`, `U → u_y` et pixels → millimètres
- [x] Rendre le facteur macroscopique `K` explicite, avec `380 MPa` nominal et
      `396 MPa` historique
- [x] Détecter les neuf valeurs non finies et appliquer seulement une politique
      explicitement sélectionnée et enregistrée
- [x] Compléter la grille nodale selon une politique explicite et enregistrée
- [x] Écrire les quatre `.npy` canoniques et un manifeste reproductible
- [x] Ajouter un test d'intégration depuis des données brutes synthétiques
- [x] Ajouter un contrôle d'intégrité des données réelles, sans les charger
      entièrement en mémoire
- [x] Documenter une séquence unique `clone → install → prepare → partition →
      stitch → postprocess`
- [x] Exécuter un sous-domaine réel versionné depuis cette séquence

**Critère de sortie :** aucune donnée ou transformation scientifique nécessaire
au calcul principal ne se trouve hors du dépôt ou dans un chemin personnel.

### Phase prioritaire A.1 — Remplacement constitutif par MFront

- [x] Installer depuis les sources TFEL/MFront 5.1.0 et MGIS 3.1
- [x] Épingler les tags, commits, options CMake et procédure d'activation
- [x] Implémenter la loi J2/Ludwik sous l'hypothèse `PlaneStress`
- [x] Exposer les cartes locales `sy0`, `K` et `n` comme propriétés matériau
- [x] Compiler l'interface générique MFront de façon reproductible
- [x] Ajouter l'adaptateur Python MGIS avec conversions Kelvin/ingénieur
- [x] Gérer explicitement les états d'essai, `commit` et `revert`
- [x] Comparer et sauvegarder trois trajets au point matériel sur 200 incréments
- [x] Déclarer les seuils avant la comparaison et conserver tous les champs
- [x] Retenir la loi MFront analytique régularisée sans plafond de PEEQ ; garder
      les 1000 segments uniquement comme régression historique explicite
- [x] Brancher MFront derrière une sélection de backend dans la boucle Newton
- [x] Vérifier la tangente MFront dans les conventions d'assemblage CPS4
- [x] Comparer les deux backends sur le crop DIC réel `10×10`
- [x] Mesurer coût et mémoire sur une partition à la taille de l'article ;
      `510×460` éléments mesurés avec MFront en 650,08 s et 4 163 308 KiB RSS,
      sans construire la table Python
- [x] Basculer le backend par défaut vers MFront après parité du sous-domaine

**Critère de sortie :** le même sous-domaine DIC converge avec les deux
backends, les six champs sauvegardés respectent des seuils ratifiés, et aucun
état MFront d'une itération Newton rejetée n'est commis.

### Phase prioritaire A.2 — Tenseurs 3D complets en contraintes planes

- [x] Maintenir strictement le solveur, les inconnues, les éléments, Newton et
      le tangent condensé en 2D
- [x] Centraliser les conversions engineering, tensorielle et Kelvin
- [x] Reconstruire `ep33` par incompressibilité plastique J2 pour Python
- [x] Reconstruire `ee33` par élasticité isotrope en contraintes planes
- [x] Assembler `S_3D`, `E_3D`, `EE_3D`, `PE_3D` après convergence seulement
- [x] Préserver `S`, `E`, `PE`, `PEEQ` et toutes leurs conventions historiques
- [x] Identifier par métadonnées MGIS `AxialStrain`, `ElasticStrain` et
      `Stress`, sans supposer leurs offsets
- [x] Vérifier par essai matériel que le gradient Kelvin ne porte pas le
      `e33` natif de ce comportement
- [x] Conserver le `S33` MFront natif comme résidu de contraintes planes
- [x] Interdire le fallback analytique implicite MFront ; n'autoriser la
      complétion J2 qu'avec la capacité explicite
      `j2_isotropic_analytical`
- [x] Étendre `FEMResult`, les exports de partitions, le raccordement et le
      chargeur des résultats anciens
- [x] Séparer `EVM_HISTORICAL` de `EVM_RECONSTRUCTED_3D`
- [x] Tester traction, traction équibiaxiale, cisaillement, déchargement et
      chargement non proportionnel
- [x] Comparer Python/MFront sur le crop DIC réel `10×10`
- [x] Comparer les six champs historiques avec la campagne antérieure
- [x] Finaliser la documentation Sphinx et reconstruire HTML/PDF

**Preuve DIC 10×10 :**
`validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1`.
Les trois groupes de contrôles passent. Le maximum `|S33|` vaut `0` pour
Python et `1,046e-14 MPa` pour MFront ; le maximum
`|trace(epsilon_p)|` vaut respectivement `0` et `1,406e-19` ; le maximum de
la décomposition additive vaut `8,132e-20` et `1,355e-19`. La différence
maximale avec les sorties historiques est nulle pour Python et
`4,263e-14 MPa` pour MFront.

**Critère de sortie :** tout résultat FEM convergé expose les quatre tenseurs
symétriques `3×3` et le résidu `S33`, sans nouvelle résolution mécanique et
sans régression des sorties 2D.

### Phase prioritaire A.3 — Loi 3D condensée en contraintes planes

- [x] Ajouter les conversions Kelvin 3D à six composantes et vérifier l'ordre
      MGIS `[11,22,33,12,13,23]` par métadonnées et essais élémentaires
- [x] Généraliser le résidu à `[S33,S13,S23]` tout en conservant
      `S33_RESIDUAL_MPA` comme vue compatible
- [x] Introduire le protocole transactionnel commun
      `PlaneStressMaterialBatch`
- [x] Adapter les backends Python J2 et MFront `PlaneStress` au protocole
- [x] Compiler la même loi J2/Ludwik sous l'hypothèse `Tridimensional`
- [x] Résoudre localement `[epsilon33,gamma13,gamma23]` depuis le même état
      constitutif validé à chaque itération
- [x] Condenser la tangente 6×6 par complément de Schur sans inversion
      explicite
- [x] Ajouter les diagnostics de résidu au point de Gauss, itérations locales,
      échecs locaux et conditionnement de `Cbb`
- [x] Vérifier la tangente condensée par différences finies dans un état
      plastique éloigné du seuil
- [x] Tester l'échec local et l'absence de pollution de l'état validé
- [x] Comparer les deux chemins sur les trajets matériels et un maillage 4×4
- [x] Comparer et sauvegarder les deux chemins sur le crop DIC réel 10×10
- [x] Comparer temps complet, temps constitutif et pic RSS des trois backends
      sur le même crop DIC 100×100, trois processus frais par backend
- [x] Documenter l'architecture, ses limites et le contrat pour une future loi
      cristalline 3D

**Preuve DIC 10×10 :**
`validation/reference_data/mfront_3d_condensed_dic_10x10_v1`. Les deux chemins
convergent en 66 itérations Newton globales, sans cutback. L'écart maximal sur
la contrainte dans le plan vaut `4,804e-08 MPa`. Le backend condensé atteint
un résidu transverse maximal au point de Gauss de `2,705e-08 MPa` en quatre
itérations locales au plus, avec zéro échec et
`max(cond(Cbb)) = 1,896`.

**Preuve de performance DIC 100×100 :**
`validation/reference_data/plane_stress_backend_performance_100x100_v1`.
Les neuf calculs convergent sans cutback. Les médianes
temps mur / pic RSS sont `134,36 s / 248,96 MiB` pour Python,
`27,03 s / 269,65 MiB` pour MFront natif et
`83,43 s / 320,30 MiB` pour MFront 3D condensé. Les deux chemins MFront
diffèrent au maximum de `2,307e-07 MPa` sur la contrainte ; Python diffère au
maximum de `6,763e-02 MPa` et respecte tous les seuils déclarés du cas
d'étude.

**Critère de sortie :** le solveur global ne connaît plus la loi J2 ou les
détails MGIS ; les chemins J2 MFront natif et J2 3D condensé sont équivalents
aux tolérances numériques sur le cas DIC, et la substitution future d'une loi
3D petites déformations reste confinée à l'adaptateur constitutif.

### Phase prioritaire A.4 — Diagnostic de non-localité par Helmholtz

- [x] Ajouter un filtre scalaire de Helmholtz aux centres des éléments
      structurés, avec flux nul et résolution DCT orthonormale
- [x] Garantir que `ell=0` restitue une copie exacte, sans DCT ni projection
      élément-nœud-élément
- [x] Vérifier conservation de la moyenne, principe du maximum, décroissance
      de variance, anisotropie `hx != hy`, résidu et référence sparse directe
- [x] Reconstruire séparément EVM DIC et EVM FEM avec la chaîne commune
      `strain_from_displacement → plane_stress_equivalent_strain →
      cell_average`
- [x] Filtrer le domaine résolu complet avec padding et calculer les métriques
      uniquement sur le cœur issu des métadonnées
- [x] Signaler les longueurs dont le rapport padding/longueur est inférieur au
      seuil numérique configurable
- [x] Ajouter les erreurs de champ, recouvrements par quantile, seuils absolus
      DIC et métriques de diffusivité
- [x] Conserver PEEQ comme indicateur interne séparé, sans RMSE ou MAE
      d'amplitude contre EVM DIC
- [x] Ajouter les modes exploratoire et confirmatoire, avec seuils
      confirmatoires fournis en YAML ou JSON avant calcul
- [x] Ajouter `fem-inhouse diagnose-nonlocality`, les rapports atomiques,
      manifestes, champs et figures reproductibles
- [x] Exécuter le balayage `0–58,88 µm` sur la partition article 0 sauvegardée
      avec padding 150 pixels
- [x] Documenter la méthode et la campagne selon Diátaxis

**Preuve partition article :**
`validation/reference_data/nonlocality_helmholtz_article_p0000_v1`.
Le filtre porte sur les `510×460` éléments résolus et les métriques sur le
cœur `360×310`. À `58,88 µm`, RMSE et erreur L2 relative diminuent de
`49,45 %`, la corrélation passe de `-0,0292` à `0,0926` et l'IoU des 10 %
les plus élevés de `0,0503` à `0,1312`. La moyenne dérive au plus de
`8,674e-19`, le résidu relatif au plus de `5,575e-13`, et toutes les longueurs
respectent `padding/ell >= 4`.

**Interprétation :** hypothèse de largeur spatiale **partiellement soutenue**
sur cette partition exploratoire. Le meilleur point des critères principaux
est la borne supérieure du balayage et atténue fortement les pics ; la
corrélation reste faible. Aucune longueur interne matérielle n'est identifiée.
Une confirmation devra fixer la longueur sur une partition puis l'appliquer
sans ajustement à des partitions tenues à l'écart.

**Révision pré-enregistrée avant nouveau calcul :** la partition 0 est jugée
peu représentative à partir des figures 6 et 8. La partition 48, cœur
`x=[1440,1800)`, `y=[2480,2790)` et domaine paddé `660×610`, devient l'unique
partition de sélection. P0 est exclue de la sélection. La décision donnera la
priorité à la corrélation, à l'IoU top-10 % et au seuil absolu DIC 90 %, avec
RMSE/L2 comme métriques d'amplitude secondaires. Le protocole complet est figé
dans `validation/nonlocality_p48_preregistration.md` avant le calcul.

**Résultat de sélection P48 :** le calcul MFront converge sur 402 600 éléments
en `1335,97 s` de temps processus, avec 20/20 incréments, zéro cutback et
`7 869 356 KiB` de RSS maximal. Les trois métriques spatiales
pré-enregistrées sélectionnent `ell=58,88 µm` : corrélation
`0,2983 → 0,6160`, IoU top-10 % `0,1598 → 0,2822` et IoU au seuil absolu DIC
90 % `0,1676 → 0,3085`. RMSE et L2 relative diminuent de `64,61 %`. L'aire
active q90 reste `14,09 %` contre `10 %` DIC, sans collapsus. Le candidat étant
à la borne supérieure, l'optimum n'est pas encadré. Il est figé pour une
application sans ajustement aux partitions de confirmation.

**Confirmation tenue à l'écart :** P42, proposée avant l'exécution P48, est
pré-enregistrée comme premier cas de transfert. Seules les longueurs `0` et
`58,88 µm` seront comparées. Les seuils automatiques sont fixés avant calcul :
gain de corrélation `>=0,05`, réduction L2 relative `>=5 %`, gain d'IoU top-10
`>=0,02`, dérive moyenne relative `<=1e-10`. Au seuil absolu DIC 90 %, le gain
d'IoU doit être `>=0,02` et l'aire active filtrée rester entre 5 % et 20 %.
Voir `validation/nonlocality_p42_confirmation_preregistration.md`.

**Résultat confirmatoire P42 :** le calcul MFront converge sur 402 600
éléments en `1484,55 s` de temps processus, 20/20 incréments et zéro cutback.
Sans aucun ajustement de longueur, `58,88 µm` passe tous les seuils :
corrélation `0,4007 → 0,7036`, réduction L2 `65,43 %`, IoU top-10 %
`0,1334 → 0,2759`, IoU au seuil DIC 90 % `0,1774 → 0,2573`, et aire active
q90 `7,74 %` dans la plage pré-déclarée `[5 %,20 %]`.

**Conclusion de l'étape 1 : hypothèse de largeur spatiale soutenue.** Le même
candidat améliore les métriques d'amplitude et de localisation sur la
partition de sélection P48 et sur la partition P42 tenue à l'écart. Cette
conclusion ne transforme pas `58,88 µm` en longueur interne matérielle : le
point reste la borne supérieure du balayage et une seule partition de
confirmation est disponible.

**Critère de sortie :** un résultat FEM sauvegardé peut faire l'objet d'une
campagne de largeur spatiale traçable sans modifier le calcul mécanique, et le
rapport sépare faits numériques, sélection diagnostique et interprétation
physique.

### Phase prioritaire A.5 — Couplage constitutif micromorphique J2

- [x] Pré-enregistrer P154, le profil `20×20`, le padding 128, la longueur
      `58,88 µm`, le balayage de `Hchi` et les critères scientifiques
- [x] Ajouter la configuration typée et les options CLI non locales
- [x] Conserver la compatibilité `--count 25/100` et ajouter
      `--parts-x/--parts-y`
- [x] Ajouter les comportements MFront natif et tridimensionnel sans modifier
      les deux comportements de référence
- [x] Exposer `MicromorphicCouplingModulus` et
      `NonlocalEquivalentPlasticStrain`
- [x] Ajouter `Hchi*(p-chi)` au rayon de charge et `Hchi` à sa dérivée locale
- [x] Réutiliser le solveur DCT Helmholtz existant aux centres des éléments
- [x] Imbriquer le point fixe relaxé dans chaque essai Newton, sans `commit`
      intermédiaire
- [x] Restaurer conjointement déplacement, état MFront et `chi` lors d'un
      cutback
- [x] Sauvegarder `PEEQ_NONLOCAL`, `PEEQ_MISMATCH`,
      `NONLOCAL_HARDENING_MPA`, `YIELD_SURFACE_RADIUS_MPA` et
      `NONLOCAL_RESIDUAL`
- [x] Ajouter les temps MFront/Helmholtz, itérations, résidus, dérive de
      moyenne et échecs aux diagnostics
- [x] Vérifier que `Hchi=0` reproduit le calcul MFront local dans Newton
- [x] Vérifier le cas homogène, la tangente à `chi` fixé, les transactions et
      l'équivalence natif/3D condensé sur cas réduit
- [x] Ajouter une commande empreintée calculant
      `Href = median(K*n*p**(n-1))` sur le cœur plastifié local
- [x] Exécuter P154 local à 20 incréments et produire `HREF.json`
- [x] Ajouter un validateur empreinté local/couplé qui reconstruit les EVM
      depuis les déplacements bruts sur le cœur, sans post-filtrage
- [x] Exécuter les smoke tests à 5 incréments pour `alpha=0,0.5,1`
- [x] Exécuter les candidats retenus à 20 incréments avec padding 128
- [x] Comparer les champs bruts couplés à la DIC sur le cœur P154
- [!] Figer `Hchi` avant tout transfert vers P42 ou P48 : aucun candidat ne
      passe les huit critères pré-enregistrés
- [x] Rejouer un cas réduit avec le backend 3D condensé

**Preuve logicielle :** commits `3fe01d9`, `2102520`, `d3dfd33` et les commits
de validation ultérieurs. Le cas homogène couplé converge sans cutback. La
norme du point fixe est la norme mixte relative \(L_\infty\), indépendante du
nombre d'éléments, et utilise une branche absolue unitaire lors de
l'apparition de plasticité.

**Référence locale P154 :**
`validation/nonlocal_p154_local_reference.md`. Les 179 196 éléments convergent
en `793,98 s`, 20/20 incréments, 119 Newton et zéro cutback. Le cœur contient
24 507 éléments plastifiés sur 27 900. La médiane pré-enregistrée donne
`Href = 6547,530617 MPa`, donc `Hchi = 3273,765308 MPa` pour `alpha=0,5` et
`6547,530617 MPa` pour `alpha=1`.

**Smoke P154 :** `validation/nonlocal_p154_smoke_results.md`. Après
pré-enregistrement d'une norme mixte \(L_\infty\) indépendante du maillage,
`alpha=0,5` converge en `406,28 s`, `alpha=1` en `503,04 s` et `alpha=2` en
`226,30 s`, sans aucun échec du point fixe. Les trois passent tous les
critères smoke ; le prolongement à `alpha=2` était autorisé parce que le
meilleur point se trouvait à la borne supérieure du balayage initial.

**Validation P154 :** `validation/nonlocal_p154_validation_results.md`. Les
trois candidats positifs convergent à 20 incréments, padding 128, sans cutback.
`alpha=2` est le meilleur point testé : `+0,1643` de corrélation, `42,17 %`
de réduction L2, `+0,0331` d'IoU top-10 et `+0,0722` d'IoU q90. Il échoue
seulement sur l'aire active q90 (`21,85 %` au lieu de `<=20 %`). Le seuil
n'est pas déplacé a posteriori et aucun transfert confirmatoire n'est lancé.

**Critère de sortie :** partiellement atteint. P154 padding 128 converge à 20
incréments pour trois `Hchi>0` et la voie 3D condensée reproduit la voie native
sur le cas réduit. Aucun candidat ne passe toutefois tous les critères
scientifiques ; `Hchi` ne peut donc pas être figé ni transféré sans nouveau
protocole prospectif.

### Phase prioritaire A.6 — ROI P43 et chemin constitutif léger

- [x] Conserver le classement morphologique automatisé comme outil de
      présélection, sans lui déléguer le choix scientifique
- [x] Retenir P43 `(4,3)` après inspection visuelle de ses deux bandes
      diagonales ; cœur `360×310`, `x=[1440,1800)`, `y=[930,1240)`
- [x] Séparer les essais MFront sans tangente, avec tangente, puis la
      complétion tensorielle 3D finale
- [x] Préallouer les buffers Kelvin, PEEQ et `chi` du point fixe
- [x] Précalculer la direction du prédicteur pour le chargement DIC
      proportionnel
- [x] Chronométrer MFront avec/sans tangente, Kelvin, tenseurs 3D, forces
      internes, matrices élémentaires, assemblage, extraction et PARDISO
- [x] Comparer les états constitutifs bit à bit sur un crop réel et sur P43
- [x] Comparer un solveur EF complet avant/après sur la même zone et avec les
      mêmes paramètres
- [x] Figer la structure CSR libre-libre et mettre à jour uniquement `data`
- [x] Piloter explicitement PARDISO : phase 11 unique, puis phases 22/33
- [x] Conserver et tester le chemin générique `mtype=11`
- [x] Activer le CSR triangulaire et `mtype=2` uniquement pour le J2 vérifié
- [x] Rejeter tout tangent J2 dont l'asymétrie relative dépasse `1e-12`
- [x] Lancer la référence locale P43 avec le profil scientifique retenu
- [x] Estimer `Href` sur le cœur P43, puis pré-enregistrer le balayage
      `alpha=0,1,2,4`
- [x] Lancer et visualiser les campagnes P43 sans ajustement rétroactif

**Preuve :** commit `d5b0e7e` et
`validation/performance/nonlocal_hot_path_optimization.json`. Sur P43, le
benchmark constitutif passe de `14,357 s` à `7,605 s` et de `796 856` à
`564 508 KiB`, avec quatre empreintes de champs identiques. Sur le gate EF
P187, le temps processus passe de `396,78 s` à `273,56 s` et le pic RSS baisse
de `12,7 %`. Les deux versions conservent exactement 20 tentatives,
13 incréments acceptés, 7 cutbacks, 156 Newton et 623 itérations non locales.
Les écarts des champs physiques restent inférieurs à `1,1e-12` relativement à
leur amplitude globale.

**Campagne P43 :** `validation/nonlocal_p0043_validation_results.md`. La
référence locale et les candidats `alpha=1,2,4` convergent tous à 20
incréments sans cutback. La corrélation EVM passe de `0,3791` à
`0,4624/0,4814/0,5036` et l'erreur L2 relative de `0,9516` à
`0,6174/0,5256/0,4341`. `alpha=2` maximise légèrement l'IoU top-10 tandis que
`alpha=4` maximise corrélation et IoU q90 : les deux restent non dominés et
aucun `Hchi` n'est figé.

**Interprétation détaillée :**
`docs/explanation/p43_coupled_results.md` commente séparément les cartes EVM,
les erreurs signées, les champs et distributions PEEQ, le coût numérique et
les conclusions temporaires. Le couplage réduit le pic PEEQ de `81,9 %`, son
RMS de gradient de `65,3 %` et sa variation totale de `56,0 %`, pour seulement
`9,3 %` de baisse de moyenne : il s'agit bien d'une redistribution. La baisse
du rappel q90 et le léger recul de l'IoU top-10 à `alpha=4` empêchent toutefois
de conclure que toute augmentation supplémentaire serait bénéfique.

**Critère de sortie :** atteint pour l'optimisation technique et l'exécution
du balayage P43. Cette phase ne modifie ni la loi, ni
`ell`, ni `Hchi`, ni les tolérances, ni Newton, ni le point fixe, ni la
tangente. Le second lot modifie seulement l'assemblage sparse et le cycle
PARDISO. Sur P187, il ajoute `-10,6 %` de temps processus et `-16,7 %` de pic
RSS par rapport au chemin constitutif déjà optimisé ; une phase 11 et 139
paires 22/33 sont enregistrées. Le troisième lot J2 symétrique réduit encore
le temps `244,67→227,34 s`, PARDISO de `38,0 %` et le pic RSS de `8,7 %`.
La plasticité cristalline reste par défaut sur le chemin complet `mtype=11`.

### Phase prioritaire A.7 — Identification conjointe rapide de `ell` et `Hchi`

**Mission :** déterminer si la longueur `ell` et le module de couplage `Hchi`
sont séparément identifiables et transférables, sans grille F2 exhaustive et
sans déclarer prématurément une longueur matérielle. Le domaine initial est
`alpha∈[1,6]`, `ell∈[20,60] µm`, avec le témoin local unique `alpha=0`.

**Contrats scientifiques :**

- conserver le modèle micromorphique, MFront, MGIS transactionnel, Newton et
  les deux voies de contraintes planes inchangés ;
- paramétrer et enregistrer `alpha`, `Hchi`, `ell` et
  `Achi=Hchi*ell**2`, avec interpolation possible dans
  `(log(Hchi), log(Achi))` ;
- comparer uniquement l'EVM totale reconstruite par le même opérateur de
  mesure DIC ; PEEQ reste un diagnostic interne ;
- distinguer explicitement F0 heuristique, F1 de classement et F2
  scientifique ;
- ne jamais réutiliser un cache dont les empreintes physiques, numériques ou
  d'observation diffèrent ;
- ne lancer aucun nouveau calcul F2 sans validation humaine explicite.

**Ordre imposé et suivi :**

- [x] Auditer les solveurs Helmholtz, métriques, validateurs, partitions,
      formats de campagne et commandes réutilisables
- [ ] Formaliser les unités et les conversions
      `(alpha,ell) <-> (Hchi,Achi)`, y compris le cas local canonique
- [ ] Formaliser et empreinter l'opérateur `M_DIC`
- [ ] Ajouter les métriques d'amplitude, localisation, spectre spatial et
      diagnostics PEEQ
- [ ] Implémenter le crible F0 sur PEEQ local figé et ses diagnostics
      énergétiques/spectraux
- [ ] Valider les tendances F0 contre les F2 P43 existants
      `alpha=0,1,2,4`, `ell=58,88 µm`
- [ ] Implémenter F1 avec réduction spatiale configurable, historique complet,
      reprise, cache strict et statuts individuels
- [ ] Valider le classement F1 contre les quatre points F2 P43 existants
- [ ] Implémenter le profil `Hchi*(ell)`, PCHIP/sécante contrôlée et la courbe
      de recherche principalement unidimensionnelle
- [ ] Construire le front de Pareto amplitude-localisation, le genou et les
      cartes `(ell,alpha)` / `(Hchi,Achi)`
- [ ] Générer un manifeste de cinq nouveaux calculs F2 au maximum, incluant
      obligatoirement `(ell=58,88 µm, alpha=6)`, sans les lancer
- [ ] Préparer une validation de transfert de trois couples au maximum sur une
      autre ROI, sans recalage
- [ ] Produire configuration, CSV consolidé, figures, rapport, documentation,
      tests, HTML et PDF

**Point d'arrêt obligatoire :** après génération du manifeste F2, présenter
pour chaque candidat `ell`, `alpha`, `Hchi`, `Achi`, justification, coût
estimé et métrique discriminée. Attendre une validation humaine avant toute
exécution haute fidélité. Un éventuel second lot de deux points au maximum
nécessitera également une commande explicite.

**État initial :** P43 fournit quatre F2 réutilisables à
`ell=58,88 µm`, `alpha=0,1,2,4`. `H_ref` doit toujours être lu dans
`HREF.json` ou les métadonnées de campagne ; la valeur numérique n'est jamais
codée en dur. La page `docs/explanation/p43_coupled_results.md` constitue le
diagnostic scientifique de départ.

**Audit d'architecture (terminé) :**

- F0 réutilise directement
  `postprocessing.helmholtz.helmholtz_filter_element_field`; une seule
  résolution DCT est effectuée par longueur, puis tous les `Hchi` réemploient
  le même écart `p-chi`.
- L'observable principale réutilise
  `workflows.nonlocality_diagnostic.reconstruct_historical_evm` et les
  métriques existantes de `postprocessing.metrics`.
- Les lectures de campagnes doivent reprendre les contrôles de manifeste,
  statut et empreinte de `coupled_nonlocal_validation`; les fonctions
  génériques seront extraites dans un module partagé au lieu d'être copiées.
- F1 ne constitue pas un nouveau solveur : les champs globaux sont réduits
  de façon déterministe, puis transmis au `PartitionWorkflow` existant, qui
  conserve MFront/MGIS, Newton, PARDISO, les cutbacks, les sorties atomiques
  et la reprise.
- Le cache d'identification complète, sans remplacer, les manifestes du
  `PartitionWorkflow`. Sa clé inclut les empreintes de maillage, DIC,
  paramètres locaux, historique, opérateur DIC, fidélité, paramètres
  micromorphiques et commit.
- La CLI est une sous-commande unique avec actions explicites. Seule l'action
  F1 peut lancer des calculs réduits ; la génération F2 écrit uniquement un
  manifeste et des commandes reproductibles.

### Phase différée B — Validation externe Abaqus

- [ ] Récupérer ou régénérer un petit `.inp` de référence
- [ ] Extraire les mêmes champs aux mêmes emplacements physiques
- [ ] Comparer automatiquement `U/S/E/PE/PEEQ/RF`
- [ ] Étendre la comparaison à plusieurs pseudo-temps si les ODB deviennent
      disponibles

Cette phase ne bloque pas la phase A ni les campagnes DIC/EF du cas d'étude.

### Semaine 1 — Contrat scientifique

- [x] Écrire les conventions d'axes `U/V`, `x/y`, axes NumPy 0/1
- [x] Définir les unités de toutes les entrées et sorties
- [x] Définir `epsilon_xy` tensoriel et `gamma_xy` ingénieur
- [x] Définir la formule de `epsilon_vM` sous contrainte plane
- [x] Définir les quatre courbes de contrainte-déformation
- [ ] Vérifier la section et l'épaisseur réellement utilisées dans Abaqus
- [x] Identifier et versionner les quatre tableaux disponibles du ROI
- [~] Établir un jeu de données réduit pour les tests rapides ; les données
  complètes sont versionnées par Git LFS
- [ ] Décider et documenter les tolérances avant comparaison finale

**Critère de sortie :** document scientifique relu et approuvé, sans convention
implicite.

### Semaines 2–3 — Préparation DIC et calcul autonome

- [x] Reproduire la table plastique Abaqus :
  - domaine `0 <= ep <= 0.2` ;
  - 1000 points ;
  - traitement documenté du premier incrément `1e-6`.
- [x] Tester séparément loi analytique et loi tabulée
- [x] Corriger le calcul et l'étiquetage de la contrainte EF directe
- [x] Corriger les conventions d'axes et de cisaillement
- [ ] Comparer contraintes et déformations au même emplacement physique
- [x] Définir la méthode commune de calcul des déformations depuis `U`
- [~] Vérifier `U1`, `U2`, `S11`, `S22`, `S12`, `PEEQ` : contrats internes
  couverts, comparaison Abaqus encore absente
- [x] Vérifier le signe et la définition des réactions
- [x] Ajouter des assertions sur PEEQ au test biaxial

**Critère de sortie :** sous-domaine réel préparé depuis les données brutes,
calculé par `fem_inhouse` et accompagné d'un manifeste complet. La parité Abaqus
est reportée à la phase B.

### Semaines 4–5 — Ingénierie logicielle

- [x] Créer un `pyproject.toml`
- [x] Verrouiller les dépendances et versions
- [x] Créer un paquet sous `src/fem_inhouse`
- [~] Séparer :
  - [x] maillage ;
  - [x] élément ;
  - [x] matériau constitutif ;
  - [x] assemblage ;
  - [x] solveur non linéaire ;
  - [x] résultats ;
  - [x] post-traitement.
- [x] Remplacer les 19 paramètres de `run_fem` par des configurations typées
- [x] Ajouter les validations d'entrée
- [x] Supprimer les effets de bord lors des imports
- [x] Supprimer les chemins absolus
- [x] Ajouter une CLI limitée au cas d'étude
- [x] Ajouter Ruff, Pyright ou mypy, pytest et couverture
- [x] Ajouter une journalisation structurée
- [x] Échouer explicitement si PyPardiso n'est pas disponible en production

**Critère de sortie :** installation fraîche et cas réduit exécutables par une
commande documentée.

### Semaines 6–8 — Partitionnement, padding et raccordement

- [x] Définir une grille déterministe de 25 partitions
- [x] Définir une grille déterministe de 100 partitions
- [x] Gérer correctement les partitions de bord et de coin
- [x] Extraire les cartes matériau et les déplacements locaux
- [x] Ajouter le padding configurable
- [x] Résoudre indépendamment chaque partition
- [x] Enregistrer uniquement les résultats nécessaires par partition
- [x] Extraire et raccorder les cœurs non recouverts
- [x] Garantir l'absence de trous, doublons et décalages d'indices
- [x] Permettre une reprise après interruption
- [x] Ajouter un manifeste et une empreinte des entrées
- [x] Produire des fichiers `.npy` mappés en mémoire pour le champ global
- [x] Rendre l'ordre d'exécution des partitions sans effet sur le résultat
- [x] Ajouter un modèle de job array pour le calcul parallèle

**Critère de sortie :** domaine réduit identique entre calcul monolithique et
calcul partitionné avec padding suffisant.

### Semaine 9 — Performance et ressources

- [~] Mesurer temps et mémoire pour 10k, 50k, 100k et 350k éléments
- [~] Mesurer séparément assemblage, factorisation, Newton et écriture
- [~] Vérifier le nombre de threads MKL
- [ ] Définir la taille maximale d'une partition pour la machine cible
- [x] Réserver le repli SciPy au diagnostic ; PyPardiso reste obligatoire
- [ ] Définir un budget mémoire et un budget de temps par partition
- [~] Vérifier l'absence de copies mémoire évitables : tenseurs constitutifs
  globaux supprimés, structures sparse encore à profiler à grande taille
- [x] Documenter la stratégie de parallélisation

**Critère de sortie :** dimensionnement documenté avant tout calcul sur
11,16 millions d'éléments.

### Semaines 10–11 — Validation hiérarchique

#### Niveau 1 : vérification mathématique

- [x] Partition de l'unité et dérivées des fonctions de forme
- [x] Jacobien positif
- [x] Patch test élastique
- [x] Trois modes rigides
- [x] Retour plastique uniaxial, biaxial et en cisaillement
- [x] Tangente par différences finies
- [x] Cas tabulé dans chaque segment et au-delà de `ep = 0.2`
- [x] Équilibre des réactions
- [x] Convergence en nombre d'incréments

#### Niveau 2 : parité Abaqus

- [ ] Campagne différée jusqu'à stabilisation du pipeline DIC autonome
- [ ] Comparaison sur 10×10 ou 20×20
- [ ] Comparaison sur un sous-domaine hétérogène représentatif
- [ ] Comparaison à plusieurs pseudo-temps
- [~] Rapport automatique avec seuils de succès : commande prête, références
  Abaqus/DIC encore absentes

#### Niveau 3 : partitionnement

- [x] Référence monolithique sur domaine réduit
- [x] Comparaison sans recouvrement
- [ ] Comparaison avec padding 50, 100, 150 et 200
- [ ] Étude du nombre de partitions
- [!] Calcul et convergence de la métrique BGE
- [x] Mesure spécifique des erreurs aux interfaces

#### Niveau 4 : reproduction scientifique

- [~] RMSE du déplacement `U2` : outil et champ final DIC disponibles,
  résultat EF global à produire
- [~] RMSE et MAE de `epsilon_vM` : outil et champ final DIC disponibles,
  résultat EF global à produire
- [~] Carte de différence signée : génération prête, préparation DIC à finaliser
- [ ] BGE
- [~] Corrélation spatiale des champs : outil et champ DIC disponibles
- [~] Recouvrement des zones de plus forte localisation : métrique testée,
  résultat EF global encore absent
- [ ] Quatre courbes de contrainte-déformation séparées
- [ ] Intervalles de confiance calculés selon la méthode documentée
- [ ] Comparaison 25 partitions / 100 partitions avec padding 150

**Critère de sortie :** rapport reproductible expliquant les accords, écarts et
artefacts de raccordement.

### Semaine 12 — Documentation et version de référence

- [x] README de démarrage rapide
- [x] Documentation Sphinx intégralement en anglais et structurée avec Diátaxis
- [x] Landing page Read the Docs orientant vers tutoriels, guides, référence et explications
- [x] Compilations HTML stricte et PDF disponibles
- [x] Figures scientifiques vectorielles SVG/PDF reproductibles
- [x] Vérification automatique de la documentation HTML et des figures dans la CI
- [x] Tutoriel complet du cas réduit
- [x] Documentation du modèle numérique
- [x] Documentation des conventions
- [x] Documentation du partitionnement
- [x] Documentation de la validation
- [x] Documentation des limites scientifiques
- [x] Commandes uniques `test`, `validate`, `example`
- [x] CI verte sur une installation fraîche
- [ ] Revue indépendante scientifique
- [ ] Revue indépendante logicielle
- [ ] Version figée `1.0.0-case-study`

## 9. Architecture cible

```text
fem_inhouse/
├── pyproject.toml
├── README.md
├── LICENSE                 # décision juridique encore ouverte
├── Claude.md
├── data/
│   └── raw/case_study/     # tableaux immuables suivis par Git LFS + manifeste
├── references/
│   └── legacy_abaqus/      # provenance, jamais importée par le paquet
├── src/fem_inhouse/
│   ├── data_preparation.py
│   ├── config.py
│   ├── core/
│   │   ├── mesh.py
│   │   ├── element.py
│   │   ├── constitutive.py
│   │   ├── assembly.py
│   │   ├── nonlinear.py
│   │   └── solver_legacy.py
│   ├── partitioning/
│   │   ├── layout.py
│   │   ├── overlap.py
│   │   ├── extract.py
│   │   └── stitch.py
│   ├── postprocessing/
│   │   ├── strain.py
│   │   ├── invariants.py
│   │   ├── stress_curves.py
│   │   └── metrics.py
│   ├── workflows/
│   │   ├── solve_partition.py
│   │   └── reconstruct_roi.py
│   ├── results.py
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── examples/case_study/
├── validation/
│   ├── abaqus_input/
│   ├── abaqus_extraction/
│   ├── reference_data/
│   └── reports/
└── docs/
```

Cette architecture doit rester limitée au cas d'étude. Aucun système générique
de plugins pour des éléments ou matériaux non prévus n'est demandé.

## 10. Critères de maturité 4/5

### Noyau numérique

- [x] Tous les tests mathématiques critiques du modèle supporté passent
- [x] Tangente cohérente vérifiée automatiquement
- [x] Convergence robuste sur cas homogène et hétérogène
- [x] Échec de convergence diagnostiqué sans résultat silencieusement invalide

### Validation scientifique

- [ ] Parité Abaqus démontrée sur petits cas, validation externe différée
- [ ] Métriques de l'article reproduites ou écarts expliqués
- [x] Contrainte directe séparée des reconstructions
- [ ] Artefacts de partition quantifiés
- [ ] Seuils définis avant lecture des résultats finaux

### Ingénierie logicielle

- [x] API publique typée
- [x] Au moins 85 % de couverture des lignes
- [x] Au moins 80 % de couverture des branches
- [x] Couverture dédiée de toutes les fonctions constitutives critiques
- [x] Aucun avertissement qualité non justifié
- [~] Revue de code obligatoire pour les formules numériques : procédure et
  modèle de PR ajoutés, protection de branche non activée

### Reproductibilité

- [x] Installation fraîche reproductible
- [x] Versions verrouillées
- [x] Données DIC et cartes locales brutes versionnées par Git LFS et identifiées
  par empreinte
- [x] Préparation brute → canonique automatisée, atomique et manifestée
- [x] Aucun chemin dépendant d'un poste personnel
- [x] Résultats accompagnés de leur configuration et version du code
- [x] Workflow reprenable partition par partition

### Performance

- [x] PyPardiso/MKL utilisé et vérifié
- [~] Temps et mémoire mesurés : 10k, 50k et 100k terminés ; 350k reporté
- [x] Cas de production compatible avec la machine cible : les partitions
  intérieures P48 et P42 de 402 600 éléments ont convergé en 22 min 16 s et
  24 min 45 s, avec respectivement 7,51 GiB et 7,71 GiB de pic RSS
- [x] Traitement hors mémoire du ROI complet
- [~] Absence de régression de performance supérieure au seuil défini :
  comparaison A/B disponible, seuil global encore à ratifier

### Documentation

- [x] Une personne externe peut installer et exécuter le cas réduit
- [ ] Une personne externe peut reproduire les figures principales
- [x] Les hypothèses et limites sont visibles
- [x] Les descripteurs locaux ne sont pas présentés comme propriétés de grains

### Évaluation provisoire au 2026-07-25

| Axe | Note | Justification principale |
|---|---:|---|
| Noyau numérique | 4,5/5 | Cas fermés, tangente, cutback, réactions et cisaillement testés |
| Validation scientifique | 4,0/5 | Sélection P48 et confirmation P42 pré-enregistrées et archivées ; la même longueur améliore amplitude et localisation, mais le ROI raccordé et la comparaison Abaqus globale manquent |
| Ingénierie logicielle | 4,5/5 | API typée, modules séparés, CI, 230 tests, revue documentée |
| Reproductibilité | 4,5/5 | Données LFS, préparation atomique, manifestes et smoke test DIC réel |
| Performance | 4,0/5 | 10k–100k mesurés ; deux partitions intérieures de 402,6k exécutées avec temps et mémoire archivés |
| Documentation | 4,5/5 | Site Sphinx anglais structuré avec Diátaxis, PDF, contrats, API et figures vectorielles reproductibles ; revue externe encore requise |

Les notes ne doivent pas être relevées artificiellement par des cas
synthétiques. Pour atteindre 4/5 partout, les chemins critiques sont désormais :

1. terminer et vérifier le pipeline autonome depuis les données DIC versionnées ;
2. sélectionner une longueur diagnostique sur une partition, la tester sans
   ajustement sur des partitions tenues à l'écart, puis raccorder les champs
   du calcul réel et reproduire les métriques expérimentales accessibles ;
3. réserver une fenêtre machine permettant les mesures 350k et le
   dimensionnement d'une partition de production ;
4. réaliser ensuite la validation externe Abaqus lorsqu'une référence
   exploitable sera disponible.

## 11. Seuils de validation à ratifier

Ces valeurs sont des propositions initiales. Elles doivent être approuvées avant
la validation finale et adaptées aux conventions exactes d'extraction Abaqus.

- Déplacements, erreur L2 relative : cible `< 0,5 %`
- Contrainte de von Mises, petit cas Abaqus : cible `< 2 %`
- PEEQ, petit cas Abaqus : cible `< 5 %`
- Réaction globale : cible `< 1 %`
- Variation entre deux raffinements d'incréments : cible `< 0,5 %`
- Tangente par différences finies : cible `< 1e-6` en erreur relative

Valeurs scientifiques rapportées dans l'article à utiliser comme références :

- RMSE `U2` : `1,32 × 10^-2 %`
- RMSE `epsilon_vM`, 25 partitions directes : `0,361 %`
- MAE `epsilon_vM`, 25 partitions directes : `0,039 %`
- RMSE `epsilon_vM`, 100 partitions avec padding 150 : `0,220 %`
- MAE `epsilon_vM`, 100 partitions avec padding 150 : `0,156 %`

Ces métriques ne doivent pas être interprétées isolément. Une amélioration de
RMSE peut accompagner une carte visuellement plus bruitée aux interfaces.

## 12. Décisions ouvertes

| Décision | Statut | Responsable | Échéance |
|---|---|---|---|
| Fichiers Abaqus exacts de référence | Différé, non bloquant | À définir | Phase B |
| Épaisseur de section EF utilisée dans Abaqus | Différé, non bloquant | À définir | Phase B |
| Convention définitive U/V et x/y | Résolu dans `docs/scientific_contract.md` | Projet | S1 |
| Complétion nodale du bord supérieur | À ratifier ; profil initial explicite `edge-pad` | Revue scientifique | Phase A |
| Facteur de carte d'écrouissage | `380 MPa` nominal, `396 MPa` historique | Revue scientifique | Phase A |
| Traitement des neuf NaN d'écrouissage | Politique explicite à enregistrer | Revue scientifique | Phase A |
| Format des données globales hors mémoire | Résolu : `.npy` memmap | Projet | S4 |
| Machine cible et budget mémoire | Ouvert | À définir | S9 |
| Seuils finaux de parité Abaqus | Différé | Revue scientifique | Phase B |
| Schéma de production 25 ou 100 partitions | Ouvert | Revue scientifique | S11 |
| Métrique de localisation complémentaire | Ouvert | Revue scientifique | S10 |
| Licence du logiciel avant publication | Ouvert | Propriétaire du projet | S12 |

## 13. Registre des validations

| Date | Validation | Commande ou rapport | Résultat | Statut |
|---|---|---|---|---|
| 2026-07-24 | Backend PyPardiso | Import et résolution sparse 2×2 | Backend MKL actif | Réussi |
| 2026-07-24 | Test biaxial 20×20 | `.venv/bin/python fem_pixel.py` | 0 % erreur SVM | Réussi |
| 2026-07-24 | Tangente constitutive | Différences finies | `1e-10` à `7e-9` | Réussi |
| 2026-07-24 | Cas hétérogène | 6×6, quatre incréments | 4 NR/incrément | Réussi |
| 2026-07-24 | Scripts complets | Imports des scripts | `test_config.py` absent | Bloqué |
| 2026-07-24 | Socle de paquet | `pytest --cov=fem_inhouse` | 44 tests, 100 % | Réussi |
| 2026-07-24 | Qualité du nouveau code | `ruff check src tests` | Aucun défaut | Réussi |
| 2026-07-24 | Partitionnement et raccordement | `pytest --cov=fem_inhouse --cov-branch` | 62 tests, 98 % | Réussi |
| 2026-07-24 | Grilles de l'article | Tests `(5,5)` et `(10,10)`, padding 150 | 25/100 cœurs sans trou | Réussi |
| 2026-07-24 | API solveur et noyau EF | `pytest --cov=fem_inhouse --cov-branch` | 82 tests, 94 %, sans avertissement | Réussi |
| 2026-07-24 | Tangente cohérente automatisée | Différences finies du retour plastique | Erreur relative `< 1e-5` | Réussi |
| 2026-07-24 | Compatibilité historique | `.venv/bin/python fem_pixel.py` via pytest | Biaxial SVM/PEEQ réussi | Réussi |
| 2026-07-24 | Workflow reprenable | Tests manifestes, corruption, reprise, raccordement | 87 tests, 95 % | Réussi |
| 2026-07-24 | Exemple réduit tabulé 4×4 | `python -m fem_inhouse validate --nx 4 --ny 4` | erreur SVM `5,84e-6`, PEEQ `2,17e-6` | Réussi |
| 2026-07-24 | Suite complète et seuil CI | `pytest --cov=fem_inhouse --cov-branch` | 92 tests, 95,04 % | Réussi |
| 2026-07-24 | Construction du paquet | `pip wheel . --no-deps` et inspection | cœur, workflow et CLI présents | Réussi |
| 2026-07-24 | Portabilité des scripts historiques | Contrat `.npy`, chemins par environnement | 97 tests, 95,20 %, aucun chemin personnel | Réussi |
| 2026-07-24 | CI sur installation fraîche | GitHub Actions `30086978438` | installation, Ruff et tests verts | Réussi |
| 2026-07-24 | Parité monolithique/partitionnée | Cas homogène 6×6, padding 0 et 1 | `U/S/E/PEEQ` égaux aux tolérances | Réussi |
| 2026-07-24 | Métriques de champs/interfaces | `pytest --cov=fem_inhouse --cov-branch` | 104 tests, 95,36 % | Réussi |
| 2026-07-24 | Robustesse constitutive/globale | 3 trajets plastiques, hétérogène, cutback, réactions | 111 tests, 96,14 % | Réussi |
| 2026-07-24 | Performance 10k/50k/100k | `/usr/bin/time -v fem-inhouse validate` | 5,01/10,60/21,87 s ; 163/557/1061 MiB | Réussi |
| 2026-07-24 | Performance 350k | Vérification mémoire avant lancement | 3,7 GiB disponibles, swap saturé | Reporté |
| 2026-07-24 | Modules maillage/élément/assemblage | Suite complète après extraction | 117 tests, 96,26 % | Réussi |
| 2026-07-24 | Module constitutif public | Suite complète après extraction | 123 tests, 96,32 % | Réussi |
| 2026-07-24 | Module solveur non linéaire | Suite complète et compatibilité historique | 123 tests, 96,33 % | Réussi |
| 2026-07-24 | Diagnostics structurés | Événements `logging` et rapport JSON | Convergence et cutbacks traçables | Réussi |
| 2026-07-24 | Suite après diagnostics | `pytest --cov=fem_inhouse --cov-branch` | 123 tests, 96,66 % | Réussi |
| 2026-07-24 | Typage statique | `mypy src/fem_inhouse` | 25 fichiers, aucun défaut | Réussi |
| 2026-07-24 | Profil par phase 10k hétérogène | `SolverDiagnostics` | 31,948 s, 78 Newton, 0 cutback | Réussi |
| 2026-07-24 | Suite après instrumentation | `pytest --cov=fem_inhouse --cov-branch` | 123 tests, 96,59 % | Réussi |
| 2026-07-24 | Réactions du patch affine | Sommes sur les quatre bords | Signes et résultantes analytiques | Réussi |
| 2026-07-24 | Patch affine en cisaillement | Solution fermée | `U1/U2/E12/S12/PEEQ` conformes | Réussi |
| 2026-07-24 | Suite après patch cisaillement | `pytest --cov=fem_inhouse --cov-branch` | 124 tests, 96,59 % | Réussi |
| 2026-07-24 | CLI partitionnée | Reprise, partition isolée, raccordement | Workflow job array exécutable | Réussi |
| 2026-07-24 | Suite après CLI partitionnée | Ruff, mypy, pytest et `bash -n` | 125 tests, 96,46 % | Réussi |
| 2026-07-24 | Qualité dépôt complet | `ruff check .` | Aucun défaut, scripts historiques inclus | Réussi |
| 2026-07-24 | Wheel typé | `pip wheel . --no-deps` et inspection | `py.typed`, cœur et métadonnées présents | Réussi |
| 2026-07-24 | Recouvrement des localisations | Jaccard, Dice, rappel, précision | Cas identique, partiel et masqué testés | Réussi |
| 2026-07-24 | Suite après métrique de localisation | Ruff, mypy et couverture | 127 tests, 96,55 % | Réussi |
| 2026-07-24 | Provenance de l'article | Manifeste SHA-256 vérifié par test | PDF 2 698 182 octets identifié | Réussi |
| 2026-07-24 | Rapport de comparaison | CLI à seuils pré-déclarés | JSON, carte signée et code retour testés | Réussi |
| 2026-07-24 | Suite après rapport automatique | Ruff, mypy et couverture | 135 tests, 96,70 % | Réussi |
| 2026-07-24 | Assemblage tangent par blocs | A/B hétérogène 10k | -22,4 % tangent, -3,2 % RSS | Réussi |
| 2026-07-24 | Suite après optimisation mémoire | Ruff, mypy et couverture | 143 tests, 96,93 % | Réussi |
| 2026-07-24 | Gouvernance technique | ADR, guide et modèle de PR | Règles numériques explicites | Réussi |
| 2026-07-24 | CI complète distante | GitHub Actions `30089878592` | lint, mypy, wheel et tests verts | Réussi |
| 2026-07-24 | Inventaire scientifique reçu | Formes, statistiques, SHA-256 et scripts de provenance | 4 tableaux `3600×3100` identifiés | Réussi |
| 2026-07-24 | Sous-domaine DIC réel | Centre 10×10, PyPardiso, 10 incréments | Tous champs finis, 0 cutback | Réussi |
| 2026-07-24 | Données scientifiques versionnées | Git LFS + `data/raw/case_study/manifest.json` | 4 tableaux bruts immuables | Réussi |
| 2026-07-24 | Préparation ROI complet | `fem-inhouse prepare-case --nonfinite-policy nearest` | 4 champs canoniques, manifestés, 9 réparations | Réussi |
| 2026-07-24 | Idempotence de préparation | Deuxième exécution sur les mêmes sorties | Empreintes vérifiées, aucune réécriture | Réussi |
| 2026-07-24 | Chaîne DIC réelle 10×10 | Préparation centrale, 25 partitions, raccordement | `U/S/E/PEEQ` finis et complets | Réussi |
| 2026-07-24 | Suite après pipeline DIC | Ruff, mypy, pytest avec branches | 156 tests, 95,26 % | Réussi |
| 2026-07-24 | Clone distant avec Git LFS | Clone isolé, `git lfs pull`, SHA-256, crop 4×4 | Données récupérées et préparées depuis GitHub | Réussi |
| 2026-07-24 | CI distante du pipeline DIC | GitHub Actions `30091651001` | Ruff, mypy, wheel et tests verts | Réussi |
| 2026-07-24 | Sauvegarde exhaustive des partitions | Tests CLI et reprise | `U/S/E/PE/PEEQ/RF` atomiques et empreintés | Réussi |
| 2026-07-24 | Partition article DIC réelle | 100 partitions, padding 150, partition 0 (`510×460`) | 20/20 incréments, 0 cutback, 18 min 08 s, 3,59 GiB RSS | Réussi |
| 2026-07-24 | Intégrité partition article | `validation-report.json`, SHA-256 et contrôles mécaniques | 6 champs finis, bords DIC à `4,16e-17 mm`, équilibre `4,39e-14` | Réussi |
| 2026-07-24 | Comparaison exploratoire DIC/EF | `epsilon_vM` sur la zone résolue | RMSE `0,253 %`, MAE `0,185 %`, corrélation `0,016` | À approfondir |
| 2026-07-24 | MFront nominal sur partition article | Même partition `510×460`, loi analytique non capée, 8 threads MGIS | 20/20 incréments, 10 min 50,08 s, 4 163 308 KiB RSS | Réussi |
| 2026-07-24 | Comparaison longue MFront/tabulé | Champs, temps et mémoire sauvegardés | -40,35 % mur, constitutif 6,905× plus rapide, RSS +10,49 % | Réussi |
| 2026-07-24 | Installation TFEL/MFront et MGIS | Versions et imports depuis `.venv` | TFEL 5.1.0, MGIS 3.1, interface générique active | Réussi |
| 2026-07-24 | Parité constitutive Python/MFront | `validation/reference_data/mfront_material_point_v1/report.json` | L2 contrainte `0,227–0,368 %`, erreur PEEQ max `<3,88e-5` | Réussi |
| 2026-07-24 | Suite après backend MFront | Ruff, mypy, compilation MFront et couverture | 165 tests, 94,25 %, dont 2 tests MGIS réels | Réussi |
| 2026-07-24 | Performance constitutive Python/MFront | 200k points, 20 incréments, 2 répétitions | Python 12,347 s ; MFront série 13,333 s ; MFront 8 threads 3,527 s | Réussi |
| 2026-07-24 | Reproductibilité MFront parallèle | États série/parallèle sur 4 millions de mises à jour | Écarts max contrainte et PEEQ strictement nuls | Réussi |
| 2026-07-24 | Suite après pool MGIS | Ruff, mypy et couverture avec bibliothèque réelle | 167 tests, 94,21 % | Réussi |
| 2026-07-24 | Couplage MFront/Newton | Cas biaxial homogène complet | Parité champs `4,4e-11–1,2e-10`, 0 cutback | Réussi |
| 2026-07-24 | Parité MFront/Python sur DIC réelle | Crop central 10×10, 6 champs sauvegardés | L∞ relatif `4,7e-9–3,3e-4`, 20/20 incréments, 0 cutback | Réussi |
| 2026-07-24 | Performance EF complète MFront | Crop central 10×10, PyPardiso | 0,669 s et 66 Newton contre 1,583 s et 84 Newton | Réussi |
| 2026-07-24 | Suite après couplage Newton | Ruff, mypy, MGIS réel | 172 tests | Réussi |
| 2026-07-25 | Parité MFront natif/J2 3D condensé | Crop DIC 10×10, 20 incréments | 66 Newton chacun, contrainte max `4,804e-08 MPa`, 0 échec local | Réussi |
| 2026-07-25 | Résolution locale de contraintes planes | Résidu GP, itérations et `Cbb` | `2,705e-08 MPa`, 4 itérations max, `cond(Cbb)=1,896` | Réussi |
| 2026-07-25 | Suite architecture constitutive commune | Ruff, mypy et MGIS/MFront réel | 206 tests | Réussi |
| 2026-07-25 | Performance EF des trois backends | Crop DIC 100×100, 20 incréments, 3 répétitions | Python `134,36 s / 248,96 MiB` ; natif `27,03 s / 269,65 MiB` ; condensé `83,43 s / 320,30 MiB` | Réussi |
| 2026-07-25 | Équivalence des trois backends à échelle 100×100 | Comparaison des champs complets sauvegardés | MFront/MFront `2,307e-07 MPa` ; Python/MFront `6,763e-02 MPa`, tous seuils réussis | Réussi |
| 2026-07-25 | Filtre Helmholtz élémentaire | DCT, référence sparse et invariants | 12 tests dédiés, résidu `< 1e-11` | Réussi |
| 2026-07-25 | Workflow de diagnostic non local | Cas synthétique, padding, seuils et non-régression `ell=0` | Sélection cohérente et sorties atomiques | Réussi |
| 2026-07-25 | Campagne Helmholtz partition article 0 | `0–58,88 µm`, cœur `360×310`, padding 150 | RMSE/L2 `-49,45 %`, hypothèse partiellement soutenue | Réussi |
| 2026-07-25 | Suite après diagnostic Helmholtz | Ruff, mypy et MGIS/MFront réel | 230 tests | Réussi |
| 2026-07-25 | Calcul MFront partition de sélection P48 | 402 600 éléments, padding 150, 20 incréments | `1335,97 s`, `7 869 356 KiB`, zéro cutback | Réussi |
| 2026-07-25 | Sélection Helmholtz P48 | Balayage pré-enregistré `0–58,88 µm` | Corrélation `0,2983→0,6160`, IoU top-10 `0,1598→0,2822` | Réussi |
| 2026-07-25 | Calcul MFront confirmation P42 | 402 600 éléments, padding 150, 20 incréments | `1484,55 s`, `8 079 896 KiB`, zéro cutback | Réussi |
| 2026-07-25 | Confirmation Helmholtz tenue à l'écart P42 | `ell=58,88 µm` sans ajustement, seuils pré-déclarés | Corrélation `0,4007→0,7036`, tous critères réussis | Réussi |
| 2026-07-25 | Comportements MFront micromorphiques | Natif PlaneStress et Tridimensional, `Hchi*(p-chi)` | Compilation, métadonnées, signe, tangente et transactions | Réussi |
| 2026-07-25 | Couplage `p ↔ chi` dans Newton | DCT existante, relaxation, commit unique, cutback conjoint | 247 tests avec MGIS réel | Réussi |
| 2026-07-25 | Outil de sélection `Href` | Médiane du tangent de Ludwik sur le cœur plastifié | Tests synthétiques, empreintes et refus d'écrasement | Réussi |
| 2026-07-25 | Référence locale P154 padding 128 | 179 196 éléments, 20 incréments | `793,98 s`, 119 Newton, zéro cutback | Réussi |
| 2026-07-25 | Estimation `Href` sur le cœur P154 | 24 507 éléments plastifiés sur 27 900 | `Href=6547,530617 MPa` | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=0,5` | 87 164 éléments, norme mixte L∞ | `406,28 s`, 3 cutbacks, tous critères smoke réussis | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=1` | 87 164 éléments, norme mixte L∞ | `503,04 s`, 2 cutbacks, tous critères smoke réussis | Réussi |
| 2026-07-25 | Smoke micromorphique P154 `alpha=2` | 87 164 éléments, norme mixte L∞ | `226,30 s`, zéro cutback, tous critères smoke réussis | Réussi |
| 2026-07-25 | Validation P154 `alpha=0,5` | 179 196 éléments, padding 128, 20 incréments | `1453,77 s`, zéro cutback, 6/8 critères | Partiel |
| 2026-07-25 | Validation P154 `alpha=1` | 179 196 éléments, padding 128, 20 incréments | `1680,46 s`, zéro cutback, 6/8 critères | Partiel |
| 2026-07-25 | Validation P154 `alpha=2` | 179 196 éléments, padding 128, 20 incréments | `1867,20 s`, zéro cutback, 7/8 critères | Partiel |
| 2026-07-25 | Rejeu micromorphique natif/3D condensé | MFront réel, tangente FD et régression `Hchi=0` | 3 tests ciblés en `1,27 s` | Réussi |
| 2026-07-25 | Validation finale après campagne P154 | Ruff, mypy, MGIS/MFront réel, Sphinx strict | 257 tests en `14,38 s`, HTML et PDF 131 pages | Réussi |
| 2026-07-26 | Sélection scientifique P43 après classement morphologique | Cœur DIC `360×310`, deux bandes diagonales | P43 retenue, aucun calcul lourd lancé | Réussi |
| 2026-07-26 | Benchmark constitutif léger P43 | 446 400 points de Gauss, 14 itérations | `14,357→7,605 s`, RSS `-29,2 %`, champs identiques | Réussi |
| 2026-07-26 | Gate EF complet avant/après | P187 paddée, 39 644 éléments, paramètres identiques | `396,78→273,56 s`, RSS `-12,7 %`, convergence identique | Réussi |
| 2026-07-26 | Validation après optimisation | Ruff, mypy, 271 tests MGIS/MFront, Sphinx strict | HTML et PDF 144 pages | Réussi |
| 2026-07-26 | CSR fixe et phases PARDISO explicites | Même P187, chemin constitutif optimisé inchangé | `273,56→244,67 s`, RSS `-16,7 %`, une phase 11 et 139 phases 22/33 | Réussi |
| 2026-07-26 | J2 symétrique défini positif | Même P187, CSR supérieur et `mtype=2` | `244,67→227,34 s`, PARDISO `-38,0 %`, RSS `-8,7 %` | Réussi |
| 2026-07-26 | Balayage couplé P43 `alpha=1,2,4` | 402 600 éléments, 20 incréments, `ell=58,88 µm` | `26:40 / 29:56 / 36:14`, zéro cutback | Réussi |
| 2026-07-26 | Validation et figures P43 | EVM brute/DIC sur cœur, PEEQ interne | 8/8 critères pour les trois candidats ; `alpha=2` et `4` non dominés | Réussi |

## 14. Journal des mises à jour

### 2026-07-26 — P43 et optimisation du chemin chaud micromorphique

- P43 retenue comme prochaine ROI de calibration après inspection des bandes
- Évaluations intermédiaires limitées à PEEQ, sans tangent ni tenseurs 3D
- Une seule évaluation tangentielle par point fixe convergé et une seule
  reconstruction 3D par calcul FEM convergé
- Buffers réutilisables et prédicteur proportionnel préassemblé
- Chronométrages détaillés ajoutés jusqu'à PARDISO
- Équivalence constitutive bit à bit et équivalence EF sous `1e-10` validées
- Structure CSR libre-libre figée et buffers numériques mis à jour en place
- PARDISO piloté en phases 11/22/33 explicites
- J2 vérifié en CSR supérieur `mtype=2`; comportement inconnu en CSR complet
  `mtype=11`, notamment la future plasticité cristalline par défaut
- Contrôle runtime de l'asymétrie tangentielle sans symétrisation artificielle
- Gate P187 : `-10,6 %` de temps processus et `-16,7 %` de pic RSS
  supplémentaires, sans modifier Newton, le point fixe ou la tangente
- Gate symétrique P187 : `-7,1 %` de temps, `-38,0 %` dans PARDISO et
  `-8,7 %` de pic RSS supplémentaires
- Rapports reproductibles ajoutés sous `validation/performance/`

### 2026-07-24 — Documentation Sphinx anglaise avec Diátaxis

- Création d'une landing page Read the Docs présentant le but scientifique,
  le périmètre supporté, les limites et les résultats validés
- Organisation en quatre quadrants Diátaxis : tutoriel guidé, guides
  opératoires, référence des contrats et explications scientifiques
- Documentation détaillée de la chaîne DIC, de la loi J2/Ludwik analytique,
  de MFront/MGIS, de Newton, du partitionnement, des entrées et des sorties
- Génération reproductible de schémas vectoriels en paires SVG/PDF afin
  d'adapter automatiquement le format aux sorties HTML et LaTeX
- Compilation Sphinx stricte sans avertissement et production locale d'un PDF
  de 70 pages avec LuaLaTeX
- Configuration Read the Docs v2 pour publier `htmlzip` et PDF
- Ajout d'un job CI régénérant les figures, contrôlant leur stabilité et
  compilant le HTML avec les avertissements traités comme des erreurs

### 2026-07-24 — Loi MFront nominale et calcul long de l'article

- Passage des valeurs par défaut de la configuration, de l'API basse et de la
  CLI vers `constitutive_backend=mfront` et `hardening_mode=ludwik`
- Construction paresseuse de la table Python uniquement si le backend
  historique est demandé explicitement ; aucun tableau de 1000 points n'est
  créé sur le chemin nominal
- Conservation de la loi tabulée plafonnée à PEEQ `0,2` uniquement pour les
  régressions historiques et la future comparaison Abaqus
- Exécution complète de la partition de coin `510×460` avec les entrées DIC,
  PyPardiso, 20 incréments et huit threads MGIS
- Convergence 20/20 sans cutback en `648,402 s` solveur et `650,08 s` mur
  global, 112 itérations Newton, résidu relatif final `2,207e-8`
- Conservation des six champs, manifeste, logs, temps `/usr/bin/time -v`,
  empreintes, cartes dérivées, aperçu et rapport de comparaison reproductible
- Gain mur de `40,35 %` et gain constitutif de `6,905×` face à l'ancien calcul
  Python tabulé ; différences L2 relatives de `0,72–0,91 %` sur
  `E/PE/PEEQ/S`
- Pic RSS complet de `4 163 308 KiB`, supérieur de `10,49 %` à l'ancien run :
  la suppression de la table est effective, mais le stockage MGIS et le
  système EF sparse dominent la mesure globale
- PEEQ maximal `0,06496` : le plafond historique `0,2` n'aurait pas été atteint
  sur cette partition, mais il est désormais absent du modèle nominal
- Validation par 172 tests avec MGIS réel, 167 tests et 5 skips sans MGIS,
  Ruff, mypy, smoke CLI MFront et préflight de la partition

### 2026-07-24 — Premier backend constitutif MFront/MGIS

- Installation source de TFEL/MFront 5.1.0 et MGIS 3.1 sous
  `/home/jeff/.local`, avec commits et options CMake enregistrés
- Contournement documenté du suffixe de module Python TFEL 5.1.0 en désactivant
  uniquement `TFEL_APPEND_VERSION`, sans patcher les sources
- Ajout d'une loi J2/Ludwik en contrainte plane avec propriétés locales
  `InitialYieldStress`, `HardeningCoefficient` et `HardeningExponent`
- Ajout d'un adaptateur MGIS vectorisé avec conversions Kelvin, tangente
  cohérente et transactions explicites `evaluate/commit/revert`
- Sauvegarde de 200 incréments pour trois trajets dans un NPZ, avec rapport
  JSON, empreintes et figure
- Passage des seuils initiaux de contrainte et PEEQ ; écart de tangente
  `1,02–6,39 %` conservé comme diagnostic avant branchement dans Newton
- Validation complète par Ruff, mypy, recompilation MFront et 165 tests avec
  94,25 % de couverture ; les deux tests MGIS utilisent la bibliothèque réelle
- Maintien explicite du backend Python en production tant que la parité du
  sous-domaine DIC et la loi tabulée exacte ne sont pas validées

### 2026-07-24 — Benchmark constitutif d'une minute

- Ajout d'un pool de threads MGIS explicite et vérification de sa parité avec
  l'intégration série
- Construction d'un cas hétérogène de 200 000 points, 20 incréments et tangente
  cohérente à chaque mise à jour
- Deux répétitions avec inversion de l'ordre des backends pour limiter le biais
  thermique et de cache
- Temps médian Python `12,347 s`, MFront série `13,333 s` et MFront 8 threads
  `3,527 s`
- Gain MFront parallèle de `3,500×` sur Python et `3,780×` sur MFront série
- Durée complète `1 min 03,24 s`, pic RSS `393,45 MiB`, aucun swap
- Conservation des temps bruts, états finaux complets, échantillons de
  tangentes, empreintes, figure et mesure `/usr/bin/time -v`
- Validation complète par Ruff, mypy et 167 tests avec 94,21 % de couverture
- Limitation maintenue : benchmark du noyau constitutif uniquement, sans
  assemblage CPS4, Newton global ni PyPardiso

### 2026-07-24 — Couplage MFront dans Newton

- Ajout de la sélection `python|mfront` dans `SolverConfig`, l'API typée et la
  CLI de partitionnement, avec chemin de bibliothèque et pool MGIS configurés
- Intégration de la contrainte, des variables internes et de la tangente MFront
  aux points de Gauss dans la boucle Newton CPS4
- Garantie transactionnelle : chaque essai repart du dernier état convergé,
  `commit` uniquement après convergence globale et `revert` avant cutback
- Test homogène plastique de bout en bout avec parité Python/MFront de l'ordre
  de `1e-10`, sans cutback
- Campagne DIC réelle `10×10` sauvegardée avec les six champs de chaque
  backend, diagnostics, empreintes, seuils et rapport JSON
- Passage de tous les seuils : L∞ relatif maximal `3,26e-4`; 20 incréments et
  aucun cutback pour les deux backends
- Temps indicatifs sur le crop : Python `1,583 s`, 84 itérations ; MFront
  2 threads `0,669 s`, 66 itérations
- Validation par Ruff, mypy et 172 tests avec la bibliothèque MGIS réelle
- Maintien du backend Python par défaut jusqu'à décision sur la réplication
  exacte de la table Abaqus à 1000 segments et essai d'une partition article

### 2026-07-24 — Première partition à la taille de l'article

- Exécution de la partition de coin 0 sur la grille `10×10` de l'article avec
  padding 150, soit `510×460` éléments résolus et `360×310` éléments de cœur
- Conservation atomique des six champs finaux `U/S/E/PE/PEEQ/RF`, du manifeste,
  des journaux, de la consommation de ressources et de toutes les empreintes
- Convergence des 20 incréments sans cutback en `1088,13 s` solveur, avec
  113 itérations de Newton et un pic RSS processus de `3 768 132 KiB`
- Vérification des déplacements DIC prescrits à `4,16e-17 mm` et de l'équilibre
  global relatif des réactions à `4,39e-14`
- Archivage des cartes `epsilon_vM` DIC/EF, de leur différence, de `S_Mises` et
  d'une synthèse graphique
- Première comparaison exploratoire : RMSE `0,253` et MAE `0,185` points de
  pourcentage, proches en amplitude des `0,220/0,156` du ROI complet publié,
  mais corrélation spatiale faible (`0,016`) ; aucune revendication de parité
  avant raccordement du ROI et vérification des conventions exactes

### 2026-07-24 — Recentrage sur le calcul autonome depuis la DIC

- Déplacement de la comparaison Abaqus vers une phase de validation externe
  différée et non bloquante
- Adoption du pipeline prioritaire `raw DIC → préparation canonique → partitions
  → raccordement → post-traitement`
- Copie sans modification des quatre tableaux scientifiques dans
  `data/raw/case_study`
- Versionnement des grands tableaux par Git LFS
- Ajout d'un manifeste avec empreintes, formes, types, unités et ambiguïtés
- Conservation des deux générateurs historiques sous `references/legacy_abaqus`
- Exclusion du ZIP duplicatif et du HDF5 de plasticité cristalline, qui
  n'apportent aucune entrée supplémentaire au calcul ciblé
- Enregistrement explicite des trois décisions encore nécessaires : complétion
  nodale, facteur `K=380/396 MPa`, traitement des neuf valeurs non finies

### 2026-07-24 — Préparation canonique et smoke test DIC

- Ajout de `fem-inhouse prepare-case`
- Vérification en flux des tailles et empreintes SHA-256 brutes
- Conversion explicite `V → u_x`, `U → u_y` et pixel → millimètre
- Facteur `K=380 MPa` nominal et `396 MPa` historique sélectionnable
- Refus par défaut des valeurs non finies et politique `nearest` explicite
- Complétion nodale `edge-pad-upper` enregistrée dans le manifeste
- Écriture atomique hors du répertoire brut et réutilisation idempotente
- Ajout d'un crop central reproductible pour les contrôles rapides réels
- Préparation réussie du ROI complet en `3601×3101` nœuds et
  `3600×3100` éléments
- Calcul réussi du crop réel `10×10` en 25 partitions, puis raccordement de
  `U`, `S`, `E` et `PEEQ`
- Documentation du chemin complet depuis un clone neuf
- Clone distant isolé vérifié avec téléchargement des quatre objets Git LFS,
  empreintes identiques et préparation réussie d'un crop `4×4`
- CI GitHub verte sur le commit du pipeline autonome

### 2026-07-24 — Conservation exhaustive des calculs coûteux

- Extension des sorties persistantes à tous les champs finaux du solveur :
  `U`, `S`, `E`, `PE`, `PEEQ` et `RF`
- Écriture atomique et empreinte SHA-256 de chaque champ avant validation du
  statut de partition
- Conservation des diagnostics de convergence dans `status.json`
- Activation de Git LFS pour les résultats numériques de référence
- Validation par la suite complète : 156 tests, couverture 95,26 %

### 2026-07-24 — Création

- Audit initial du code existant
- Installation de l'environnement scientifique et de PyPardiso/MKL
- Vérifications élémentaires du noyau numérique
- Lecture de `ArticleSource/ArticleAdil.pdf`
- Recentrage du projet sur la reconstruction cinématique partitionnée
- Extension du planning de 9 à 12 semaines

### 2026-07-24 — Premier lot scientifique et logiciel

- Ajout du contrat scientifique exécutable et documenté
- Ajout de `pyproject.toml`, du paquet `src/fem_inhouse` et de pytest/Ruff
- Formalisation des configurations matériau, maillage et solveur
- Implémentation commune des déformations DIC/EF et de l'invariant plane-stress
- Séparation entre contrainte EF directe et reconstruction depuis la déformation
- Passage de la table historique de 50 à 1000 points
- Ajout d'une assertion PEEQ au test biaxial

### 2026-07-24 — Partitionnement déterministe

- Ajout des grilles équilibrées de 25 et 100 partitions du ROI complet
- Gestion explicite des cœurs, du padding et des bords du domaine
- Extraction locale des champs aux éléments et aux nœuds
- Raccordement à propriétaire unique, indépendant de l'ordre d'exécution
- Écriture du champ global au format `.npy` mappé en mémoire
- Ajout d'un manifeste JSON déterministe et de la documentation associée
- Validation par 62 tests avec 98 % de couverture lignes et branches combinées

### 2026-07-24 — API solveur et noyau testable

- Déplacement du noyau historique dans le paquet avec point d'entrée compatible
- Ajout de `CaseStudyConfig` comme API publique à la place des 19 paramètres
- Ajout de résultats typés et nommés, avec contrôle des valeurs non finies
- Validation des dimensions, cartes matériau, pseudo-temps et domaines physiques
- Échec explicite si PyPardiso/MKL est absent du calcul de production
- Alignement de la loi tabulée sur la grille `0`, `1e-6`, puis jusqu'à `0.2`
- Ajout des tests élémentaires, du patch affine, du retour plastique et de la
  tangente par différences finies
- Verrouillage exact de l'environnement Linux/Python 3.12
- Validation par 82 tests, 94 % de couverture totale et aucun avertissement

### 2026-07-24 — Workflow partitionné reprenable

- Résolution autonome de chaque zone de calcul paddée avec configuration locale
- Écriture atomique des seuls champs `U`, `S`, `E` et `PEEQ`
- Manifeste immuable avec empreintes des entrées, du code et de la configuration
- Reprise automatique avec détection des fichiers manquants ou corrompus
- Raccordement hors mémoire uniquement lorsque toutes les partitions sont valides
- Validation de la reprise et du raccordement par 87 tests, couverture 95 %

### 2026-07-24 — Exemple exécutable et intégration continue

- Ajout des commandes `backend`, `validate`, `example` et `layout`
- Ajout d'un cas équibiaxial réduit avec seuils déclarés avant exécution
- Sauvegarde de résultats auto-décrits et tutoriel de reproduction
- Construction et inspection réussies du wheel Python
- Ajout d'une CI GitHub avec environnement exact, Ruff et seuil de couverture 85 %
- Validation locale par 92 tests avec 95,04 % de couverture

### 2026-07-24 — Portabilité des données historiques

- Remplacement du `test_config.py` externe manquant par un contrat versionné
- Suppression des chemins Windows personnels dans les scripts conservés
- Configuration des données et résultats uniquement par variables d'environnement
- Validation des formes, valeurs finies et domaines des quatre champs d'entrée
- Documentation explicite des noms de fichiers `.npy` attendus
- Validation par 97 tests avec 95,20 % de couverture

### 2026-07-24 — Métriques et parité de partition

- Ajout de RMSE, MAE, erreur signée, L2 relative et corrélation spatiale
- Ajout d'un ratio de gradient spécifique aux interfaces de raccordement
- Parité vérifiée entre résolution monolithique et quatre partitions homogènes
- Comparaison vérifiée sans padding et avec padding d'un élément
- BGE exact maintenu bloqué : l'article ne donne pas la formule complète et le
  script d'analyse source n'est pas livré
- Validation complète par 104 tests avec 95,36 % de couverture

### 2026-07-24 — Robustesse du solveur

- Vérification des retours plastiques uniaxial, équibiaxial et en cisaillement
- Vérification de la saturation de la table plastique au-delà de `ep = 0.2`
- Stabilité vérifiée pour 5, 10 et 20 incréments
- Convergence vérifiée sur un damier hétérogène de paramètres
- Équilibre global des réactions intégré au seuil de l'exemple
- Échec de convergence forcé et diagnostiqué après réduction de pas
- Validation complète par 111 tests avec 96,14 % de couverture

### 2026-07-24 — Première campagne de performance

- Mesure homogène tabulée avec PyPardiso et 20 incréments
- 10k éléments : 5,01 s et 163 MiB
- 50 176 éléments : 10,60 s et 557 MiB
- 99 856 éléments : 21,87 s et 1,04 GiB
- Utilisation multithread observée entre 349 % et 552 % CPU
- Point 350k reporté pour éviter un OOM avec 3,7 GiB disponibles et swap saturé
- Protocole, limites et conditions de reprise documentés

### 2026-07-24 — Extraction du noyau EF

- Extraction du maillage rectangulaire structuré dans `core.mesh`
- Extraction du CPS4, de la quadrature et de l'élasticité dans `core.element`
- Extraction de l'assemblage sparse et des forces internes dans `core.assembly`
- Validation explicite des géométries, Jacobien, paramètres et formes matricielles
- Boucle Newton conservée comme dette isolée à l'issue de ce lot
- Validation complète par 117 tests avec 96,26 % de couverture

### 2026-07-24 — Extraction du modèle constitutif

- Extraction de l'invariant de von Mises plane-stress dans `core.constitutive`
- Extraction des écrouissages analytique et tabulé avec contrats d'entrée
- Extraction du retour plastique vectorisé et de la tangente cohérente
- Utilisation directe de ce module par le solveur et l'API publique du cœur
- Conservation de l'alias historique `_vm` dans `fem_pixel.py` uniquement
- Validation complète par 123 tests avec 96,32 % de couverture

### 2026-07-24 — Isolation du solveur non linéaire

- Déplacement de l'incrémentation, de Newton-Raphson et du cutback dans
  `core.nonlinear`
- Branchement direct de l'API publique typée sur ce module
- Réduction de `core.solver_legacy` à une couche de compatibilité historique
- Conservation du test de non-convergence par injection du solveur linéaire
- Validation complète par 123 tests avec 96,33 % de couverture

### 2026-07-24 — Diagnostics de convergence structurés

- Ajout de `SolverDiagnostics` au résultat public typé
- Enregistrement du backend, du temps, des incréments, cutbacks et itérations
- Enregistrement du résidu final et du critère de convergence réellement actif
- Émission d'événements `logging` structurés du début à la fin du calcul
- Inclusion des diagnostics dans le `report.json` de l'exemple reproductible
- Validation complète par 123 tests avec 96,66 % de couverture

### 2026-07-24 — Contrôle statique du paquet

- Ajout de mypy aux dépendances de développement verrouillées
- Correction des types des gradients, empreintes par blocs et emplacements
  de champs partitionnés
- Ajout du contrôle mypy à la CI après Ruff
- Validation sans défaut des 25 fichiers du paquet

### 2026-07-24 — Instrumentation des performances

- Chronométrage séparé de l'initialisation et de l'assemblage élastique
- Cumul des temps de retour constitutif, tangentes/assemblages et PyPardiso
- Chronométrage de la construction des sorties et de l'écriture des partitions
- Profil hétérogène 10k : 31,948 s, 78 itérations de Newton, aucun cutback
- Factorisation et substitutions encore regroupées par l'appel PyPardiso

### 2026-07-24 — Convention des réactions

- Définition explicite des réactions comme forces internes sur les DDL prescrits
- Vérification des signes sur les quatre bords du patch affine
- Vérification des résultantes analytiques horizontales et verticales
- Documentation des unités et de l'épaisseur implicite de 1 mm
- Maintien de l'épaisseur Abaqus exacte comme donnée externe encore absente

### 2026-07-24 — Patch test en cisaillement

- Ajout d'un champ affine de cisaillement simple sur maillage 4×3
- Vérification de la convention de cisaillement d'ingénieur `gamma12`
- Vérification de `S12 = G gamma12` et des composantes normales nulles
- Vérification simultanée de `U1`, `U2` et de PEEQ nulle

### 2026-07-24 — CLI partitionnée et job array

- Ajout d'une commande unique pour lister, résoudre, reprendre et raccorder
- Chargement mappé des quatre champs `.npy` et inférence de la taille du ROI
- Ajout du point d'entrée `--partition-id` adapté aux tâches indépendantes
- Ajout d'un modèle Slurm pour les grilles de 25 et 100 partitions
- Fichiers temporaires rendus uniques pour les écritures atomiques concurrentes
- Documentation du lancement, de la reprise et du raccordement hors mémoire

### 2026-07-24 — Qualité statique de tout le dépôt

- Extension de Ruff aux scripts historiques de comparaison et visualisation
- Formatage mécanique sans modification des formules scientifiques
- Exceptions limitées aux noms de variables scientifiques et imports de compatibilité
- Passage de la CI de chemins sélectionnés à `ruff check .`
- Validation locale : Ruff, mypy et 125 tests réussis

### 2026-07-24 — Distribution et citation

- Déclaration PEP 561 du paquet public avec `py.typed`
- Vérification du contenu du wheel construit localement
- Ajout de la construction du wheel à la CI
- Ajout de `CITATION.cff` depuis le titre et les auteurs de l'article source
- Licence laissée explicitement ouverte avant publication publique

### 2026-07-24 — Recouvrement des zones localisées

- Ajout d'une sélection indépendante par quantile supérieur
- Ajout des scores de Jaccard et Dice
- Ajout du rappel de la zone de référence et de la précision de la prédiction
- Conservation des seuils et effectifs pour interpréter les ex æquo
- Tests des recouvrements identique, partiel, masqué et des contrats invalides

### 2026-07-24 — Provenance de la référence scientifique

- Ajout d'un manifeste versionné pour le PDF fourni
- Enregistrement du titre, des auteurs, de la taille et du SHA-256
- DOI maintenu explicitement nul car absent du manuscrit fourni
- Ajout d'un test empêchant une modification silencieuse de la référence

### 2026-07-24 — Rapports automatiques de champs

- Ajout de seuils typés pour RMSE, MAE, corrélation et recouvrement
- Ajout d'une décision globale reproductible sans ajustement après calcul
- Ajout d'une carte signée `prédiction - référence` avec masque et NaN explicites
- Ajout de la commande `compare-fields` et d'un code retour exploitable en CI
- Documentation explicite de l'exigence de co-enregistrement préalable

### 2026-07-24 — Réduction mémoire de la tangente

- Suppression des tenseurs `C_ep` et `C B` matérialisés pour tous les points
- Assemblage par corrections plastiques sur la matrice élastique, par blocs
- Parité avec la formulation dense vérifiée à `rtol=1e-13`
- Réduction théorique de 1 568 à 800 octets globaux par élément
- Mesure A/B 10k : poste tangent -22,4 %, pic RSS processus -3,2 %

### 2026-07-24 — Décisions et revue numérique

- Ajout d'ADR sur le périmètre, PyPardiso et le raccordement des cœurs
- Ajout d'un guide de contribution limité au cas d'étude
- Exigence écrite d'un second relecteur pour toute formule numérique
- Ajout d'un modèle de PR avec preuves mathématiques et performance
- Protection de branche laissée inactive pour ne pas bloquer le dépôt mono-auteur

### 2026-07-24 — Actualisation des actions CI

- CI complète validée sur installation fraîche, wheel inclus
- Mise à jour de `actions/checkout` et `actions/setup-python` vers la version 7
- Suppression attendue de l'annotation de dépréciation Node.js 20

### 2026-07-24 — Reconstruction des tenseurs 3D en contraintes planes

- Ajout d'une couche vectorisée de post-traitement du seul état 2D convergé
- Conservation stricte du maillage CPS4, de Newton, du tangent et des sorties
  historiques
- Reconstruction analytique Python par élasticité plane-stress et
  incompressibilité plastique J2
- Extraction MFront native depuis `AxialStrain`, `ElasticStrain` et `Stress`
- Conservation du résidu numérique `S33` MFront sans remplacement artificiel
- Ajout des sorties `S_3D`, `E_3D`, `EE_3D`, `PE_3D` et
  `S33_RESIDUAL_MPA`
- Extension du résultat typé, des partitions, du raccordement et du chargeur
  de campagnes anciennes
- Ajout des invariants 3D et séparation explicite de `EVM_HISTORICAL`
  et `EVM_RECONSTRUCTED_3D`
- Validation des trajets proportionnels, du déchargement et du chargement non
  proportionnel
- Campagne DIC 10×10 sauvegardée avec non-régression des six anciens champs
- Documentation Diátaxis anglaise reconstruite sans erreur en HTML strict et
  en PDF de 78 pages
- Validation finale avec le vrai backend MGIS/MFront : 199 tests réussis ;
  Ruff sans défaut et mypy sans défaut sur le paquet et le script de campagne

### 2026-07-25 — Condensation d'une loi MFront 3D en contraintes planes

- Ajout d'un contrat commun `PlaneStressMaterialBatch` utilisé par le Newton
  global, sans connaissance de J2 ni des variables internes MGIS
- Compilation de la loi J2/Ludwik identique sous les hypothèses
  `PlaneStress` et `Tridimensional`
- Vérification de l'ordre Kelvin 3D MGIS `[11,22,33,12,13,23]` par
  métadonnées et six essais élastiques indépendants
- Résolution locale transactionnelle de
  `[epsilon33,gamma13,gamma23]` et condensation de la tangente par complément
  de Schur
- Ajout du résidu vectoriel `[S33,S13,S23]` et des diagnostics locaux au point
  de Gauss, sans rupture du champ historique `S33_RESIDUAL_MPA`
- Suppression du fallback J2 implicite pour un comportement MFront ne
  déclarant pas la capacité correspondante
- Tests de parité sur trajets matériels, tangent par différences finies,
  échec local sans pollution d'état, maillage homogène 4×4 et DIC 10×10
- Campagne immuable
  `validation/reference_data/mfront_3d_condensed_dic_10x10_v1` sauvegardée
- Validation avec MGIS/MFront réel : 206 tests réussis ; Ruff, mypy et
  contrôle des différences Git sans défaut

### 2026-07-25 — Benchmark EF comparatif des trois backends

- Ajout d'un pilote reproductible exécutant `python`,
  `mfront-native-plane-stress` et `mfront-3d-condensed-plane-stress` dans des
  processus frais et un ordre alterné
- Mesure du temps mur complet par GNU `time`, du temps solveur, du temps
  constitutif et du pic RSS sur le même crop DIC central 100×100
- Trois répétitions par backend, 20 incréments, deux threads MKL et deux
  threads MGIS ; neuf convergences sans cutback ni échec local
- Sauvegarde systématique de tous les champs, diagnostics, journaux,
  mesures de ressources, configuration et rapport agrégé
- Médianes temps mur / RSS : Python `134,36 s / 248,96 MiB`, MFront natif
  `27,03 s / 269,65 MiB`, MFront 3D condensé
  `83,43 s / 320,30 MiB`
- Confirmation que MFront natif et condensé sont équivalents à la précision
  numérique et que Python respecte les tolérances scientifiques déclarées

### 2026-07-25 — Diagnostic de non-localité par filtre de Helmholtz

- Ajout d'un filtre DCT élémentaire à flux nul, sans matrice globale ni
  modification du solveur mécanique
- Ajout des invariants numériques, de la comparaison sparse directe et d'une
  non-régression exacte pour `ell=0`
- Ajout d'une campagne CLI atomique retrouvant domaine résolu, cœur et padding
  depuis les métadonnées
- Reconstruction commune d'EVM depuis les déplacements DIC et FEM, puis
  filtrage du seul champ FEM
- Séparation stricte de PEEQ et d'EVM DIC dans les métriques d'amplitude
- Ajout des métriques de diffusivité, quantiles égaux, seuils absolus DIC,
  sélection Pareto et modes exploratoire/confirmatoire
- Sauvegarde de tous les champs, rapports, tableaux, figures et empreintes de
  la campagne réelle sur la partition article 0
- Diminution de `49,45 %` de RMSE et L2 relative à `58,88 µm`, mais
  corrélation finale `0,0926` et pics trop atténués
- Conclusion limitée à « hypothèse de largeur spatiale partiellement soutenue
  sur cette partition exploratoire », sans identification de longueur matériau
- Validation complète avec MGIS/MFront réel : 230 tests réussis ; Ruff et mypy
  sans défaut

### 2026-07-25 — Sélection P48 et confirmation tenue à l'écart P42

- Pré-enregistrement de P48 comme unique partition de sélection avant calcul
- Convergence de P48 sur 402 600 éléments en 22 min 16 s de processus, zéro
  cutback et 7,51 GiB de RSS maximal
- Sélection commune de `58,88 µm` par corrélation, IoU top-10 et seuils
  absolus DIC
- Réduction de `64,61 %` de RMSE/L2 sur P48 et corrélation finale `0,6160`
- Pré-enregistrement de P42 comme partition tenue à l'écart avec longueur et
  seuils figés
- Conservation d'une première tentative P42 interrompue par un SIGTERM externe
  avant toute écriture partielle
- Relance P42 inchangée dans une unité utilisateur isolée, puis convergence en
  24 min 45 s, zéro cutback et 7,71 GiB de RSS maximal
- Réussite de tous les seuils confirmatoires sur P42 : corrélation finale
  `0,7036`, réduction L2 `65,43 %`, gains d'IoU quantile et absolu
- Conclusion d'étape 1 relevée à « hypothèse de largeur spatiale soutenue »,
  sans interprétation de `58,88 µm` comme longueur interne matérielle
- Sauvegarde exhaustive des deux calculs mécaniques, champs filtrés, figures,
  journaux, rapports, seuils et empreintes
