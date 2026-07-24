# Plan de mise à niveau de `fem_inhouse`

Dernière mise à jour : 2026-07-24
Statut global : **pipeline autonome DIC → entrées canoniques → calcul
partitionné validé sur une partition article de 234 600 éléments ; backend
MFront/MGIS validé au point matériel mais pas encore branché dans Newton ;
exécution et raccordement des 100 partitions du ROI complet à planifier**
Objectif de maturité : **au moins 4/5 sur tous les axes**

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
- [ ] Décider entre loi MFront analytique régularisée et réplication exacte des
      1000 segments tabulés jusqu'à `PEEQ=0.2`
- [ ] Brancher MFront derrière une sélection de backend dans la boucle Newton
- [ ] Vérifier la tangente MFront dans les conventions d'assemblage CPS4
- [ ] Comparer les deux backends sur le crop DIC réel `10×10`
- [~] Mesurer coût, mémoire et stratégie de traitement par blocs aux points de
      Gauss avant tout calcul de taille article ; noyau constitutif mesuré sur
      200 000 points, branchement EF et taille de bloc encore à faire
- [ ] Basculer le backend par défaut seulement après parité du sous-domaine

**Critère de sortie :** le même sous-domaine DIC converge avec les deux
backends, les six champs sauvegardés respectent des seuils ratifiés, et aucun
état MFront d'une itération Newton rejetée n'est commis.

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
- [~] Cas de production compatible avec la machine cible : une partition de
  coin de 234 600 éléments a convergé en 18 min 08 s avec 3,59 GiB de pic RSS ;
  les partitions intérieures plus grandes restent à dimensionner
- [x] Traitement hors mémoire du ROI complet
- [~] Absence de régression de performance supérieure au seuil défini :
  comparaison A/B disponible, seuil global encore à ratifier

### Documentation

- [x] Une personne externe peut installer et exécuter le cas réduit
- [ ] Une personne externe peut reproduire les figures principales
- [x] Les hypothèses et limites sont visibles
- [x] Les descripteurs locaux ne sont pas présentés comme propriétés de grains

### Évaluation provisoire au 2026-07-24

| Axe | Note | Justification principale |
|---|---:|---|
| Noyau numérique | 4,5/5 | Cas fermés, tangente, cutback, réactions et cisaillement testés |
| Validation scientifique | 3,0/5 | Une partition article réelle a convergé et les métriques DIC/EF sont archivées ; ROI raccordé et conventions exactes à valider |
| Ingénierie logicielle | 4,5/5 | API typée, modules séparés, CI, 156 tests, revue documentée |
| Reproductibilité | 4,5/5 | Données LFS, préparation atomique, manifestes et smoke test DIC réel |
| Performance | 4,0/5 | 10k–100k mesurés et partition article de 234,6k exécutée ; plus grande partition intérieure non mesurée |
| Documentation | 4,0/5 | Contrats, ADR, tutoriel et limites ; figures finales non reproductibles |

Les notes ne doivent pas être relevées artificiellement par des cas
synthétiques. Pour atteindre 4/5 partout, les chemins critiques sont désormais :

1. terminer et vérifier le pipeline autonome depuis les données DIC versionnées ;
2. exécuter un calcul réel partitionné, raccorder les champs et reproduire les
   métriques expérimentales accessibles ;
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
| 2026-07-24 | Installation TFEL/MFront et MGIS | Versions et imports depuis `.venv` | TFEL 5.1.0, MGIS 3.1, interface générique active | Réussi |
| 2026-07-24 | Parité constitutive Python/MFront | `validation/reference_data/mfront_material_point_v1/report.json` | L2 contrainte `0,227–0,368 %`, erreur PEEQ max `<3,88e-5` | Réussi |
| 2026-07-24 | Suite après backend MFront | Ruff, mypy, compilation MFront et couverture | 165 tests, 94,25 %, dont 2 tests MGIS réels | Réussi |
| 2026-07-24 | Performance constitutive Python/MFront | 200k points, 20 incréments, 2 répétitions | Python 12,347 s ; MFront série 13,333 s ; MFront 8 threads 3,527 s | Réussi |
| 2026-07-24 | Reproductibilité MFront parallèle | États série/parallèle sur 4 millions de mises à jour | Écarts max contrainte et PEEQ strictement nuls | Réussi |
| 2026-07-24 | Suite après pool MGIS | Ruff, mypy et couverture avec bibliothèque réelle | 167 tests, 94,21 % | Réussi |

## 14. Journal des mises à jour

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
