# Plan de mise à niveau de `fem_inhouse`

Dernière mise à jour : 2026-07-24
Statut global : **API solveur testée et partitionnement déterministe disponibles**
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
2. Les fichiers d'entrée Abaqus et le script d'extraction ODB utilisés pour
   produire les résultats de l'article
3. Les données DIC et cartes de paramètres réellement utilisées
4. Les tests automatisés du présent projet

Toute contradiction entre l'article, les entrées Abaqus et le code doit être
documentée et résolue explicitement. Le comportement courant du code n'est pas
considéré comme une spécification par défaut.

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
  raccordement.

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
- comparaison avec les références Abaqus et les champs expérimentaux.

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

### Semaine 1 — Contrat scientifique

- [x] Écrire les conventions d'axes `U/V`, `x/y`, axes NumPy 0/1
- [x] Définir les unités de toutes les entrées et sorties
- [x] Définir `epsilon_xy` tensoriel et `gamma_xy` ingénieur
- [x] Définir la formule de `epsilon_vM` sous contrainte plane
- [x] Définir les quatre courbes de contrainte-déformation
- [ ] Vérifier la section et l'épaisseur réellement utilisées dans Abaqus
- [ ] Identifier les fichiers exacts ayant produit les résultats de l'article
- [ ] Établir un jeu de données réduit, versionnable et non confidentiel
- [ ] Décider et documenter les tolérances avant comparaison finale

**Critère de sortie :** document scientifique relu et approuvé, sans convention
implicite.

### Semaines 2–3 — Parité numérique avec Abaqus

- [x] Reproduire la table plastique Abaqus :
  - domaine `0 <= ep <= 0.2` ;
  - 1000 points ;
  - traitement documenté du premier incrément `1e-6`.
- [x] Tester séparément loi analytique et loi tabulée
- [x] Corriger le calcul et l'étiquetage de la contrainte EF directe
- [x] Corriger les conventions d'axes et de cisaillement
- [ ] Comparer contraintes et déformations au même emplacement physique
- [x] Définir la méthode commune de calcul des déformations depuis `U`
- [ ] Vérifier `U1`, `U2`, `S11`, `S22`, `S12`, `PEEQ`
- [ ] Vérifier le signe et la définition des réactions
- [x] Ajouter des assertions sur PEEQ au test biaxial

**Critère de sortie :** petit cas identique exécuté par Abaqus et `fem_inhouse`,
avec rapport automatique champ par champ.

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
- [ ] Ajouter un modèle de job array pour le calcul parallèle si nécessaire

**Critère de sortie :** domaine réduit identique entre calcul monolithique et
calcul partitionné avec padding suffisant.

### Semaine 9 — Performance et ressources

- [~] Mesurer temps et mémoire pour 10k, 50k, 100k et 350k éléments
- [ ] Mesurer séparément assemblage, factorisation, Newton et écriture
- [~] Vérifier le nombre de threads MKL
- [ ] Définir la taille maximale d'une partition pour la machine cible
- [x] Réserver le repli SciPy au diagnostic ; PyPardiso reste obligatoire
- [ ] Définir un budget mémoire et un budget de temps par partition
- [ ] Vérifier l'absence de copies mémoire évitables
- [~] Documenter la stratégie de parallélisation

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

- [ ] Comparaison sur 10×10 ou 20×20
- [ ] Comparaison sur un sous-domaine hétérogène représentatif
- [ ] Comparaison à plusieurs pseudo-temps
- [ ] Rapport automatique avec seuils de succès

#### Niveau 3 : partitionnement

- [x] Référence monolithique sur domaine réduit
- [x] Comparaison sans recouvrement
- [ ] Comparaison avec padding 50, 100, 150 et 200
- [ ] Étude du nombre de partitions
- [!] Calcul et convergence de la métrique BGE
- [x] Mesure spécifique des erreurs aux interfaces

#### Niveau 4 : reproduction scientifique

- [ ] RMSE du déplacement `U2`
- [ ] RMSE et MAE de `epsilon_vM`
- [ ] Carte de différence signée
- [ ] BGE
- [ ] Corrélation spatiale des champs
- [ ] Recouvrement des zones de plus forte localisation
- [ ] Quatre courbes de contrainte-déformation séparées
- [ ] Intervalles de confiance calculés selon la méthode documentée
- [ ] Comparaison 25 partitions / 100 partitions avec padding 150

**Critère de sortie :** rapport reproductible expliquant les accords, écarts et
artefacts de raccordement.

### Semaine 12 — Documentation et version de référence

- [x] README de démarrage rapide
- [x] Tutoriel complet du cas réduit
- [x] Documentation du modèle numérique
- [ ] Documentation des conventions
- [x] Documentation du partitionnement
- [x] Documentation de la validation
- [ ] Documentation des limites scientifiques
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
├── LICENSE
├── Claude.md
├── src/fem_inhouse/
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

- [ ] Tous les tests mathématiques critiques passent
- [x] Tangente cohérente vérifiée automatiquement
- [x] Convergence robuste sur cas homogène et hétérogène
- [x] Échec de convergence diagnostiqué sans résultat silencieusement invalide

### Validation scientifique

- [ ] Parité Abaqus démontrée sur petits cas
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
- [ ] Revue de code obligatoire pour les formules numériques

### Reproductibilité

- [x] Installation fraîche reproductible
- [x] Versions verrouillées
- [~] Données de référence identifiées par empreinte
- [x] Aucun chemin dépendant d'un poste personnel
- [x] Résultats accompagnés de leur configuration et version du code
- [x] Workflow reprenable partition par partition

### Performance

- [x] PyPardiso/MKL utilisé et vérifié
- [ ] Temps et mémoire mesurés
- [ ] Cas de production compatible avec la machine cible
- [x] Traitement hors mémoire du ROI complet
- [ ] Absence de régression de performance supérieure au seuil défini

### Documentation

- [x] Une personne externe peut installer et exécuter le cas réduit
- [ ] Une personne externe peut reproduire les figures principales
- [x] Les hypothèses et limites sont visibles
- [x] Les descripteurs locaux ne sont pas présentés comme propriétés de grains

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
| Fichiers Abaqus exacts de référence | Ouvert | À définir | S1 |
| Épaisseur de section EF utilisée dans Abaqus | Ouvert | À définir | S1 |
| Convention définitive U/V et x/y | Résolu dans `docs/scientific_contract.md` | Projet | S1 |
| Format des données globales hors mémoire | Résolu : `.npy` memmap | Projet | S4 |
| Machine cible et budget mémoire | Ouvert | À définir | S9 |
| Seuils finaux de parité Abaqus | Ouvert | Revue scientifique | S3 |
| Schéma de production 25 ou 100 partitions | Ouvert | Revue scientifique | S11 |
| Métrique de localisation complémentaire | Ouvert | Revue scientifique | S10 |

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

## 14. Journal des mises à jour

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
