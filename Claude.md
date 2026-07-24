# Plan de mise à niveau de `fem_inhouse`

Dernière mise à jour : 2026-07-24
Statut global : **cadrage validé, travaux non commencés**
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

### Blocages et défauts connus

- [ ] `test_config.py` est absent du projet livré
- [ ] Les scripts de validation ne sont pas exécutables de manière autonome
- [ ] Des chemins Windows absolus sont présents
- [ ] La courbe étiquetée « FEM stress » remplace la contrainte EF directe par
      une reconstruction de Ludwik après plastification
- [ ] Les quatre courbes scientifiques de l'article ne sont pas séparées
- [ ] La table plastique par défaut utilise 50 points, contre 1000 points dans
      l'article
- [ ] Les conventions d'axes DIC ne sont pas cohérentes dans tous les scripts
- [ ] Les conventions cisaillement tensoriel/ingénieur ne sont pas garanties
- [ ] Le seul test intégré n'asserte pas la valeur de PEEQ
- [ ] Aucun moteur de partitionnement/raccordement n'existe
- [ ] Aucun traitement hors mémoire du ROI complet n'existe
- [ ] Aucun manifeste de dépendances ou verrouillage des versions n'existe
- [ ] Aucun historique Git exploitable n'est présent dans le dossier
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

- [ ] Écrire les conventions d'axes `U/V`, `x/y`, axes NumPy 0/1
- [ ] Définir les unités de toutes les entrées et sorties
- [ ] Définir `epsilon_xy` tensoriel et `gamma_xy` ingénieur
- [ ] Définir la formule de `epsilon_vM` sous contrainte plane
- [ ] Définir les quatre courbes de contrainte-déformation
- [ ] Vérifier la section et l'épaisseur réellement utilisées dans Abaqus
- [ ] Identifier les fichiers exacts ayant produit les résultats de l'article
- [ ] Établir un jeu de données réduit, versionnable et non confidentiel
- [ ] Décider et documenter les tolérances avant comparaison finale

**Critère de sortie :** document scientifique relu et approuvé, sans convention
implicite.

### Semaines 2–3 — Parité numérique avec Abaqus

- [ ] Reproduire la table plastique Abaqus :
  - domaine `0 <= ep <= 0.2` ;
  - 1000 points ;
  - traitement documenté du premier incrément `1e-6`.
- [ ] Tester séparément loi analytique et loi tabulée
- [ ] Corriger le calcul et l'étiquetage de la contrainte EF directe
- [ ] Corriger les conventions d'axes et de cisaillement
- [ ] Comparer contraintes et déformations au même emplacement physique
- [ ] Définir la méthode commune de calcul des déformations depuis `U`
- [ ] Vérifier `U1`, `U2`, `S11`, `S22`, `S12`, `PEEQ`
- [ ] Vérifier le signe et la définition des réactions
- [ ] Ajouter des assertions sur PEEQ au test biaxial

**Critère de sortie :** petit cas identique exécuté par Abaqus et `fem_inhouse`,
avec rapport automatique champ par champ.

### Semaines 4–5 — Ingénierie logicielle

- [ ] Créer un `pyproject.toml`
- [ ] Verrouiller les dépendances et versions
- [ ] Créer un paquet sous `src/fem_inhouse`
- [ ] Séparer :
  - maillage ;
  - élément ;
  - matériau ;
  - assemblage ;
  - solveur non linéaire ;
  - résultats ;
  - post-traitement.
- [ ] Remplacer les 19 paramètres de `run_fem` par des configurations typées
- [ ] Ajouter les validations d'entrée
- [ ] Supprimer les effets de bord lors des imports
- [ ] Supprimer les chemins absolus
- [ ] Ajouter une CLI limitée au cas d'étude
- [ ] Ajouter Ruff, Pyright ou mypy, pytest et couverture
- [ ] Ajouter une journalisation structurée
- [ ] Échouer explicitement si PyPardiso n'est pas disponible en production

**Critère de sortie :** installation fraîche et cas réduit exécutables par une
commande documentée.

### Semaines 6–8 — Partitionnement, padding et raccordement

- [ ] Définir une grille déterministe de 25 partitions
- [ ] Définir une grille déterministe de 100 partitions
- [ ] Gérer correctement les partitions de bord et de coin
- [ ] Extraire les cartes matériau et les déplacements locaux
- [ ] Ajouter le padding configurable
- [ ] Résoudre indépendamment chaque partition
- [ ] Enregistrer uniquement les résultats nécessaires par partition
- [ ] Extraire et raccorder les cœurs non recouverts
- [ ] Garantir l'absence de trous, doublons et décalages d'indices
- [ ] Permettre une reprise après interruption
- [ ] Ajouter un manifeste et une empreinte des entrées
- [ ] Produire des fichiers `.npy` mappés en mémoire pour le champ global
- [ ] Rendre l'ordre d'exécution des partitions sans effet sur le résultat
- [ ] Ajouter un modèle de job array pour le calcul parallèle si nécessaire

**Critère de sortie :** domaine réduit identique entre calcul monolithique et
calcul partitionné avec padding suffisant.

### Semaine 9 — Performance et ressources

- [ ] Mesurer temps et mémoire pour 10k, 50k, 100k et 350k éléments
- [ ] Mesurer séparément assemblage, factorisation, Newton et écriture
- [ ] Vérifier le nombre de threads MKL
- [ ] Définir la taille maximale d'une partition pour la machine cible
- [ ] Comparer PyPardiso au repli SciPy sur les petits cas
- [ ] Définir un budget mémoire et un budget de temps par partition
- [ ] Vérifier l'absence de copies mémoire évitables
- [ ] Documenter la stratégie de parallélisation

**Critère de sortie :** dimensionnement documenté avant tout calcul sur
11,16 millions d'éléments.

### Semaines 10–11 — Validation hiérarchique

#### Niveau 1 : vérification mathématique

- [ ] Partition de l'unité et dérivées des fonctions de forme
- [ ] Jacobien positif
- [ ] Patch test élastique
- [ ] Trois modes rigides
- [ ] Retour plastique uniaxial, biaxial et en cisaillement
- [ ] Tangente par différences finies
- [ ] Cas tabulé dans chaque segment et au-delà de `ep = 0.2`
- [ ] Équilibre des réactions
- [ ] Convergence en nombre d'incréments

#### Niveau 2 : parité Abaqus

- [ ] Comparaison sur 10×10 ou 20×20
- [ ] Comparaison sur un sous-domaine hétérogène représentatif
- [ ] Comparaison à plusieurs pseudo-temps
- [ ] Rapport automatique avec seuils de succès

#### Niveau 3 : partitionnement

- [ ] Référence monolithique sur domaine réduit
- [ ] Comparaison sans recouvrement
- [ ] Comparaison avec padding 50, 100, 150 et 200
- [ ] Étude du nombre de partitions
- [ ] Calcul et convergence de la métrique BGE
- [ ] Mesure spécifique des erreurs aux interfaces

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

- [ ] README de démarrage rapide
- [ ] Tutoriel complet du cas réduit
- [ ] Documentation du modèle numérique
- [ ] Documentation des conventions
- [ ] Documentation du partitionnement
- [ ] Documentation de la validation
- [ ] Documentation des limites scientifiques
- [ ] Commandes uniques `test`, `validate`, `example`
- [ ] CI verte sur une installation fraîche
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
│   │   ├── material.py
│   │   ├── assembly.py
│   │   └── solver.py
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
- [ ] Tangente cohérente vérifiée automatiquement
- [ ] Convergence robuste sur cas homogène et hétérogène
- [ ] Échec de convergence diagnostiqué sans résultat silencieusement invalide

### Validation scientifique

- [ ] Parité Abaqus démontrée sur petits cas
- [ ] Métriques de l'article reproduites ou écarts expliqués
- [ ] Contrainte directe séparée des reconstructions
- [ ] Artefacts de partition quantifiés
- [ ] Seuils définis avant lecture des résultats finaux

### Ingénierie logicielle

- [ ] API publique typée
- [ ] Au moins 85 % de couverture des lignes
- [ ] Au moins 80 % de couverture des branches
- [ ] Couverture dédiée de toutes les fonctions constitutives critiques
- [ ] Aucun avertissement qualité non justifié
- [ ] Revue de code obligatoire pour les formules numériques

### Reproductibilité

- [ ] Installation fraîche reproductible
- [ ] Versions verrouillées
- [ ] Données de référence identifiées par empreinte
- [ ] Aucun chemin dépendant d'un poste personnel
- [ ] Résultats accompagnés de leur configuration et version du code
- [ ] Workflow reprenable partition par partition

### Performance

- [ ] PyPardiso/MKL utilisé et vérifié
- [ ] Temps et mémoire mesurés
- [ ] Cas de production compatible avec la machine cible
- [ ] Traitement hors mémoire du ROI complet
- [ ] Absence de régression de performance supérieure au seuil défini

### Documentation

- [ ] Une personne externe peut installer et exécuter le cas réduit
- [ ] Une personne externe peut reproduire les figures principales
- [ ] Les hypothèses et limites sont visibles
- [ ] Les descripteurs locaux ne sont pas présentés comme propriétés de grains

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
| Convention définitive U/V et x/y | Ouvert | À définir | S1 |
| Format des données globales hors mémoire | Proposition : `.npy` memmap | À définir | S4 |
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

## 14. Journal des mises à jour

### 2026-07-24 — Création

- Audit initial du code existant
- Installation de l'environnement scientifique et de PyPardiso/MKL
- Vérifications élémentaires du noyau numérique
- Lecture de `ArticleSource/ArticleAdil.pdf`
- Recentrage du projet sur la reconstruction cinématique partitionnée
- Extension du planning de 9 à 12 semaines
