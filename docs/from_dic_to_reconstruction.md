# Refaire le calcul depuis les données DIC

Ce document décrit le chemin principal du dépôt. La comparaison Abaqus est une
validation externe différée : elle ne bloque ni la préparation des données, ni
le calcul, ni le raccordement des champs.

## 1. Données versionnées

`data/raw/case_study` contient quatre tableaux immuables suivis par Git LFS :

| Fichier brut | Rôle |
|---|---|
| `U_40.npy` | déplacement DIC `u_y`, en pixels |
| `V_40.npy` | déplacement DIC `u_x`, en pixels |
| `el_thresh50.npy` | limite d'élasticité locale, en MPa |
| `Hardening_coeff_el_Thresh50.npy` | multiplicateur local du coefficient `K` |

Le manifeste brut contient les empreintes SHA-256, formes, types et ambiguïtés
connues. `prepare-case` refuse une taille, une empreinte, une forme ou un type
différent.

Le ZIP reçu n'est pas conservé car il duplique exactement ces fichiers. Le
HDF5 de plasticité cristalline contient les mêmes déplacements ainsi que des
données microstructurales hors du périmètre supporté.

## 2. Préparation canonique

Installation depuis un clone neuf :

```bash
git lfs install
git lfs pull
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/fem-inhouse backend
```

Préparation nominale de l'article :

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

La commande produit :

```text
data/processed/case-study/
├── displacement_x_mm.npy            # 3601 × 3101, float64
├── displacement_y_mm.npy            # 3601 × 3101, float64
├── yield_stress_mpa.npy              # 3600 × 3100, float64
├── hardening_coefficient_mpa.npy     # 3600 × 3100, float64
└── manifest.json
```

Les choix scientifiques ne sont jamais silencieux :

- `V_40` devient `u_x` et `U_40` devient `u_y` ;
- la conversion vaut `0,00184 mm/pixel` ;
- `K=380 MPa` correspond à l'article ;
- `--hardening-scale-mpa 396` reproduit le générateur historique ;
- `--nonfinite-policy nearest` remplace les neuf valeurs déclarées par le plus
  proche voisin fini, avec indices enregistrés ;
- `edge-pad-upper` duplique la dernière ligne et la dernière colonne pour
  construire la grille nodale `3601×3101`.

La préparation est atomique. Une exécution répétée avec la même source et la
même configuration vérifie puis réutilise les sorties. Une configuration
différente dans le même répertoire est refusée.

## 3. Calcul de contrôle sur des données réelles

Un crop central permet de vérifier rapidement toute la chaîne sans créer un cas
synthétique :

```bash
examples/run_dic_smoke.sh
```

La séquence détaillée exécutée par ce script est :

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study-10x10 \
  --crop-nx 10 \
  --crop-ny 10

.venv/bin/fem-inhouse partition \
  --input data/processed/case-study-10x10 \
  --output results/dic-smoke-10x10 \
  --count 25 \
  --padding 0 \
  --increments 10 \
  --solve-pending

for field in U S E PE PEEQ RF; do
  .venv/bin/fem-inhouse partition \
    --input data/processed/case-study-10x10 \
    --output results/dic-smoke-10x10 \
    --count 25 \
    --padding 0 \
    --increments 10 \
    --stitch "$field"
done
```

Ce contrôle utilise les pixels centraux exacts du ROI. Le manifeste de
préparation enregistre les bornes du crop dans les coordonnées des tableaux
bruts.

Après compilation de la loi MFront, la comparaison reproductible des deux
backends sur ce même crop est :

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/compare_fem_backends.py \
  --input data/processed/case-study-10x10 \
  --output results/mfront-newton-dic-10x10 \
  --threads 2
```

Cette campagne conserve les six champs de chaque backend, les diagnostics
Newton, les empreintes et les métriques. Pour une partition ordinaire, la CLI
sélectionne MFront avec `--constitutive-backend mfront`.

## 4. Calcul de production

Le ROI complet ne doit pas être résolu monolithiquement. Chaque tâche traite une
partition, puis les champs sont raccordés après validation de toutes les
sorties.

Préparer le manifeste et lister les tâches :

```bash
.venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --list-pending
```

Calculer une partition :

```bash
.venv/bin/fem-inhouse partition \
  --input data/processed/case-study \
  --output results/reconstruction-100 \
  --count 100 \
  --padding 150 \
  --partition-id 0
```

Sur un ordonnanceur, `--partition-id` reçoit l'identifiant de tâche. Le modèle
Slurm est fourni dans `examples/slurm_partition_array.sh`.

Après la dernière partition :

```bash
for field in U S E PE PEEQ RF; do
  .venv/bin/fem-inhouse partition \
    --input data/processed/case-study \
    --output results/reconstruction-100 \
    --count 100 \
    --padding 150 \
    --stitch "$field"
done
```

## 5. Traçabilité et limites actuelles

Chaque niveau conserve son propre manifeste :

- le manifeste brut fixe l'identité des données reçues ;
- le manifeste de préparation fixe toutes les transformations ;
- le manifeste de calcul fixe le code, la configuration, le partitionnement et
  les empreintes des entrées ;
- chaque partition possède un statut et les empreintes de ses sorties.

Les étapes DIC 1 à 5 utilisées historiquement comme état initial ne sont pas
disponibles. Le dépôt reproduit donc actuellement le calcul piloté par le champ
final de l'étape 40, sans prétendre reproduire une soustraction de baseline
absente.

La formule exacte du BGE de l'article n'est pas publiée. Le dépôt conserve une
métrique d'interface explicitement nommée `interface_gradient_ratio`.
