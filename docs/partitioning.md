# Partitionnement du ROI

Le domaine global est décrit en **éléments** par `(nx, ny)`, selon les
conventions d'axes de [`scientific_contract.md`](scientific_contract.md).
`PartitionLayout` découpe chaque axe en blocs équilibrés et numérote les
partitions de manière déterministe :

```text
partition_id = index_x * parts_y + index_y
```

Les deux schémas de l'article s'écrivent donc :

```python
layout_25 = PartitionLayout((3600, 3100), (5, 5), padding=150)
layout_100 = PartitionLayout((3600, 3100), (10, 10), padding=150)
```

Chaque partition possède :

- un **cœur**, qui couvre une zone unique du domaine ;
- une zone de **calcul**, obtenue en ajoutant le padding puis en la tronquant
  aux frontières globales ;
- des tranches globales et locales explicites pour les champs aux éléments et
  aux nœuds.

Le raccordement ne moyenne pas les zones recouvertes : seul le cœur de chaque
partition est conservé. Pour les champs nodaux, les nœuds partagés à gauche et
en bas appartiennent à la partition précédente. Cette règle garantit qu'un
nœud global est écrit exactement une fois.

`stitch_partition_files` ouvre les résultats locaux `.npy` en lecture mappée et
crée le champ global avec `numpy.lib.format.open_memmap`. La taille du ROI
complet n'impose donc pas de charger simultanément tous les champs en mémoire.

Le manifeste JSON enregistre les bornes exactes de chaque partition. Il est
déterministe pour une configuration donnée et constitue la base du futur
mécanisme de reprise et d'empreinte des entrées.

## Workflow reprenable

`PartitionWorkflow` associe le découpage aux quatre champs globaux et à la
configuration scientifique. `prepare()` écrit un manifeste immuable contenant :

- toute la configuration typée ;
- les bornes des partitions ;
- la version et l'empreinte SHA-256 du code Python empaqueté ;
- l'empreinte SHA-256 de chaque champ d'entrée ;
- la liste explicite des sorties conservées.

Les empreintes des tableaux sont calculées par blocs et acceptent donc les
tableaux `numpy.memmap`. Elles ne nécessitent pas une copie de la totalité du
ROI.

Chaque partition écrit atomiquement `U`, `S`, `E` et `PEEQ`, puis un
`status.json` contenant leurs empreintes. Au redémarrage, une partition n'est
réutilisée que si le manifeste et chaque fichier correspondent encore. Un
fichier manquant ou corrompu replace automatiquement la partition dans la liste
des calculs en attente.

## Exécution par CLI et job array

La commande `partition` ouvre les quatre fichiers `.npy` du contrat de données
en lecture mappée et infère `(nx, ny)` depuis les cartes matériau :

```bash
fem-inhouse partition \
  --input data/case_study \
  --output results/reconstruction-25 \
  --count 25 \
  --padding 150 \
  --list-pending
```

Une partition isolée se lance avec `--partition-id N`. C'est le point d'entrée
prévu pour un ordonnanceur ; chaque tâche écrit dans son propre répertoire et
les écritures atomiques utilisent des fichiers temporaires uniques. Le modèle
Slurm versionné se soumet ainsi :

```bash
export FEM_INPUT_DIR="$PWD/data/case_study"
export FEM_OUTPUT_DIR="$PWD/results/reconstruction-25"
mkdir -p logs
sbatch --array=0-24 examples/slurm_partition_array.sh
```

Pour la grille de 100 partitions :

```bash
FEM_PARTITION_COUNT=100 sbatch --array=0-99 examples/slurm_partition_array.sh
```

Après succès de toutes les tâches, chaque champ global est raccordé
indépendamment, sans le charger en mémoire :

```bash
fem-inhouse partition \
  --input data/case_study \
  --output results/reconstruction-25 \
  --count 25 \
  --padding 150 \
  --stitch S \
  --field-output results/reconstruction-25/global/S.npy
```

`--solve-pending` existe pour une exécution séquentielle ou un diagnostic
local. L'identité et les sorties des partitions ne dépendent pas de l'ordre
d'exécution. Le manifeste et les empreintes empêchent de reprendre
silencieusement avec d'autres données ou une autre version du code.
