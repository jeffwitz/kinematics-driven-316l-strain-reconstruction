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

