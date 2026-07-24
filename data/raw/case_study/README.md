# Données brutes du cas d'étude

Ce répertoire contient les quatre tableaux reçus avec le code du doctorant.
Ils sont conservés sans modification et versionnés avec Git LFS. Le fichier
`manifest.json` fixe leurs empreintes, formes, types et conventions connues.

Ces fichiers ne sont pas directement les entrées canoniques du solveur :

- `U_40.npy` et `V_40.npy` sont exprimés en pixels ;
- les noms historiques correspondent respectivement à `u_y` et `u_x` ;
- la carte d'écrouissage est un multiplicateur sans unité ;
- neuf valeurs de cette carte sont non finies ;
- la grille DIC doit être complétée explicitement pour fournir les nœuds d'un
  maillage de `3600 × 3100` éléments.

La commande `fem-inhouse prepare-case` réalise ces transformations, contrôle
les empreintes et écrit un manifeste de préparation dans un répertoire séparé.
Les données brutes ne doivent jamais être modifiées en place.

Le fichier `../../ArticleSource/ArticleAdil.pdf` et les scripts placés dans
`../../references/legacy_abaqus/` documentent la provenance scientifique et
les conventions historiques.
