# E-SRIX-FEMU-SHADOW-003C — pré-enregistrement

## Objet

Qualifier le pas de différence finie des histoires shadow sur le chemin L3
complet de 809 incréments, sans modifier le chemin, les tolérances, le modèle
SRIX, le solveur global ou l'observation.

## Calculs

- chemin L3 exact de `srix_femu_path_convergence_v4` : 809 incréments ;
- `h=0.0015` : candidat principal ;
- `h=0.001` : contrôle indépendant ;
- mêmes huit points d'observation, mêmes conditions initiales et même profil
  d'observation ;
- aucun L4, aucune identification et aucun calcul P43.

## Critères fixés avant calcul

Entre les deux Jacobiennes L3 :

- erreur L2 relative de chaque colonne `< 5e-3` ;
- cosinus de chaque colonne `> 0.99999` ;
- angle maximal du sous-espace de rang 3 `< 2°` ;
- aucun changement qualitatif du mode faible `Q-b`.

Le gate L2→L3 conserve ensuite les seuils PATH-002S pré-enregistrés, sans
modification. `h=0.0015` ne sera adopté que si les deux calculs L3 terminent et
respectent ces critères de stabilité.
