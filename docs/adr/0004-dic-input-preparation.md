# ADR 0004 — Préparation explicite des entrées DIC

Statut : adopté comme profil reproductible initial, à soumettre à la revue
scientifique.

Date : 2026-07-24.

## Contexte

Les quatre tableaux reçus ont tous une forme `3600×3100`, mais ils ne suivent
pas directement le contrat du solveur :

- `U_40` et `V_40` sont en pixels et leurs noms ne portent pas les composantes
  physiques ;
- `3600×3100` éléments nécessitent `3601×3101` valeurs nodales ;
- la carte d'écrouissage est un multiplicateur et non un coefficient en MPa ;
- le générateur historique utilise `396 MPa`, tandis que l'article publie
  `380 MPa` ;
- neuf multiplicateurs d'écrouissage sont non finis.

Masquer ces transformations dans un script de lancement rendrait le calcul
impossible à auditer ou à reproduire.

## Décision

1. Les données reçues restent immuables sous `data/raw/case_study` et sont
   identifiées par SHA-256.
2. `prepare-case` écrit quatre tableaux canoniques dans un autre répertoire.
3. La convention retenue est `V → u_x`, `U → u_y`, conformément au générateur
   historique et au chargement observé.
4. La conversion des déplacements vaut `1,84 µm/pixel`, soit
   `0,00184 mm/pixel`.
5. Le profil nominal utilise `K=380 MPa`, valeur publiée. Le profil historique
   `396 MPa` reste accessible par option.
6. Aucune valeur non finie n'est réparée par défaut. Le calcul complet exige
   explicitement `--nonfinite-policy nearest`, qui utilise le plus proche
   voisin fini et enregistre chaque indice modifié.
7. La grille nodale est complétée sur les bornes supérieures en dupliquant la
   dernière ligne et la dernière colonne. Cette règle est nommée
   `edge-pad-upper` et enregistrée.
8. Un crop central éventuel est effectué avec des bornes déterministes,
   enregistrées dans les coordonnées de la source.

## Conséquences

- Une préparation peut être reproduite et comparée byte-à-byte.
- Un changement de facteur `K`, de crop ou de politique nécessite un nouveau
  répertoire de sortie.
- Le bord ajouté possède un gradient normal nul par construction. Son influence
  devra être étudiée lors de la revue scientifique du ROI complet.
- La réparation par voisin le plus proche est négligeable en proportion
  (`9 / 11 160 000`) mais reste une hypothèse et non une vérité expérimentale.
- L'absence des étapes DIC de baseline interdit de prétendre reproduire leur
  soustraction ; le profil actuel traite seulement l'étape 40 fournie.
