# Modèle numérique supporté

Le moteur résout uniquement le problème de reconstruction de l'article :
petites déformations, contraintes planes, maillage rectangulaire structuré
CPS4, intégration de Gauss 2×2, plasticité J2 et écrouissage
Ludwik-Hollomon.

## Entrées publiques

`run_case_study` reçoit une `CaseStudyConfig` et quatre champs :

- `displacement_x_mm`, `displacement_y_mm` aux nœuds, de forme
  `(nx + 1, ny + 1)` ;
- `yield_stress_mpa`, strictement positif aux éléments, de forme `(nx, ny)` ;
- `hardening_coefficient_mpa`, positif ou nul aux éléments.

Les tailles, valeurs non finies et domaines physiques sont contrôlés avant
l'appel au noyau. PyPardiso/MKL est obligatoire par défaut et son absence
provoque une erreur explicite avant l'assemblage.

## Sorties

`FEMResult` sépare et nomme les champs finaux :

- déplacement nodal en mm ;
- contraintes `S11`, `S22`, `S12` en MPa ;
- déformations totales `E11`, `E22`, `gamma12` ;
- déformations plastiques ;
- PEEQ ;
- réactions nodales ;
- instantanés éventuellement demandés aux pseudo-temps.

Le résultat brut historique sous forme de dictionnaire n'est plus l'API
publique.

## Intégration constitutive

La loi tabulée est le mode nominal pour la parité avec l'article. Sa grille
contient 1000 valeurs : zéro, `1e-6`, puis un espacement linéaire jusqu'à 0,2.
Le mode analytique de Ludwik reste disponible pour les vérifications fermées.

Le retour radial vectorisé utilise un Newton borné avec repli par bissection.
La tangente élastoplastique cohérente est contrôlée par différences finies. Le
solveur global utilise des incréments de pseudo-temps, Newton-Raphson et une
réduction automatique du pas en cas de non-convergence.

## État du refactoring

Le noyau historique a été déplacé dans
`fem_inhouse.core.solver_legacy` pour qu'il soit réellement empaqueté et
testable. Il reste volontairement identifié comme dette technique : les
prochaines étapes doivent en extraire le maillage, l'élément, la loi
constitutive, l'assemblage et la boucle non linéaire sans changer les résultats
de référence.

