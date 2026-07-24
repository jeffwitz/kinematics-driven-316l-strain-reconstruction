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

Le maillage structuré, l'élément CPS4, le modèle constitutif et l'assemblage
sparse sont maintenant séparés dans :

- `fem_inhouse.core.mesh` ;
- `fem_inhouse.core.element` ;
- `fem_inhouse.core.constitutive` ;
- `fem_inhouse.core.assembly`.

Le solveur non linéaire les utilise directement ; la parité du résultat est
protégée par les cas analytiques et partitionnés. L'incrémentation,
Newton-Raphson et la réduction automatique du pas sont isolés dans
`fem_inhouse.core.nonlinear`. `fem_inhouse.core.solver_legacy` ne conserve que
les imports de compatibilité des scripts historiques. Cette séparation
n'introduit aucun mécanisme générique au-delà du cas d'étude.
