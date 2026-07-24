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
- instantanés éventuellement demandés aux pseudo-temps ;
- diagnostics typés de convergence : backend, temps total et par phase,
  incréments tentés et convergés, cutbacks, itérations de Newton et résidu
  final.

Le résultat brut historique sous forme de dictionnaire n'est plus l'API
publique.

## Réactions et épaisseur implicite

`reaction_force[i, j, component]` est la force nodale interne sur un degré de
liberté prescrit, avec le même axe et le même signe que le déplacement. En
traction uniforme, la somme est négative sur le bord de coordonnée minimale et
positive sur le bord opposé. Les degrés de liberté libres sont remis à zéro
dans ce champ de sortie.

Le noyau 2D n'applique aucun multiplicateur d'épaisseur : avec les longueurs en
mm et les contraintes en MPa, il représente donc une épaisseur implicite de
1 mm et les réactions sont en N. Cette convention est vérifiée sur le patch
test affine. L'épaisseur de la référence Abaqus ayant produit l'article reste
à identifier avant toute comparaison quantitative de réactions.

Les événements `nonlinear_solve_started`, `newton_iteration`,
`increment_cutback`, `snapshot_recorded` et `nonlinear_solve_completed` sont
émis via le module standard `logging`. Les itérations détaillées ne sont
émises que lorsque `verbose=True`.

## Intégration constitutive

Le mode nominal est la loi MFront J2/Ludwik analytique régularisée au voisinage
de zéro. Elle ne plafonne pas PEEQ : après le premier segment
`0 <= PEEQ <= 1e-6`, l'écrouissage suit `sy0 + K*PEEQ**n` sur tout le domaine
atteint par le calcul.

La loi Python tabulée à 1000 valeurs jusqu'à `PEEQ=0.2` est conservée uniquement
comme chemin historique explicite. Elle n'est ni construite ni allouée lorsque
le backend MFront est sélectionné.

MFront calcule la contrainte, les variables internes et la tangente cohérente.
Le solveur global utilise des incréments de pseudo-temps, Newton-Raphson et une
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
