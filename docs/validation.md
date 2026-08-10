# Stratégie de validation

La validation est hiérarchique : les tests fermés du noyau ne remplacent pas la
comparaison Abaqus, et la comparaison Abaqus ne remplace pas la reproduction
des champs expérimentaux.

## Qualification of coupled solvers

Coupled solver qualification follows this order:

0. **Material point:** qualify the constitutive behaviour independently.
1. **Small coupled pilot:** verify formulation, Jacobian, transactions and
   solution identity on a homogeneous or controlled heterogeneous case.
2. **Real P43 M20:** run the DIC path with the real local material maps.
3. **P43 M100:** qualify the robustness reference before evaluating a candidate.
4. **Scaling and performance:** compare timings only after both methods
   converge to the same discrete solution with comparable criteria and
   physical parameters.

Performance from a candidate solver is not qualification evidence when the
robustness reference has not first converged on the same case. The current
partitioned reference is the nested nonlocal fixed point documented in
{doc}`reference/numerics/nonlocal_fixed_point`; the monolithic `(u, chi)` path
is a candidate under qualification.

## Niveau mathématique

La suite automatique contrôle actuellement :

- dérivées et partition de l'unité du CPS4 ;
- Jacobien positif ;
- symétrie et trois modes rigides de la matrice élémentaire ;
- patch affine élastique ;
- retour élastique et retour plastique ;
- retours plastiques uniaxial, équibiaxial et en cisaillement ;
- loi MFront analytique non capée comme chemin nominal ;
- comportement tabulé au-delà de la dernière valeur comme régression
  historique ;
- tangente cohérente par différences finies ;
- traction équibiaxiale plastique avec solution analytique ;
- stabilité entre 5, 10 et 20 incréments ;
- convergence d'un damier hétérogène de paramètres ;
- équilibre de la résultante des réactions ;
- erreur explicite après échec des réductions de pas ;
- résultat fini et diagnostic explicite des entrées invalides.

## Parité monolithique/partitionnée

Le cas équibiaxial homogène 6×6 est calculé :

- en une résolution monolithique ;
- en quatre partitions sans padding ;
- en quatre partitions avec un padding d'un élément.

Les champs `U`, `S`, `E` et `PEEQ` raccordés sont comparés au champ monolithique.
Ce test valide les indices, les conditions aux limites locales et le
raccordement sur un problème affine. Il ne prédit pas le padding nécessaire au
cas hétérogène.

## Métriques de champs

`field_error_metrics` calcule sur le même masque de valeurs valides :

- RMSE ;
- erreur absolue moyenne ;
- erreur moyenne signée (`prédiction - référence`) ;
- maximum absolu ;
- erreur L2 relative ;
- corrélation de Pearson.

Le nombre de pixels effectivement comparés est toujours conservé. Les champs de
formes différentes et les masques incompatibles sont refusés.

`localization_overlap_metrics` sélectionne indépendamment les pixels du
quantile supérieur de chaque champ, puis calcule :

- intersection sur union (Jaccard) ;
- coefficient de Dice ;
- rappel par rapport à la zone de référence ;
- précision par rapport à la zone prédite.

Le seuil de chaque champ, les effectifs et l'intersection sont conservés. Les
ex æquo au seuil sont tous inclus ; la fraction sélectionnée peut donc être
légèrement supérieure à la fraction demandée. Cette métrique mesure la
coïncidence spatiale des zones fortement localisées sans confondre leurs
amplitudes.

## Rapport automatique avec seuils pré-déclarés

`compare-fields` exige les seuils au moment du lancement, avant d'afficher les
résultats. Il écrit un rapport JSON complet et la carte signée
`prédiction - référence` :

```bash
fem-inhouse compare-fields \
  --reference reference_evm.npy \
  --prediction reconstructed_evm.npy \
  --report validation/evm-report.json \
  --difference validation/evm-difference.npy \
  --top-fraction 0.10 \
  --max-rmse 0.005 \
  --max-mae 0.005 \
  --min-correlation 0.95 \
  --min-localization-iou 0.70
```

La commande retourne un code nul seulement si RMSE, MAE, corrélation et
recouvrement satisfont simultanément les seuils. Elle suppose que les deux
tableaux ont déjà été placés au même emplacement physique ; elle ne réalise
aucun recalage implicite.

## Gradient aux interfaces

`interface_gradient_ratio` compare, direction par direction, les gradients
absolus aux frontières des cœurs avec le gradient moyen du champ dans la même
direction. Une valeur proche de 1 signifie que le gradient d'interface est
typique du champ ; une valeur supérieure indique un saut renforcé.

L'article décrit qualitativement son BGE comme un rapport de discontinuité de
déformation aux interfaces convergeant vers 1, mais ne publie pas sa formule
complète. La métrique implémentée est donc nommée explicitement
`interface_gradient_ratio` et **ne doit pas être présentée comme le BGE exact de
l'article** tant que le script d'analyse original n'a pas été retrouvé.

## Comparaisons restant indispensables

La maturité scientifique 4/5 exige encore :

- les `.inp` exacts et le script d'extraction ODB ;
- les mêmes emplacements physiques et conventions de cisaillement ;
- la comparaison `U1`, `U2`, `S11`, `S22`, `S12`, `E11`, `E22`, `E12`, PEEQ et
  réactions ;
- l'étude padding 50/100/150/200 ;
- l'exécution et le raccordement des 100 partitions du ROI ;
- l'application au ROI raccordé du masque et des conventions métriques exactes
  de l'article.

La partition hétérogène de coin `510×460` est désormais validée avec le backend
MFront nominal. Ses six champs, contrôles mécaniques et comparaisons avec le
chemin Python tabulé sont conservés sous
`validation/reference_data/article_100p_pad150_p0000_mfront_v1`.
