# SRIX monolithique en contrainte plane généralisée

Statut : **réalisé sans modification de TFEL/MFront** (2026-08-08). Ce
document décrit la voie suivie et l'état des alternatives.

## La formulation retenue — la fermeture dans les rangées transverses du résidu

Le comportement `Fcc316LForestRubinSrixGps`
(`mfront/Fcc316LForestRubinSrixGps.mfront`) est un `@DSL Implicit` sous
l'hypothèse `Tridimensional`, sur MFront 5.1.0 **non modifié**. Son système
local garde les 18 inconnues SRIX (`deel[6] + dg[12]`).

Le résidu cinématique est assemblé dans le repère **global** :

```text
rot(deel + sum dg m) - deto = 0
```

Ses trois rangées **dans le plan** sont la cinématique. Ses trois rangées
**transverses** ne sont pas des équations de ce problème — la surface libre
laisse la déformation totale transverse indéterminée — elles portent donc la
condition de contrainte plane à la place :

```text
(Q^T sigma Q)_zz = 0
(Q^T sigma Q)_xz = 0
(Q^T sigma Q)_yz = 0
```

dans le repère structural, normalisées par un module de référence. Les
déformations transverses `ezz`, `eyz`, `exz` sont des **sorties**, reconstruites
dans `@UpdateAuxiliaryStateVariables` comme `eel + sum(g m)` lues en repère
global, et repliées dans le gradient par le pont. La rotation
`Q_global_to_material` est passée à la loi comme **neuf propriétés matériau par
point** ; le pont est passif (pas de rotation d'entrée, pas de fermeture
Python).

Avantages structurels : pas de point selle, pas d'inconnues supplémentaires,
tout le bloc élastique du Jacobien est constant (assemblé une fois dans
`@InitLocalVariables`), et la contrainte plane est effectivement dans le
Newton constitutif — le « un unique Newton MFront » du cahier des charges.

Qualification ponctuelle acceptée : fermeture `2-4e-14 MPa`, tangente par
différences finies `1,2-1,6e-7`, accord avec la référence condensée
`1e-11` (les critères A1/A2 d'accord avec la référence ont été restaurés le
2026-08-08 ; l'interprétation « deux branches » qui les avait suspendus était
un artefact d'un bug de bookkeeping de déformation, corrigé en `6bfaf86`).
P43 20×20 : matériau `1,2-1,7×` la référence ; P43 100×100 : parité
(`1,02×` matériau) avec une pénalité d'itérations globales (`85` contre `57`)
mesurée mais non expliquée — le backend condensé reste le backend de
production par défaut.

## L'alternative générateur — parkée

Le prototype du fork `jeffwitz/tfel-generalised-plane-stress` (hypothèse
`GENERALISEDPLANESTRESS` dans le générateur MFront) est **parker** : la voie
UMat atteint l'objectif sur MFront non modifié, et le prototype du fork
aurait imposé la fermeture dans le repère de la loi — incompatible avec la
condition structurale pour les cristaux tournés (le DSL ne connaît pas la
rotation). Ce document remplace l'ancienne lecture selon laquelle le
monolithique exigeait une évolution du générateur.

## Backend de référence

`mfront-3d-condensed-plane-stress` (la condensation Python) reste la
référence numérique et le backend par défaut. La voie UMat est qualifiée au
point matériel et crédible, mais pas consolidée au niveau problème global
tant que la pénalité d'itérations n'est pas expliquée.
