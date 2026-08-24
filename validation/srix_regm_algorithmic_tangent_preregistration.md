# Preregistration — REGM reconditionné par tangent algorithmique

## Question

Remplacer, sur le jumeau exact uniquement, l'inverse élastique constant `K0^-1`
du REGM par l'inverse du tangent algorithmique local `K_alg,n^-1` assemblé à
chaque état de la trajectoire constitutive prescrite.

Le but est de tester si la perte des directions 3 et 4 de la Fisher REGM vient
principalement du reconditionneur élastique. Ce calcul reste un diagnostic de
sensibilité : il ne constitue pas encore un solveur séquentiel et ne modifie
pas la trajectoire DIC.

## Définition

À chaque état et pour chaque perturbation de paramètres :

1. rejouer SRIX causalement sur la cinématique mécanique exacte ;
2. demander la contrainte et le tangent algorithmique consistant ;
3. assembler `K_alg = B^T C_alg B` sur les DOF intérieurs ;
4. résoudre `delta_u = -K_alg^-1 B^T sigma` ;
5. appliquer le même transfert d'observation que la FEMU au pseudo-déplacement.

La dérivée paramétrique reste une différence finie centrale de pas `3e-3` en
coordonnées logarithmiques. Les huit états macro sont ceux de la Fisher FEMU
archivée.

## Règle

Comparer les spectres, vecteurs droits et angles de sous-espaces avec le
rapport `srix_regm_information_geometry_v1`. Aucun nouveau solveur global et
aucun calcul P43 ne sont autorisés.
