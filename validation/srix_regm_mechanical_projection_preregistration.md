# Preregistration — projection mécanique de la cinématique DIC sur un jumeau SRIX

## Question

Tester si une ou deux corrections linéarisées d'équilibre, appliquées à la
cinématique observée avant le replay SRIX, restaurent le classement REGM/FEMU
sans utiliser la cinématique latente exacte.

Cette expérience est limitée au jumeau M8 déjà qualifié. Aucun calcul P43 et
aucune optimisation de paramètres ne sont autorisés.

## Définition

À chaque état, le replay SRIX de référence fournit le résidu faible `r` et la
correction existante `d = -K0^-1 r`. La cinématique est mise à jour par :

```text
u_(j+1) = u_j + damping * d_j
```

La correction est calculée uniquement sur les degrés de liberté intérieurs ;
les valeurs de bord restent inchangées. Le replay reste causal et l'état
constitutif est engagé entre les incréments.

## Variantes fixées avant calcul

| Variante | Passes | Amortissement |
| --- | ---: | ---: |
| observed | 0 | — |
| proj025x1 | 1 | 0.25 |
| proj050x1 | 1 | 0.50 |
| proj100x1 | 1 | 1.00 |
| proj050x2 | 2 | 0.50 |
| proj100x2 | 2 | 1.00 |

La référence constitutive de projection est le preset SRIX du jumeau. Cette
dépendance est un risque connu et sera rapportée ; aucun choix ne sera fait
sur P43 à partir de cette étude.

## Critères descriptifs

Pour les mêmes vingt candidats et la même cible FEMU observée que l'étude de
placement de l'observation, rapporter :

- Spearman REGM/FEMU ;
- Pearson des logarithmes ;
- recouvrement du meilleur quintile ;
- RMS du pseudo-déplacement à la vérité du jumeau ;
- erreur RMS de la cinématique projetée par rapport à la vérité latente ;
- norme du résidu faible avant et après projection.

Les seuils de passage existants (`Spearman >= 0.80`, Pearson logarithmique
`>= 0.70`, recouvrement `>= 3/5`) sont descriptifs uniquement dans cette
expérience. Une réussite ne vaut pas validation réelle de la reconstruction
latente.

## Règle d'arrêt

Ne pas lancer P43. Si la projection ne restaure pas le classement sur le
jumeau transféré, cette voie est abandonnée avant toute donnée expérimentale.
