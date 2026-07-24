# ADR 0003 — Raccorder les cœurs à propriétaire unique

- Statut : accepté
- Date : 2026-07-24

## Contexte

Les zones paddées se recouvrent. Une moyenne des recouvrements rendrait le
résultat dépendant d'une règle de pondération non décrite dans l'article et
pourrait masquer les artefacts d'interface.

## Décision

Chaque partition résout sa zone paddée mais seul son cœur est conservé. Chaque
élément et chaque nœud global possède exactement un propriétaire déterministe.
Les résultats locaux sont écrits atomiquement, vérifiés par empreinte et
raccordés seulement lorsque toutes les partitions sont valides.

Les partitions sont indépendantes de l'ordre d'exécution et peuvent être
lancées par job array. Le champ global est un `.npy` mappé en mémoire.

## Conséquences

Il n'existe ni trou, ni doublon, ni moyenne implicite. Les discontinuités aux
interfaces restent mesurables. Une modification de cette règle invaliderait les
tests de parité et exige un nouvel ADR.
