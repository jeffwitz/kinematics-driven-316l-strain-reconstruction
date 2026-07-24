# ADR 0002 — Imposer PyPardiso pour les calculs de production

- Statut : accepté
- Date : 2026-07-24

## Contexte

Le solveur direct sparse domine le coût du cas d'étude. SciPy/SuperLU est
nettement trop lent pour les partitions visées. Le cahier des charges impose
PyPardiso ; un repli silencieux produirait un calcul fonctionnel mais
inexploitable en production.

## Décision

`run_case_study` exige PyPardiso/MKL par défaut et échoue avant l'assemblage si
le backend manque. Le repli SciPy reste dans le noyau historique uniquement
pour les petits diagnostics explicitement lancés avec
`require_pypardiso=False`.

La CI installe l'environnement verrouillé contenant PyPardiso et vérifie le
backend sur le cas réduit.

## Conséquences

Les installations de production sont prévisibles et les erreurs de dépendance
sont précoces. La portabilité vers une plateforme sans MKL n'est pas un
objectif actuel.
