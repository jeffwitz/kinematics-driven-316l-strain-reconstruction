# ADR 0001 — Limiter le logiciel au cas de reconstruction de l'article

- Statut : accepté
- Date : 2026-07-24

## Contexte

Le prototype a été présenté comme un remplacement d'Abaqus, alors que
l'objectif scientifique est la reconstruction cinématique d'un ROI DIC précis.
Une architecture générique d'éléments, de lois matériau et de chargements
augmenterait fortement le coût de validation sans servir l'article.

## Décision

Le logiciel supporte uniquement :

- petites déformations et contraintes planes ;
- maillage rectangulaire structuré CPS4, quadrature 2×2 ;
- plasticité J2 et écrouissage Ludwik analytique ou tabulé ;
- déplacements de bord prescrits depuis les champs DIC ;
- reconstruction par partitions paddées de 25 ou 100 zones.

Tout nouveau modèle physique exige un ADR distinct et sa propre validation. Il
ne doit pas être ajouté comme option opportuniste au noyau existant.

## Conséquences

L'API reste courte et fortement validée. Le projet ne prétend pas reproduire
les capacités générales d'Abaqus. La maturité est évaluée sur le cas publié,
pas sur l'étendue fonctionnelle.
