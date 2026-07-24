# ADR 0005 — Loi MFront analytique par défaut

Date : 2026-07-24  
Statut : accepté

## Contexte

Le chemin historique Abaqus discrétise la loi de Ludwik en 1000 points entre
PEEQ 0 et 0,2, puis la représentation Python plafonne la contrainte
d'écrouissage au-delà de cette borne. Cette discrétisation n'apporte pas de
contenu scientifique au cas d'étude et introduit un comportement artificiel
hors de son intervalle.

La loi MFront J2/Ludwik a été validée au point matériel puis dans la boucle
Newton sur un crop DIC réel. Elle utilise les mêmes cartes locales `sy0`, `K`
et `n`, régularise seulement le voisinage de PEEQ nul et ne fixe aucune borne
supérieure à PEEQ.

## Décision

`SolverConfig` et la CLI utilisent désormais par défaut :

- `constitutive_backend="mfront"` ;
- `hardening_mode="ludwik"` ;
- `PixelLudwikJ2Plasticity` via MGIS.

Le chemin MFront ne construit pas la table Python. La loi tabulée reste
disponible uniquement par sélection explicite du backend Python et du mode
`tabular`.

## Conséquences

- les calculs nominaux exigent une bibliothèque MFront compilée et
  l'environnement TFEL actif ;
- PEEQ n'est plus plafonné artificiellement à 0,2 ;
- les tests historiques demandent explicitement le backend Python ;
- les anciens résultats tabulés restent immuables et identifiables par leur
  manifeste ;
- toute comparaison avec Abaqus doit indiquer si elle vise la loi analytique
  nominale ou la table historique.
