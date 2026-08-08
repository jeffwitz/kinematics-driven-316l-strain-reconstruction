# Diagnostic — tangente GPS aux états EBSD réels (pénalité 85 vs 57 Newton)

Date: 2026-08-08
Code: `scripts/diagnose_gps_tangent_on_ebsd_states.py`.

## Question

La voie UMAT converge en 85 itérations globales contre 57 pour la référence
sur P43 100×100. Deux candidats non testés restaient (handoff §8.18) : une
tangente moins exacte aux orientations/états que la qualification à trois cas
ne couvre pas, ou une divergence de trajectoire authentique entre deux
solutions qui diffèrent de `1,2e-4`.

## Protocole

40 points échantillonnés du run M100 archivé (20 uniformes + 20 pondérés vers
les zones de plus forte déformation), déformation par élément déduite du champ
de déplacement (gradient centré, échelle spatiale corrigée par le pas pixel
`0,00184 mm`), orientation EBSD du crop correspondant. Chaque point est conduit
par une histoire proportionnelle de 8 pas depuis l'état vierge (un saut unique
à l'état final n'est pas un état que le matériau voit), puis la tangente du
dernier incrément est mesurée : retournée vs différences finies (qualité
propre de l'UMat), et UMat vs référence (l'opérateur Jv que les deux Newton
globaux utilisent réellement).

## Résultats

- **Les tangentes UMat et référence sont identiques à `1e-16`** aux états
  EBSD réels (médiane `3,6e-16`, max `6,4e-16`, 0/40 au-delà de `1e-6`). Les
  opérateurs Jv des deux Newton globaux sont les mêmes : **la pénalité
  d'itérations n'est pas un écart de tangente entre les backends** — le
  candidat (a) du handoff est réfuté.
- L'erreur FD propre de l'UMat (médiane `5,8e-2`, max `2,9e-1`, 40/40) est un
  **artefact du sous-pas** : la FD est instable en fonction de la perturbation
  (`0,053` à `1e-6`, `0,071` à `1e-5`, `0,207` à `1e-4` sur un état profond
  typique) — la réponse sous-pasée n'est pas lisse à l'échelle des sondes. La
  tangente retournée après sous-pas (celle du dernier sous-pas) s'écarte de la
  dérivée lisse de 5 à 20 % aux points sous-pasés, ce qui est un vrai défaut
  localisé de qualité de tangente — mais les deux backends partagent le même
  défaut (les tangentes sont identiques), donc il n'explique pas l'écart
  d'itérations entre eux.
- Le point d'échantillonnage 24,26 (norme de déformation `4,8e-3`) illustre le
  régime : orientation EBSD `[354,3 ; 39,5 ; 81,1]°`, FD instable.

## Conclusion

La pénalité `85 vs 57` reste inexpliquée par la tangente. Le candidat restant
est la **divergence de trajectoire** : les deux solutions diffèrent de
`~1e-4` (l'écart de champ mesuré), les Newton globaux convergent vers des
états légèrement différents, et le chemin d'itération diffère. Le sous-pas
introduit par ailleurs une non-lissité de la réponse aux états profonds
(5-20 % sur la tangente aux points sous-pasés) qui devrait être étudiée comme
contributeur possible de la convergence globale, même si elle est partagée par
les deux backends à l'échelle du point.

Aucune décision n'en est tirée : la référence condensée reste le backend de
production, et la pénalité reste le terme ouvert du bilan de performance.
