# Sous-pas GPS et tangent shadow : diagnostic M20

Le diagnostic same-state corrigé a établi que la sensibilité GPS et le Schur
brut coïncident à `1e-11` ou mieux. Le shadow runtime est donc une autre
expérience : il repart de l'état committé GPS, mais réintègre la loi raw en un
pas avec la déformation transverse finale GPS. Il ne constitue pas
automatiquement la dérivée de l'application GPS réellement composée de ses
sous-pas.

## Instrumentation

Le backend GPS enregistre maintenant, pour chaque évaluation :

- le masque et le nombre de divisions de sous-pas par point ;
- l'écart relatif `C_shadow-C_GPS` par point ;
- les différences d'ISV finales entre les trajectoires ;
- les deux tangentes utilisées ;
- le périmètre du shadow : `all`, `substepped` ou `non_substepped`.

## Test causal M20

Configuration : P43 M20 EBSD, SRIX, 8 incréments, Eisenstat--Walker,
LGMRES recyclé, 4 threads MFront.

| tangent de remplacement | Newton GPS | Newton référence |
|---|---:|---:|
| aucun shadow | 52 | 46 |
| shadow sur points non sous-pasés | 52 | 46 |
| shadow sur points sous-pasés | 47 | 46 |
| shadow sur tous les points | 47 | 46 |

Le test localise donc l'effet causal sur la classe de points sous-pasés, mais
les écarts `C_shadow-C_GPS` ne sont pas strictement équivalents au seul masque
de sous-pas : certains points non marqués sous-pasés suivent également une
trajectoire raw full-step différente. Le masque est donc un indicateur
opérationnel de sélection, pas une preuve que tous les écarts de trajectoire
se réduisent à un booléen `substepped`.

## Différences finies de l'application complète

Pour les points 96, 95 et 59, une différence finie centrale est calculée en
restaurant le même snapshot avant chaque perturbation et en laissant la
politique réelle de sous-pas s'exécuter.

| point | sous-pas | `h` | FD/GPS | FD/shadow |
|---:|:---:|---:|---:|---:|
| 96 | oui | `1e-7` | `1.35e-1` | `6.35e-2` |
| 95 | non | `1e-7` | `4.82e-9` | `4.82e-9` |
| 59 | non | `1e-7` | `1.85e-9` | `1.85e-9` |

La partition de sous-pas est restée identique pour les perturbations testées.
Au point 96, le shadow est plus proche de la dérivée composée mesurée par FD,
mais ne la reproduit pas exactement. Le résultat soutient donc l'hypothèse
du tangent de dernière sous-intégration, sans justifier encore une correction
de production.

Artefacts :

```text
validation/_generated/performance/gps_shadow_runtime_none_m20.json
validation/_generated/performance/gps_shadow_runtime_substepped_m20.json
validation/_generated/performance/gps_shadow_runtime_non_substepped_m20.json
validation/_generated/performance/gps_shadow_runtime_all_m20.json
validation/_generated/performance/gps_composite_tangent_fd_m20.json
```

Conclusion actuelle : le défaut algébrique GPS est rétracté. Le shadow est un
quasi-Newton issu d'une trajectoire raw différente ; il améliore M20 sur les
points sous-pasés, mais il ne doit pas être présenté comme un Schur same-state
ni activé par défaut sans une stratégie de tangent composé qualifiée.

## Première implémentation FD sélective

Une option expérimentale `gps_composite_fd_tangent=True` calcule maintenant la
différence finie centrale uniquement sur les points sous-pasés, à partir du
snapshot committé, avec `h=1e-6`. Les points sains gardent exactement le tangent
GPS de MFront.

Le benchmark M20 donne :

```text
GPS                 : 52 Newton
composite FD ciblé  : 45 Newton
référence raw       : 46 Newton
points FD cumulés   : 19
trajectoires FD     : 114
changements partition : 0
temps FD cumulé     : 0,28 s
```

Par rapport au GPS sans FD, les écarts finaux sont :

```text
déplacement : 4,24e-13
contrainte  : 2,65e-9
réactions   : 1,68e-9
glissements : 4,00e-9
```

Le critère M20 `N_Newton <= 47` est donc satisfait. Cette implémentation reste
expérimentale : elle crée des évaluateurs mono-point cachés et doit être
mesurée sur M100 avant toute décision de production.

Artefact comparatif :

```text
validation/_generated/performance/gps_composite_fd_vs_gps_m20.json
scripts/benchmark_gps_composite_fd_m20.py
```

## Vérification P43 M100

Un A/B a ensuite été exécuté sur P43 M100, 8 incréments, avec exactement la
même configuration :

```text
GPS natif                 76,37 s, 51 Newton
FD composite sélectif    65,62 s, 51 Newton
points FD                 0
```

Les champs sont identiques (`déplacement`, contraintes, réactions et
observables : écart relatif nul dans l'artefact). Comme aucun point M100 n'a
utilisé de sous-pas, le chemin FD n'a pas été activé : le rapport de temps
`0,859` ne constitue donc pas un gain attribuable à la tangente composite,
mais une variation de mesure entre deux exécutions du même chemin natif.

Artefact :

```text
validation/_generated/performance/gps_composite_fd_vs_gps_p43_m100.json
```
