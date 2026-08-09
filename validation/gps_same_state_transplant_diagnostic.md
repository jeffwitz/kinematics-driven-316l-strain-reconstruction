# GPS / raw Schur : diagnostic same-state corrigé

Ce diagnostic corrige l'ancien protocole qui transplantait un point dans
`manager.s0`, puis restaurait immédiatement le snapshot original pendant
l'évaluation. Les valeurs historiques de `gps_tangent_blocks.json` et de
`gps_direct_sensitivity.json` sont donc conservées comme archives, mais ne sont
pas des comparaisons same-state valides.

## Protocole

Pour les points 96, 95 et 59, le diagnostic courant :

1. construit un nouveau snapshot cible immutable ;
2. copie les variables internes partagées par nom ;
3. convertit explicitement le gradient GPS global vers le repère cristal de la
   référence, ou l'inverse ;
4. conserve séparément le gradient global committé ;
5. vérifie l'égalité physique du snapshot avant l'intégration ;
6. reconstruit à chaque exécution le tangent 3D brut de la référence et son
   complément de Schur.

Aucune loi `.mfront` n'est modifiée par ce diagnostic et aucune campagne M20 ou
M100 n'est lancée.

## Résultat

Le test Python `raw*`, qui remplace seulement les lignes de fermeture GPS par
les lignes cinématiques brutes, donne le même tangent que la sensibilité GPS.
La comparaison avec l'oracle Schur reconstruit en direct est :

| point | `|C_sens-C_raw,Schur|/|C_raw,Schur|` | `|C_raw*-C_sens|/|C_sens|` |
| ---: | ---: | ---: |
| 96 | `1.36e-13` | `9.76e-16` |
| 95 | `1.82e-11` | `7.30e-16` |
| 59 | `3.54e-14` | `3.70e-16` |

Le résultat est donc compatible avec l'identité algébrique attendue : au même
état physique committé, au même incrément et avec la même orientation, la
sensibilité GPS et le Schur brut coïncident à la précision numérique observée.
Les anciens écarts de l'ordre de `1e-3` ne doivent plus être utilisés pour
conclure à une différence intrinsèque entre les deux formulations.

## Interprétation et limites

Le diagnostic établit la cohérence de l'oracle de tangent au checkpoint. Il ne
qualifie pas encore le tangent de production `CondensedTangent`, qui reste
désactivé, et ne démontre pas que les trajectoires complètes des deux
backends sont identiques lorsqu'elles partent de leurs historiques propres.
Les différences de Newton observées précédemment doivent être réanalysées avec
ce protocole d'état corrigé.

Reproduction :

```bash
source /home/jeff/.local/share/tfel/env/env.sh
export PYTHONPATH=$PWD:/home/jeff/.local/lib/python3.12/site-packages:$PWD/src
export MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so
.venv/bin/python scripts/diagnose_gps_tangent_blocks.py \
  --output validation/_generated/performance/gps_tangent_blocks_same_state_v2.json
.venv/bin/python scripts/diagnose_gps_direct_sensitivity.py \
  --output validation/_generated/performance/gps_direct_sensitivity_same_state_v2.json
```
