# Choose an MFront backend

For the qualified 316L SRIX + EBSD workflow, use
`mfront-native-generalised-plane-stress` with
`gps_composite_fd_tangent: true`. Use
`mfront-3d-condensed-plane-stress` as the independent numerical reference and
for 3D behaviours without a GPS implementation.

| Backend | Usage recommandé | Avantage | Limite |
|---|---|---|---|
| `mfront-native-plane-stress` | lois natives 2D simples | direct | pas général pour CP 3D |
| `mfront-3d-condensed-plane-stress` | référence / nouvelle loi 3D | fonctionne avec toute loi 3D | condensation Python |
| `mfront-native-generalised-plane-stress` | SRIX GPS qualifié | monolithique et performant | nécessite une variante GPS de la loi |
| `python` | régression J2 historique | indépendant de MFront | pas production CP |

GPS (*generalised plane stress*) conserve les six composantes 3D et résout
localement les trois déformations hors plan nécessaires pour imposer
`sigma_zz = sigma_xz = sigma_yz = 0` dans le repère global. La variante GPS
SRIX porte cette fermeture dans son Newton constitutif, alors que la route
condensée l'effectue dans le bridge Python.

Lorsqu'un point GPS doit sous-intégrer un incrément, le dernier sous-pas ne
représente pas à lui seul la dérivée de la trajectoire composée. Le tangent FD
composite reconstruit cette dérivée pour les seuls points concernés. Il est
donc activé dans le workflow SRIX qualifié, avec un coût mesuré et limité.

## Production qualifiée

```yaml
solver:
  constitutive_backend: mfront-native-generalised-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4

  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6

    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3

    crystal_orientation:
      mode: ebsd
      # orientation source defined by the case
```

## Référence indépendante

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
```

La première configuration est la route de production qualifiée pour SRIX +
EBSD. La seconde est la référence indépendante à utiliser pour qualifier une
nouvelle loi ou vérifier un résultat GPS.

Pour les détails du choix de `R`, des paramètres 316L, des orientations et des
sorties par système, voir {doc}`use_srix_crystal_law`. La formulation de la
référence est détaillée dans
{doc}`../reference/numerics/three_dimensional_condensation`.
