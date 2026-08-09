# Choose an MFront backend

For the qualified 316L SRIX + EBSD workflow, use
`mfront-native-generalised-plane-stress` with
`gps_composite_fd_tangent: true`. Use
`mfront-3d-condensed-plane-stress` as the independent numerical reference and
for 3D behaviours without a GPS implementation.

| Backend | Recommended use | Advantage | Limitation |
|---|---|---|---|
| `mfront-native-plane-stress` | simple native 2D laws | direct | not general for 3D crystal plasticity |
| `mfront-3d-condensed-plane-stress` | independent reference / new 3D law | works with any 3D law | Python condensation |
| `mfront-native-generalised-plane-stress` | qualified SRIX GPS workflow | monolithic and performant | requires a GPS law variant |
| `mfront-structural-plane-stress` | generic qualified SRIX workflow | reusable structural closure | limited to the demonstrated V1 MFront contract |
| `python` | historical J2 regression | independent of MFront | not production crystal plasticity |

GPS (*generalised plane stress*) conserve les six composantes 3D et résout
localement les trois déformations hors plan nécessaires pour imposer
`sigma_zz = sigma_xz = sigma_yz = 0` dans le repère global. La variante GPS
SRIX carries this closure in its constitutive Newton. The generic structural
backend applies the same closure through the reusable `StructuralPlaneStress3D`
transformation, while the condensed route performs it in the Python bridge.

Lorsqu'un point GPS doit sous-intégrer un incrément, le dernier sous-pas ne
représente pas à lui seul la dérivée de la trajectoire composée. Le tangent FD
composite reconstruit cette dérivée pour les seuls points concernés. Il est
donc activé dans le workflow SRIX qualifié, avec un coût mesuré et limité.

## Qualified production route

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

## Independent reference

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
```

The first configuration is the qualified production route for SRIX + EBSD. The
second is the independent reference to qualify a new law or check a GPS result.

The generic structural route uses the same qualified host substepping and
composite-tangent policy:

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6
```

It is currently qualified for the small-strain, `Implicit`,
`StandardElasticity`-compatible SRIX workflow. See
{doc}`../reference/numerics/mfront_structural_plane_stress` for the
formulation and its demonstrated scope.

For the details of `R`, the 316L parameters, orientations and per-system
outputs, see {doc}`use_srix_crystal_law`. The reference formulation is detailed
in
{doc}`../reference/numerics/three_dimensional_condensation`.
