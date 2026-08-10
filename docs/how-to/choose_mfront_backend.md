# Choose an MFront backend

For the qualified 316L SRIX + EBSD workflow, use
`mfront-structural-plane-stress` with `gps_composite_fd_tangent: true`.
Use `mfront-3d-condensed-plane-stress` as the independent numerical reference
and for three-dimensional behaviours without a structural backend.

| Backend | Use it for | Main advantage | Limitation |
|---|---|---|---|
| `mfront-native-plane-stress` | Native two-dimensional laws | Direct 2D constitutive response | Not a general 3D crystal closure |
| `mfront-3d-condensed-plane-stress` | Reference calculations and any compatible 3D law | Independent external closure | Local condensation is performed by the host |
| `mfront-structural-plane-stress` | Registered `StructuralPlaneStress3D` behaviours | Generic three-traction closure with the 3D state retained | Behaviour must satisfy the V1 contract |
| `mfront-native-generalised-plane-stress` | The specialised legacy GPS behaviour | Direct specialised SRIX GPS implementation | Use only when reproducing that registered behaviour explicitly |
| `python` | Historical J2 regression workflows | Independent of MFront | Not the production crystal-plasticity route |

## Qualified SRIX/EBSD configuration

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6
    paired_parameter_set: 316l_guilhem2013_nasri2018_meric_srix_rate_1e-3
    crystal_orientation:
      mode: ebsd
      # orientation source is defined by the case
```

The structural backend performs the local three-traction closure. The optional
composite tangent is a host-side derivative for points that are integrated by
multiple constitutive substeps.

## Independent 3D reference

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    paired_parameter_set: 316l_guilhem2013_nasri2018_meric_srix_rate_1e-3
    crystal_orientation:
      mode: ebsd
```

Keep the MFront, Krylov-BLAS, and FFTW thread settings explicit in benchmark
reports. See {doc}`../reference/configuration` for the complete configuration
contract and {doc}`../reference/numerics/mfront_structural_plane_stress` for
the formulation and its validity domain.
