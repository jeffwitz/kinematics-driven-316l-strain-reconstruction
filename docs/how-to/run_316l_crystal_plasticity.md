# Run the 316L crystal-plasticity workflow

This is the shortest path from a configured checkout to a two-dimensional
316L SRIX calculation with EBSD orientations.

## Build the MFront library

```bash
source ~/.local/share/tfel/env/env.sh
./scripts/build_mfront_behaviour.sh
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
```

## Configure the production backend

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
```

The case provides the co-registered EBSD orientation source. For a homogeneous
control, replace `mode: ebsd` with the case's homogeneous orientation mapping.
The external reference is `mfront-3d-condensed-plane-stress` with the same
behaviour and parameter set.

## Run and inspect a case

Use the case-specific driver or the configured application entry point. The
qualification driver exposes its complete interface through:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/qualify_crystal_tet2_p43.py --help
```

The result provenance should include the behaviour, backend, parameter set,
orientation source, library, MFront thread count, Krylov settings, final
residual, and Newton iteration history.

Check the following before interpreting slip maps:

- the EBSD orientation source is co-registered with the mesh;
- the input gradient is supplied in the structural/global frame;
- SRIX outputs are interpreted as twelve signed slip fields and accumulated
  absolute slip, not as a native J2 equivalent plastic strain;
- the independent 3D condensation route agrees on the declared observables
  when it is used as a reference.

For backend selection see {doc}`choose_mfront_backend`; for the material law
see {doc}`use_srix_crystal_law`; for the structural closure see
{doc}`../reference/numerics/mfront_structural_plane_stress`.
