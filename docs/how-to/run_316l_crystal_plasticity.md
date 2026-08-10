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

## A one-element check before you touch a real case

If the toolchain is right, this runs as written and prints three finite
stresses. It needs no data files, and it is executed verbatim by the test
suite, so it cannot drift from the API.

```python
import numpy as np
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

batch = create_plane_stress_material_batch(
    "mfront-structural-plane-stress",
    np.full((1, 1), 250.0),          # yield stress map, MPa
    np.full((1, 1), 500.0),          # hardening coefficient map, MPa
    0.245,                            # hardening exponent
    young_modulus_mpa=205000.0,
    poisson_ratio=0.3,
    hardening_mode="ludwik",
    plastic_strain_max=0.2,
    plastic_table_points=1000,
    first_positive_plastic_strain=1e-6,
    mfront_library="build/mfront/src/libBehaviour.so",
    mfront_threads=1,
    mfront_behaviour_id="fcc_forest_rubin_srix",
    constitutive_options={
        "gps_composite_fd_tangent": True,
        "paired_parameter_set": "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        "crystal_orientation": {
            "mode": "homogeneous",
            "euler_bunge_deg": [35.0, 20.0, 15.0],
        },
    },
)

for step in range(1, 9):
    strain = (step / 8) * 0.02 * np.array([1.0, -0.4, 0.0])
    trial = batch.evaluate(np.atleast_2d(strain), time_increment=1.0 / 8)
    batch.commit()

print(trial.stress_in_plane_mpa[0])       # sigma_xx, sigma_yy, sigma_xy in MPa
```

The yield-stress and hardening maps are required by the factory signature but
are not used by the crystal law, whose hardening comes from the registered
parameter set. `mode: homogeneous` puts one orientation everywhere, which is
the right starting point before wiring EBSD.
