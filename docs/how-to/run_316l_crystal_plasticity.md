# Run 316L crystal plasticity: the short path

**Category: How-to.** For someone who wants to run the qualified 316L
crystal-plasticity workflow and has never used MGIS or MFront. Nothing here
asks you to understand the constitutive machinery; the pages it points to do
that when you want it.

## What you are about to run

A rate-independent FCC crystal-plasticity law (SRIX) on a thin sheet, with one
crystal orientation per pixel taken from EBSD. Because the sheet is thin, the
solver works in **plane stress**, and because the law is written in three
dimensions, something has to impose that condition. There are two ways of doing
it and the choice is the one real decision on this page. Take the recommended
one; {doc}`choose_mfront_backend` explains it if you want the reasoning.

## 1. Build the constitutive library

The laws are C++ sources compiled by MFront into one shared library:

```bash
./scripts/build_mfront_behaviour.sh
```

It prints the path of `libBehaviour.so`. Export it — every command below reads
it from the environment:

```bash
export MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so"
```

If the script cannot find `mfront`, source the TFEL environment first
(`source ~/.local/share/tfel/env/env.sh`) and see {doc}`install`.

## 2. Write the configuration

```yaml
solver:
  constitutive_backend: mfront-native-generalised-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  increments: 8

  constitutive_options:
    gps_composite_fd_tangent: true
    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3
    crystal_orientation:
      mode: ebsd
      euler_bunge_deg: [...]        # (nx, ny, 3) array, degrees
```

Line by line, since three of these are not obvious:

`constitutive_backend` — **the decision**. This value runs the 3D law with the
plane-stress condition solved inside it. Use
`mfront-3d-condensed-plane-stress` instead if your law has no GPS variant, or
when you want an independent check of a GPS result.

`gps_composite_fd_tangent: true` — leave it on. A handful of deeply plastic
points cannot swallow a full load increment and get integrated in sub-steps;
this repairs the derivative they hand back. Without it the same run needs 85
Newton iterations instead of 58 and is slower than the reference.

`parameter_set` — the 316L parameters. They are **not identified on this
material**: they are transposed from a published Méric-Cailletaud set. The
registry records that provenance, and no result here should be read as an
identification. See {doc}`use_srix_crystal_law` before using them in a claim.

`mfront_threads` — MGIS integrates the material points on this many threads.
Four is what the archived campaigns use.

`crystal_orientation` — Bunge Euler angles in degrees, one triple per pixel,
shaped `(nx, ny, 3)`. `mode: homogeneous` with a single triple works too, and
is the right starting point for a first run.

## 3. Run it

A one-element sanity check, no data files needed, straight from Python:

```python
import numpy as np
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

batch = create_plane_stress_material_batch(
    "mfront-native-generalised-plane-stress",
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
        "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3",
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

If that prints three finite stresses, your toolchain is correct and the
configuration above will work on a real case.

The registered polycrystal campaign is driven by

```bash
python scripts/qualify_crystal_tet2_p43.py --help
```

whose flags mirror the YAML keys one for one.

## 4. Check what actually happened

The solver diagnostics record the backend and, for the GPS route, how much
sub-stepping was needed:

| field | reading |
|---|---|
| `native_substep_points` | points integrated in sub-steps. `0` means the composite repair changed nothing |
| `composite_fd_points` | points whose derivative was rebuilt |
| `newton_iterations` | compare against a reference run before trusting a speed claim |

## What to read next, in order

1. {doc}`choose_mfront_backend` — the two plane-stress routes, and their cost.
2. {doc}`use_srix_crystal_law` — `R`, the 316L parameters and their provenance,
   per-slip-system outputs.
3. {doc}`../reference/configuration` — every configuration key.
4. {doc}`../explanation/forest_rubin_srix` — what the law actually is.

## Two things that will bite you

**The parameters are not identified 316L.** Nothing in this repository claims
they are. A number produced with them is a numerical result, not a material
property.

**Plane stress is imposed in the global frame, not the crystal frame.** The free
surface is normal to `z` in the sheet, and for a tilted grain that condition
mixes all six components of the crystal-frame stress. Both backends handle it;
a hand-rolled closure usually does not.
