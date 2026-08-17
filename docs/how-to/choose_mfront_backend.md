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

## Calling it correctly

Four things that are easy to get wrong, each of which has cost real time.

### Always pass `thread_count`. The default is 1 and it is not representative.

`MFrontNativePlaneStressBatch(..., thread_count=1)` is the constructor default,
and every campaign in this repository runs with four. Benchmarked
single-threaded, MFront looks slower than the vectorised NumPy batch, and that
conclusion is an artefact of the default.

Measured at 80 000 material points, identical strains and committed state:

| branch | Python batch | MFront t1 | t2 | t4 | t8 |
|---|---|---|---|---|---|
| elastic | **62.8 ms** | 398.6 | 234.5 | 171.4 | 121.9 |
| plastic | 387.1 ms | 776.0 | 444.0 | 263.0 | **209.1** |

MFront overtakes at four threads on the plastic branch. End to end on the J2
bench at 256 pixels square over eight increments, changing only the backend:
160.6 s against **73.6 s**, so 2.18 on the whole run and 3.5 on the
constitutive term, whose share falls from 71 % to 45 %.

### The crossover is the cost of the local problem

Python keeps a factor of two on the *elastic* branch even at eight threads:
there the per-point work is trivial and batch marshalling dominates, while the
NumPy path is a single vectorised expression over every point. The heavier the
local problem, the more decisively MFront wins — which is why it is unarguable
for crystal plasticity, where the local closure is a joint Newton on 21
unknowns at `2.81 ms` per evaluation, and marginal for elastic J2 at
microseconds per point.

### `python_condensed` is not a Python constitutive law

In `validation/_generated/performance/p43_m100_backend_comparison_latest.json`
the run labelled `python_condensed` has `backend:
mfront-3d-condensed-plane-stress`. **"python" names where the host condensation
loop runs, not where the constitutive law runs.** All three runs in that
comparison are MFront, on SRIX, within 10 % of each other. There is no recorded
benchmark of a Python constitutive law against MFront other than the table
above.

### Conventions differ between batches, and are not interchangeable

`PythonJ2PlaneStressBatch` works in the **engineering (Voigt)** convention: its
tangent at zero strain equals `plane_stress_elasticity` exactly and differs from
the Kelvin stiffness by `mu` on the shear entry. The identification operator and
the spectral solver are **Kelvin** throughout, and `mfront_condensation`
converts explicitly.

Read the reference stiffness from the behaviour rather than rebuilding it from
`E` and `nu`, as `hyperreduction.reference_stiffness_of` does. Chaining the two
conventions without converting is what left the elastic lifting in a dozen
`scripts/*_p43.py` retaining 32 % of the interior equilibrium residual.

### A load that works

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
PYTHONPATH="$HOME/.local/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="$HOME/.local/lib" \
.venv/bin/python your_script.py
```

```python
from pathlib import Path
import numpy as np
from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch

points = 80_000
material = MFrontNativePlaneStressBatch(
    Path("build/mfront/src/libBehaviour.so"),
    np.full(points, 260.0),    # yield stress, MPa
    np.full(points, 900.0),    # Ludwik coefficient, MPa
    np.full(points, 0.32),     # Ludwik exponent
    behaviour_name="PixelLudwikJ2Plasticity",
    thread_count=8,            # never leave this at its default
)
trial = material.evaluate(strain, time_increment=1.0, consistent_tangent=True)
```

Integration goes through a batched `MaterialDataManager`, not point by point, so
the cost is genuine local work rather than per-point interface overhead.

### To find out which behaviours exist, ask the library, not the module

`mgis.behaviour` has **no** `getBehavioursList` — it exposes nothing that
enumerates behaviours, and concluding from a failed guess at its API that the
module is unavailable is a mistake this page exists to prevent. The library
itself is the register:

```bash
nm -D --defined-only build/mfront/src/libBehaviour.so \
  | awk '{print $3}' | grep -E '_getBehaviourType$' \
  | sed -E 's/_getBehaviourType$//' | sort -u
```

which currently answers:

```text
Fcc316LForestRubinSrix                          Fcc316LMericCailletaud
Fcc316LForestRubinSrixGps                       Fcc316LMericCailletaudStructuralPlaneStress
Fcc316LForestRubinSrixStructuralPlaneStress     PixelLudwikJ2Plasticity
PixelMicromorphicLudwikJ2Plasticity             PixelLudwikJ2Plasticity3D
PixelMicromorphicLudwikJ2Plasticity3D
```

Note that the crystal-plasticity behaviours are **already compiled and present**
— `Fcc316LForestRubinSrix` is the law whose local cost of `2.81 ms` per point is
what will eventually make a reduced integration domain worth building. To check
one loads, call `mgis.behaviour.load(library, name, hypothesis)` and let it
raise; there is no list to consult first.
