# MFront/MGIS constitutive backend

## Scope and current status

MFront is a second constitutive backend for the supported small-strain,
plane-stress J2/Ludwik case study. It is connected to the finite-element
Newton loop but is not yet the default. The migration sequence is:

1. compile the behaviour;
2. validate it independently at material points;
3. connect trial/commit/revert semantics to the Newton loop;
4. compare a reduced DIC subdomain;
5. switch the default only after the declared thresholds pass.

The first four steps are complete. Saved comparisons are under
`validation/reference_data/mfront_material_point_v1` and
`validation/reference_data/mfront_newton_dic_10x10_v1`. The remaining default
switch is blocked by the explicit hardening-law decision described below, not
by the Newton coupling.

## Installed versions on the development machine

The installation follows the official
[TFEL/MFront source procedure](https://thelfer.github.io/tfel/web/install.html)
and the
[MGIS CMake procedure](https://github.com/thelfer/MFrontGenericInterfaceSupport).

| Component | Version/tag | Source commit | Prefix |
|---|---|---|---|
| TFEL/MFront | `TFEL-5.1.0` | `deee4cd77e1f36efd3715f7aad9a673de9c92880` | `/home/jeff/.local` |
| MGIS | `MFrontGenericInterfaceSupport-3.1` | `38dd3082f745f736abbb6629d82e829b91132514` | `/home/jeff/.local` |

Host prerequisites installed through APT include `g++`, `gfortran`, `cmake`,
`python3`, `python3-numpy`, and `pybind11-dev`.

The reproducible configuration is:

```bash
cmake -S /home/jeff/.local/src/tfel-5.1.0 \
  -B /home/jeff/.local/src/tfel-5.1.0-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/jeff/.local \
  -DTFEL_APPEND_VERSION=OFF \
  -Denable-numpy-support=ON \
  -Denable-python=ON \
  -Denable-python-bindings=ON
cmake --build /home/jeff/.local/src/tfel-5.1.0-build --parallel
cmake --install /home/jeff/.local/src/tfel-5.1.0-build

cmake -S /home/jeff/.local/src/mgis-3.1 \
  -B /home/jeff/.local/src/mgis-3.1-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/jeff/.local \
  -DTFEL_DIR=/home/jeff/.local/share/tfel/cmake \
  -Denable-c-bindings=ON \
  -Denable-python-bindings=ON
cmake --build /home/jeff/.local/src/mgis-3.1-build --parallel
cmake --install /home/jeff/.local/src/mgis-3.1-build
```

The official quick procedure suggests version-suffixed libraries. With TFEL
5.1.0 on this machine, `TFEL_APPEND_VERSION=ON` produced an invalid
`TFEL_SUFFIX_FOR_PYTHON_MODULES=5_1_0` macro while compiling the
`mfront.database` Python binding. The only local deviation is therefore
`TFEL_APPEND_VERSION=OFF`; no source patch was applied.

## Activate and verify

The TFEL environment must be sourced in each shell that compiles a behaviour
or imports MGIS:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
mfront --version
tfel-config --version
.venv/bin/python -c \
  'import tfel, mgis.behaviour; print(tfel.getTFELVersion())'
```

The verified TFEL version is `5.1.0`, and both `mgis.behaviour.load` and
`mgis.behaviour.integrate` are available from `.venv`.

## Build the behaviour

```bash
bash scripts/build_mfront_behaviour.sh
```

The output is `build/mfront/src/libBehaviour.so`. The script accepts:

- `TFEL_ENV_FILE` to select another TFEL `env.sh`;
- `MFRONT_BUILD_DIR` to select another generated build directory.

Generated C++ and binaries remain under ignored `build/`; the MFront source is
versioned at `mfront/PixelLudwikJ2Plasticity.mfront`.

## Constitutive contract

The behaviour uses MFront's `StandardElastoViscoPlasticity` brick with:

- hypothesis `PlaneStress`, including the internal axial strain required to
  enforce `S33 = 0`;
- fixed case-study elasticity `E = 205000 MPa`, `nu = 0.3`;
- von Mises criterion and associated plastic flow;
- point-wise material properties `InitialYieldStress`,
  `HardeningCoefficient`, and `HardeningExponent`;
- a finite first Ludwik segment on `0 <= PEEQ <= 1e-6`, followed by the
  analytical power law.

The Python production law instead linearly interpolates 1000 points up to
`PEEQ = 0.2` and clamps beyond that value. The material-point campaign remains
below `0.2` and measures this known discretisation difference explicitly.
Exact replication of all 1000 segments in a spatially parameterised MFront
law remains open before the backend can be called strictly identical.

MGIS stores symmetric tensors in Kelvin notation. The Python adapter converts:

- `[e11, e22, gamma12]` to `[e11, e22, e33, gamma12/sqrt(2)]`;
- `[s11, s22, s33, sqrt(2) s12]` to `[s11, s22, s12]`;
- the 4×4 Kelvin tangent to the 3×3 engineering-shear tangent.

`MFrontMaterialPointBatch.evaluate` is non-committing by default. The explicit
`commit` and `revert` operations are required for a correct global Newton
integration. The constructor also accepts `thread_count`; values greater than
one create an explicit MGIS thread pool.

Inside `run_fem`, every Newton trial is evaluated from the last converged MGIS
state. A new trial automatically discards the previous uncommitted trial,
`commit` is called only after global convergence, and a failed increment calls
`revert` before cutback. The MFront 3×3 consistent operator is passed directly
to the existing CPS4 Gauss-point tangent assembly.

The public configuration fields are:

- `constitutive_backend`: `python` (default) or `mfront`;
- `mfront_library`: generic-interface shared library path;
- `mfront_threads`: size of the explicit MGIS thread pool.

The compiled behaviour currently fixes `E=205000 MPa`, `nu=0.3`, and the first
positive plastic strain at `1e-6`; the solver rejects incompatible MFront
configurations instead of silently mixing material definitions.

## Reproduce the material-point comparison

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/compare_constitutive_backends.py \
  --output results/mfront-material-point-$(date -u +%Y%m%dT%H%M%SZ) \
  --steps 200
```

The command refuses to overwrite a non-empty result directory and saves all
histories, hashes, metrics, thresholds, and a plot.

The versioned v1 comparison passes all declared stress and PEEQ thresholds.
Across the three paths, stress relative L2 errors are `0.227–0.368 %`, maximum
absolute PEEQ errors are `3.02e-5–3.87e-5`, and the tangent discrepancy remains
diagnostic (`1.02–6.39 %`). The complete finite-element comparison below
confirms that both tangents converge to the same global solution.

## Reproduce the complete Newton/DIC comparison

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/compare_fem_backends.py \
  --input data/processed/case-study-10x10 \
  --output results/mfront-newton-dic-10x10 \
  --library build/mfront/src/libBehaviour.so \
  --threads 2
```

The command runs the same real central DIC crop with the analytical Python
Ludwik law and MFront, then saves all six final fields, diagnostics, input and
library hashes, metrics, thresholds, and the decision. It refuses to overwrite
an existing `report.json`.

The versioned v1 campaign converges both backends in 20 increments without
cutback. MFront requires 66 Newton iterations (maximum 4 per increment), versus
84 (maximum 8) for Python. Every pre-declared relative-L∞ threshold passes:

| Field | Relative L∞ |
|---|---:|
| displacement | `4.68e-9` |
| stress | `1.44e-4` |
| total strain | `6.65e-5` |
| plastic strain | `3.26e-4` |
| PEEQ | `3.19e-4` |
| reaction force | `1.74e-4` |

On this small crop, the measured complete-solver times are `1.583 s` for
Python and `0.669 s` for two-thread MFront. This ratio is a functional smoke
measurement, not a production-size performance claim.

An exploratory run preceded threshold fixation. These are therefore transparent
regression/acceptance thresholds for the coupling, not an independent blind
validation criterion. The script saves each backend immediately after its solve
so that a later backend failure cannot waste the completed calculation.

## Constitutive performance benchmark

The versioned benchmark under
`validation/reference_data/mfront_performance_v1` evaluates 200,000
heterogeneous points over 20 increments, including state updates and consistent
tangents. Two repetitions and reversed execution order give:

| Backend | Median time | Relative result |
|---|---:|---:|
| Python/NumPy | `12.347 s` | baseline |
| MFront serial | `13.333 s` | `1.080×` slower |
| MFront, 8 threads | `3.527 s` | `3.500×` faster |

The complete benchmark lasts `1 min 03.24 s` and peaks at `393.45 MiB`. MFront
serial and parallel outputs are identical. This result covers only the
constitutive kernel; the separate end-to-end crop above validates the coupling
but remains too small to predict article-partition performance.
