# MFront/MGIS constitutive backend

## Scope and current status

MFront is being introduced as a second constitutive backend for the supported
small-strain, plane-stress J2/Ludwik case study. It is not yet the default
finite-element backend. The migration sequence is deliberately:

1. compile the behaviour;
2. validate it independently at material points;
3. connect trial/commit/revert semantics to the Newton loop;
4. compare a reduced DIC subdomain;
5. switch the default only after the declared thresholds pass.

The first two steps are complete. The saved comparison is under
`validation/reference_data/mfront_material_point_v1`.

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
integration.

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
diagnostic (`1.02–6.39 %`). The latter must be investigated during the
finite-element integration rather than silently accepted.
