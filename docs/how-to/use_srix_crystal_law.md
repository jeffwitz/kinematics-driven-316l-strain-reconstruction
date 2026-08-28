# Use the 316L Forest–Rubin SRIX law

This how-to describes the current production route for the FCC 316L SRIX
crystal-plasticity behaviour. The law uses twelve FCC slip systems, a cubic
elastic backbone, and per-point crystal orientations.

## Recommended route

Use the registered structural plane-stress backend for a two-dimensional EBSD
calculation:

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

The source case supplies the orientation file and its co-registration with the
mesh. The factory assigns one orientation to each material point. Do not
rotate the input gradient externally: the structural adapter and behaviour
share the repository frame convention.

Use the external 3D condensation route as an independent reference:

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

## Material parameters

Use a registered paired parameter set so that SRIX and Méric–Cailletaud share
the same 316L elastic and interaction-matrix provenance when they are compared.
The SRIX-specific parameter registry contains the overstress modulus, initial
slip resistance, isotropic hardening, dynamic recovery, backstrain modulus,
and elastic constants. The selected set is recorded in the run provenance.

The production law uses the sharp Macaulay and absolute-value expressions. At
exactly zero slip increment, its local Newton Jacobian uses the symmetric
Clarke element for `d|dg|/ddg`; this is not a user calibration parameter. See
{doc}`../reference/numerics/srix_semismooth_jacobian`.

## Orientations and outputs

With `mode: ebsd`, the orientation provider reads the case's co-registered
Euler angles and assigns a material orientation to every Gauss point. The
orientation convention is Bunge, in degrees, and follows the contract in
{doc}`../reference/scientific/ebsd_orientation_contract`.

SRIX does not provide a scalar J2 equivalent plastic strain. Use the twelve
signed slip increments, accumulated absolute slip, and system-wise slip
quantities for crystal-plasticity interpretation. The structural backend also
reports the relaxed transverse strains
\(\varepsilon_{zz},\gamma_{xz},\gamma_{yz}\).

## Running and checking a case

The repository's case runner should be used through the configured factory.
For a direct qualification or small test, the corresponding command-line
driver is:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/qualify_crystal_tet2_p43.py \
  --behaviour fcc_forest_rubin_srix \
  --material-backend mfront-structural-plane-stress \
  --paired-parameter-set 316l_guilhem2013_nasri2018_meric_srix_rate_1e-3 \
  --mfront-threads 4 \
  --gps-composite-fd-tangent \
  --output validation/_generated/performance/srix_run.json
```

Record the final residual, global Newton iterations, orientations, behaviour
identifier, parameter set, library, and thread settings. For an independent
check, rerun the same case with `mfront-3d-condensed-plane-stress` and compare
displacements, reactions, stresses, and the twelve slip fields.

For the mathematical definition of the structural closure, see
{doc}`../reference/numerics/mfront_structural_plane_stress`. For the complete
backend selection table, see {doc}`choose_mfront_backend`.
