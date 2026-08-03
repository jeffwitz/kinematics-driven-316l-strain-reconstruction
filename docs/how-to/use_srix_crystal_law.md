# How to use the SRIX crystal law

**Category: How-to.** Selecting the rate-independent FCC law, choosing `R`, and
checking at a material point that it behaves. The reasoning behind the law is
in {doc}`../explanation/forest_rubin_srix`.

## Build the behaviours

Both crystal laws are in `mfront/` and are compiled by the usual script:

```bash
./scripts/build_mfront_behaviour.sh
```

It prints the path of `libBehaviour.so`, which must be exported as
`MFRONT_BEHAVIOUR_LIBRARY` for anything below to run.

## Select the law

The behaviour identifiers are `fcc_forest_rubin_srix` and, for the
rate-dependent reference, `fcc_meric_cailletaud`:

```bash
--constitutive-backend mfront --mfront-behaviour-id fcc_forest_rubin_srix
```

Neither declares a native plane-stress hypothesis, so the solver condenses the
3D law. That is deliberate; see the explanation page.

## Choose or compute R

`R` is not a measured property. Compute it from a Méric-Cailletaud pair and an
explicit reference strain rate:

```python
from fem_inhouse.core.single_crystal_presets import srix_overstress_modulus_from_meric

R = srix_overstress_modulus_from_meric(
    norton_strength_mpa=12.0,
    norton_exponent=11.0,
    reference_strain_rate=1.0e-3,
)  # 18.7819100705 MPa
```

There is no default rate, on purpose. Pass the strain rate of the experiment
you are modelling, and record it alongside any result that depends on it.

The registered preset carries the same computation plus its attribution:

```python
from fem_inhouse.core.single_crystal_presets import get_srix_preset

preset = get_srix_preset("316l_forest_rubin_srix_from_nasri2018")
preset.mfront_parameters()   # R with the inherited hardening; no K, no n
preset.provenance_record()   # both citations, the rate, and the status field
```

## Run a material-point check

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
PYTHONPATH="$HOME/.local/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="$HOME/.local/lib" \
.venv/bin/python -m pytest tests/unit/core/test_forest_rubin_srix.py -q
```

Without `MFRONT_BEHAVIOUR_LIBRARY` the MGIS tests skip and only the conversion
utility is exercised, so check the summary reports no skips.

## Verify independence from `dt`

The test that matters is `test_srix_is_time_independent`: one strain path over
total times spanning a factor of a million, requiring bit-for-bit equality of
stresses, internal variables and tangent.

```bash
.venv/bin/python -m pytest tests/unit/core/test_forest_rubin_srix.py \
  -k "time_independent" -q
```

Run its control, `test_meric_cailletaud_is_not_time_independent`, at the same
time. If both pass, the harness distinguishes the two laws; if the control also
reports independence, the harness is broken and the first result means nothing.

## Run a small plane-stress case

The crystal laws have no native plane-stress hypothesis, so the 3D behaviour is
condensed. Select the condensed backend and, optionally, an orientation:

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  constitutive_options:
    crystal_orientation:
      mode: homogeneous
      matrix:
        - [1.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0]
        - [0.0, 0.0, 1.0]
```

`euler_bunge_deg: [phi1, Phi, phi2]` may be given instead of `matrix`. Omitting
`crystal_orientation` entirely means the identity: crystal axes aligned with the
specimen axes.

The matrix is `Q_global_to_material`, so `eps_crystal = Q eps_global Q^T`. The
plane-stress condition is imposed in the GLOBAL frame, and all three out-of-plane
components are solved for, because an arbitrary orientation couples the normal
and the out-of-plane shears.

A homogeneous orientation is a validation step, **not a polycrystal**: every
point being the same crystal is exactly what a real aggregate is not. Assigning
EBSD orientations to Gauss points is the next step; the bridge already accepts an
`(n_points, 3, 3)` array, so that will be a new provider and not a new bridge.

Keep the mesh small and the nonlocal coupling off. A crystal point costs roughly
sixteen times a J2 point; read {doc}`../explanation/forest_rubin_srix` on cost
before scaling anything up.

### What the crystal does not produce

There is no equivalent plastic strain. `FEMResult.equivalent_plastic_strain`
stays at zero for these behaviours rather than being filled with a substitute,
and the micromorphic coupling, which is defined on a J2 PEEQ, refuses them
outright. The twelve-component families -- `plastic_slip`,
`equivalent_plastic_slip`, `back_strain` -- and the scalar `accumulated_slip`
are available on the material batch.

## What not to do

- Do not present `R` as identified for our 316L. It is transposed from a
  rate-dependent set at an assumed rate.
- Do not compare SRIX against Méric-Cailletaud away from `[001]` and expect
  agreement: measured at 7 % for `[111]` and 14 % for `[123]`.
- Do not copy the `f > 1.1 K` guard from the viscous law. It exists to protect
  a Norton power that SRIX does not have.

## Select a registered parameter set

Every SRIX parameter is an MFront `@Parameter`, so a set is applied at run time
and nothing is recompiled:

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  constitutive_options:
    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3
    crystal_orientation:
      mode: homogeneous
      euler_bunge_deg: [0.0, 0.0, 0.0]
```

Individual values may be overridden on top of a set with a `parameters` block;
an inline value demotes its provenance group to `exploratory`, because nothing
knows where it came from. Unknown set identifiers and unknown parameter names
are refused before the first solve. The registered sets, their values and their
statuses are in {doc}`../reference/srix_parameter_sets`.

Selecting nothing applies the historical set, so an unconfigured run reproduces
every archived result.

## Read the crystal state back

A crystal result carries what a J2 one cannot:

```python
result.plastic_slip             # (nx, ny, 12) signed slip
result.equivalent_plastic_slip  # (nx, ny, 12) accumulated slip
result.back_strain              # (nx, ny, 12)
result.cumulated_slip           # (nx, ny) sum of the twelve
result.active_slip_systems      # (nx, ny)
```

`result.equivalent_plastic_strain` stays at **zero** for these runs and must not
be used. The sum of twelve accumulated slips is a different scalar with a
different definition; `cumulated_slip` carries it under its own name so the two
are never confused.
