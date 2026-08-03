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
from fem_inhouse.core.single_crystal_presets import srix_reference_stress

R = srix_reference_stress(
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

Use a small mesh and few increments, and keep the nonlocal coupling off so it
finishes quickly. A crystal point costs roughly sixteen times a J2 point, so
size the case accordingly and read {doc}`../explanation/forest_rubin_srix` on
cost before scaling anything up.

## What not to do

- Do not present `R` as identified for our 316L. It is transposed from a
  rate-dependent set at an assumed rate.
- Do not compare SRIX against Méric-Cailletaud away from `[001]` and expect
  agreement: measured at 7 % for `[111]` and 14 % for `[123]`.
- Do not copy the `f > 1.1 K` guard from the viscous law. It exists to protect
  a Norton power that SRIX does not have.
