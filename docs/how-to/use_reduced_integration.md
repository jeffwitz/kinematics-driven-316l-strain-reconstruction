# Use reduced integration

Use CPS4R only after keeping a CPS4 result as the reference for the same case.

The element algebra, the stabilisation and what its energy diagnostic does and
does not measure are in
{doc}`../explanation/reduced_integration_hourglass`; read it before choosing a
value of `beta`.

## Run a partition

The partition command exposes the reduced formulation and its numerical
controls:

```bash
fem-inhouse partition \
  --input validation/reference_data/<input> \
  --output validation/reference_data/<campaign> \
  --count 25 \
  --element-formulation cps4r \
  --hourglass-scale 1.0 \
  --hourglass-energy-warning-ratio 0.01 \
  --hourglass-energy-failure-ratio 0.05 \
  --partition-id 0
```

`cps4` remains the default. A non-default `hourglass-scale` is refused for
CPS4 because that parameter has no meaning for the fully integrated element.

The reduced formulation is currently incompatible with
`--nonlocal-plasticity`. The solver stops before starting rather than combining
two unvalidated regularisations.

## Read the result

The partition status records:

- the selected element formulation;
- Gauss points per element;
- constitutive material point count;
- accumulated internal work;
- final hourglass energy;
- final hourglass-energy ratio.

A CPS4R partition also stores:

```text
HOURGLASS_ENERGY_BY_ELEMENT.npy
```

on the element grid. Inspect that field beside PEEQ for J2 or accumulated slip
for a crystal law. A low global ratio is not enough if the numerical energy is
concentrated in the same band as the constitutive activity.

Do not compare ratios across runs with different loading paths. The numerator is
the stabilisation energy stored at the final state, the denominator accumulates
along the path, so a longer history lowers the ratio without the element
behaving any better.

**Do not use the ratio as a validity gate.** The qualification campaign found no
relationship between it and the CPS4-to-CPS4R error, globally or element by
element. A low ratio certifies nothing.

## Read this before choosing beta

The sequence this page previously recommended has been run. It is in
`validation/cps4r_qualification_preregistration.md`, and its outcome is in
`validation/cps4r_qualification_results.md`:

- **no value of `beta` met the accuracy criterion**, on either a heterogeneous
  J2 case or a tilted-orientation SRIX case. The plastic-strain error against
  CPS4 ran from 1.9 to 10 percent against a 0.5 percent bound;
- **`beta = 1` was the least accurate value tested**, not the safest. The
  stabilisation keeps the elastic reference while the constitutive tangent
  softens, so at `beta = 1` the hourglass modes stay elastically stiff while
  everything else yields. `beta = 0.1` landed six times closer to CPS4;
- the cost case did hold: 3.7 to 4.8 times on constitutive time, 1.9 to 2.9
  times on total wall time.

So: **CPS4R is not qualified for a scientific elastoplastic campaign, and this
page recommends no value of `beta`.** Use it for exploration, for cost studies,
and for elastic work where the equivalence at `beta = 1` is exact. Keep a CPS4
result as the reference for anything you intend to report.

If you want to move the verdict, the three openings are listed at the end of the
results document: a mesh-convergence study, a stabilisation built on the current
tangent instead of the fixed elastic reference, and an error estimator that
actually predicts the difference.

To re-run the campaign:

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
python scripts/qualify_reduced_integration.py \
  --mesh 32 --crystal-mesh 8 --repeats 5 \
  --output validation/_generated/cps4r_qualification
python scripts/plot_reduced_integration_diagnostic.py
```

Whatever it reports, do not select `hourglass_scale` on an affine test: every
affine field is orthogonal to the stabilised modes, so all values look
equivalent and the comparison is empty.
