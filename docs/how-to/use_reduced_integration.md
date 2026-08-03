# Use reduced integration

Use CPS4R only after keeping a CPS4 result as the reference for the same case.

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

## Qualify a value of beta

Do not select `hourglass_scale` on an affine test. Every affine field is
orthogonal to the stabilised modes, so all values appear equivalent.

Use this sequence:

1. run a small non-affine elastic case at `beta=1`;
2. verify that CPS4 and CPS4R agree while hourglass energy is nonzero;
3. run a small non-affine plastic J2 comparison for
   `beta = 0.1, 0.25, 0.5, 1.0`;
4. retain at most two candidates using displacement, reactions, plastic-field
   differences and the spatial hourglass-energy map;
5. compare those candidates with CPS4 using SRIX and one homogeneous tilted
   orientation;
6. measure constitutive and total wall time;
7. only then consider an experimental ROI.

CPS4 remains the scientific reference whenever the CPS4R sensitivity is not
bounded.
