# Add an MFront behaviour

**Category: How-to.**

## Choose the adapter

Use native `PlaneStress` for a behaviour that supplies the required axial
state and condensed tangent. Use `Tridimensional` plus the condensation
adapter for a general six-component law, including future crystal plasticity.

## Implement and compile

Add the behaviour under `mfront/`, declare material properties, external state
variables and required internal observables, then compile the project MFront
library. Do not alter an existing reference behaviour to prototype a new law.

## Verify metadata

At adapter construction, verify hypothesis, gradient and force sizes,
component ordering, variable names, offsets and tangent type from MGIS.
Never assume that axial strain is stored in a generic gradient slot.

## Implement transactions

Provide trial evaluation, commit and revert. Repeated local condensation or
micromorphic iterations must restart from the same committed state and must
not accumulate plasticity.

## Validate

1. material-point histories;
2. tangent by finite differences away from branch points;
3. transverse residuals;
4. native versus condensed J2 regression where applicable;
5. small homogeneous FEM;
6. DIC-driven reduced case;
7. one representative localized region.

Declare a symmetric matrix capability only after measuring tangent symmetry.
Otherwise retain full CSR and PARDISO `mtype=11`.

See {doc}`../reference/numerics/mfront_transaction`,
{doc}`../reference/numerics/three_dimensional_condensation` and
{doc}`../reference/numerics/sparse_solver`.
