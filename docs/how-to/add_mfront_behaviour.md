# Add an MFront behaviour

**Category: How-to.**

## Choose the adapter

Use native `PlaneStress` for a behaviour that supplies the required axial
state and condensed tangent. Use `Tridimensional` plus the condensation
adapter for a general six-component law, including future crystal plasticity.

The global solver no longer needs a new conditional branch for each law. A law
with the existing Ludwik J2 field contract can be added to `MFRONT_BEHAVIOURS`.
A law with a different state, such as crystal plasticity, must register a
constitutive plugin that returns a `PlaneStressMaterialBatch`.

## Describe the MFront contract

Create an `MFrontBehaviourSpec` containing:

- the exact behaviour name for each available modelling hypothesis;
- material-property, external-state and internal-state entry names;
- the verified tangent matrix type;
- whether a rotation matrix is required;
- a bridge-profile identifier.

The catalogue is deliberately declarative. It lets the application reject an
unsupported hypothesis or a missing nonlocal field before starting a costly FEM
solve. The built-in entries are `ludwik_j2` and `micromorphic_ludwik_j2`, whose
behaviour names are exactly those the solver used before the catalogue existed.

## Register a constitutive plugin

For a law whose state contract differs from J2, register one builder during
application start-up:

```python
from fem_inhouse.core.constitutive_plugins import register_constitutive_plugin


def build_fcc_batch(request):
    return FCCCrystalPlasticityPlaneStressBatch(
        request.mfront_library,
        point_count=len(request.initial_yield_stress_mpa),
        thread_count=request.mfront_threads,
        local_plane_stress_options=request.local_plane_stress_options,
        **request.options,
    )


register_constitutive_plugin("fcc_crystal_plasticity", build_fcc_batch)
```

An installed package may instead publish the builder in the Python entry-point
group `fem_inhouse.constitutive_plugins`. Entry-point names become backend
identifiers and are discovered automatically on first use.

Then select `constitutive_backend: fcc_crystal_plasticity`, on the command line
or in the configuration, and place law-specific entries such as slip-system
parameters or the EBSD orientation source under `constitutive_options`.

The returned batch must implement `evaluate_in_plane`, `complete_trial`,
`commit` and `revert`. It also declares the tangent matrix type. Newton does not
need to know whether the state comes from J2, crystal plasticity or another
MFront behaviour.

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

For crystal plasticity, start from the `Tridimensional` bridge with local
plane-stress condensation. Native `PlaneStress` is an optional optimisation,
not a prerequisite. Keep orientation assignment in the constitutive plugin so
that EBSD-to-Gauss-point conventions are tested independently of Newton.

See {doc}`../reference/numerics/mfront_transaction`,
{doc}`../reference/numerics/three_dimensional_condensation` and
{doc}`../reference/numerics/sparse_solver`.
