# Add an MFront behaviour

**Category: How-to.**

## Choose the adapter

Use native `PlaneStress` for a behaviour that directly supplies the required
two-dimensional contract. Use `mfront-3d-condensed-plane-stress` as the
independent reference for a general six-component law. For a small-strain
`Implicit`/`StandardElasticity`-compatible three-dimensional behaviour, use
`mfront-structural-plane-stress`; this is the registered route used by the
qualified crystal-plasticity behaviours.

The global solver does not need a new conditional branch for each supported
law. Add an `MFrontBehaviourSpec` to the catalogue, including the raw 3D
behaviour and, when the V1 structural contract is satisfied, its generated
structural variant. A separate constitutive plugin is only required when the
law's state or output contract cannot be represented by the registered
adapters.

## Describe the MFront contract

Create an `MFrontBehaviourSpec` containing:

- the exact behaviour name for each available modelling hypothesis;
- material-property, external-state and internal-state entry names;
- the verified tangent matrix type;
- whether a rotation matrix is required;
- a bridge-profile identifier.

The catalogue is deliberately declarative. It lets the application reject an
unsupported hypothesis or a missing nonlocal field before starting a costly FEM
solve. The built-in entries include the J2, SRIX and Méric behaviours; their
identifiers and compiled MFront names are kept explicit in the catalogue.

## Register a constitutive plugin when needed

For a law whose state or output contract is not covered by the registered
MFront adapters, register one builder during application start-up:

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

For crystal plasticity satisfying the structural V1 contract, add the raw
three-dimensional behaviour and its generated structural variant to the
catalogue, then select `mfront-structural-plane-stress`. Keep orientation
assignment in the standard MFront adapter so that EBSD-to-Gauss-point
conventions are tested independently of Newton. Use
`mfront-3d-condensed-plane-stress` as the independent constitutive reference.

See {doc}`../reference/numerics/mfront_transaction`,
{doc}`../reference/numerics/three_dimensional_condensation` and
{doc}`../reference/numerics/sparse_solver`.
