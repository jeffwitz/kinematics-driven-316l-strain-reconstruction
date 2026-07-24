# How MFront is coupled to global Newton

## Responsibilities are deliberately separated

The Python finite-element code owns:

- the structured mesh and CPS4 kinematics;
- prescribed DIC boundary conditions;
- global residual and sparse tangent assembly;
- Newton increments, cutbacks, and convergence;
- PyPardiso linear solves;
- output fields and provenance.

MFront owns, independently at every Gauss point:

- elastic/plastic state variables;
- J2 yield detection and return integration;
- stress update;
- PEEQ update;
- the consistent algorithmic tangent.

MGIS is the generic-interface bridge between both layers.

## Behaviour definition

The versioned file `mfront/PixelLudwikJ2Plasticity.mfront` declares:

```text
@DSL Implicit
@Behaviour PixelLudwikJ2Plasticity
@ModellingHypothesis PlaneStress
@Algorithm NewtonRaphson
```

It uses the `StandardElastoViscoPlasticity` brick with Hooke elasticity, a
Mises criterion, associative plastic flow, and user-defined isotropic
hardening.

Material properties passed per point are:

- `InitialYieldStress`;
- `HardeningCoefficient`;
- `HardeningExponent`.

The behaviour is compiled with the MFront generic interface into
`build/mfront/src/libBehaviour.so`.

## Tensor conventions

The FE kernel stores:

```text
strain: [e11, e22, gamma12]
stress: [s11, s22, s12]
```

MGIS represents symmetric tensors in Kelvin notation. Under the MFront
plane-stress hypothesis, the adapter converts:

$$
[e_{11},e_{22},\gamma_{12}]
\longrightarrow
[e_{11},e_{22},e_{33},\gamma_{12}/\sqrt{2}],
$$

and converts stress back from

$$
[s_{11},s_{22},s_{33},\sqrt{2}s_{12}]
\longrightarrow
[s_{11},s_{22},s_{12}].
$$

The $4\times4$ Kelvin tangent is reduced and scaled to the $3\times3$
engineering-shear tangent expected by CPS4 assembly. These factors are part of
the constitutive contract and are tested explicitly.

## Trial, commit, and revert

Global Newton evaluates several trial states before accepting one increment.
Committing every constitutive call would accumulate plastic strain from
rejected iterations and produce a path-dependent numerical error.

`MFrontMaterialPointBatch` therefore exposes three operations:

`evaluate`
: Integrate a trial total strain from the last converged state. Do not modify
  the committed state.

`commit`
: Promote the converged trial to the starting state of the next increment.

`revert`
: Discard the trial and return to the last converged state.

The global algorithm is conceptually:

```text
for each nominal pseudo-time increment:
    choose trial step size
    repeat Newton iterations:
        impose boundary displacement at trial pseudo-time
        compute Gauss-point total strain
        evaluate MFront from last committed state
        assemble residual and consistent tangent
        if residual converged:
            commit MFront state
            accept increment
            break
        solve tangent * correction = -residual with PyPardiso
        update free displacement
    if Newton did not converge:
        revert MFront state
        reduce step size
```

A new `evaluate` call automatically discards an older uncommitted trial. A
cutback never carries rejected plasticity into the smaller retry.

## Convergence criterion

The solve uses a relative residual when the reference force scale is
well-defined and records the final criterion name, norm, and relative value.
The nominal tolerance is `1e-6`; the validated long run reached
`2.207e-8`.

The step may be reduced until `1/1024` of the nominal increment. Exhausting the
Newton limit or minimum step is an error, not a partially successful result.

## Parallel constitutive integration

`mfront_threads > 1` creates an explicit MGIS thread pool over material points.
The state update remains deterministic: the saved serial and eight-thread
benchmark states are identical.

This pool does not replace MKL threading. Two independent parallel layers may
be active:

- MGIS threads for constitutive points;
- MKL threads for sparse linear algebra.

They should be allocated deliberately to avoid oversubscription.

## Performance and memory interpretation

On the article-sized corner partition, MFront reduced constitutive time from
`575.906 s` to `83.409 s` and complete process wall time from `1089.80 s` to
`650.08 s`.

Peak process RSS increased by 10.49%, even though the MFront path never builds
the Python 1000-point table. MGIS state/tangent arrays, sparse FE matrices, and
PyPardiso dominate the full-process peak. Removing an unnecessary table and
reducing peak memory are therefore separate claims; only the first is
established by implementation.

## Why MFront is now the default

The switch followed four validations:

1. elastic and plastic material-point paths;
2. stress, PEEQ, and tangent comparisons;
3. a complete homogeneous MFront/Newton solve;
4. a six-field comparison on a real DIC crop.

The Python implementation remains valuable as an independent regression
reference and historical Abaqus-table mode. It is no longer the nominal model.

