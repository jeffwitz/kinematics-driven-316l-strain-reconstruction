# Transient Jacobian storage feasibility under TFEL 5.1

The generic shell currently transports the converged structural Jacobian from
`@Integrator` to `@TangentOperator` through an auxiliary state variable named
`StructuralJacobian`. The prototype uses `324 = 18 x 18`, which is a known
SRIX/Méric limitation and is not an acceptable final generic representation.

The first attempted replacement was:

```mfront
@Private {
  decltype(jacobian) structuralJacobian;
};
```

TFEL 5.1 rejects this with `invalid use of 'this' at top level` (and the same
failure for `decltype(jacobian)`). In the installed DSL, `@Private` is emitted
as namespace-level private code, not as a private data member of the generated
behaviour class. It therefore cannot access the generated `jacobian` member or
carry one independent matrix per material-point instance.

The qualified generator has consequently been restored to its last working
transport. No claim of a dimension-safe transient buffer is made. The viable
next options are:

1. a first-class MFront/TFEL hook that exposes the assembled Jacobian to the
   tangent-operator phase;
2. a generic DSL-supported transient member/storage facility;
3. reconstructing the converged Jacobian at the tangent hook, provided the
   lifecycle exposes the pre-update constitutive state generically.

The failure is architectural rather than mathematical: the live same-state
Schur qualification remains valid and is independent of this storage choice.
