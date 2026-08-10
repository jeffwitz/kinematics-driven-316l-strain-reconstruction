# Nonlocal fixed-point algorithm

**Category: Reference.**

Inside each global Newton evaluation:

1. set $\chi$ from the previous Newton trial or committed increment;
2. update the MFront external state variable;
3. integrate without tangent from the committed state;
4. extract Gauss-point PEEQ and average it to elements;
5. solve Helmholtz for $\chi^\star$ with the DCT operator;
6. relax $\chi\leftarrow(1-\omega)\chi+\omega\chi^\star$;
7. repeat until the coupling criterion is satisfied;
8. integrate once with the consistent fixed-$\chi$ tangent;
9. assemble the mechanical residual and tangent.

No MFront state is committed during the fixed point. Buffers for Kelvin strain,
PEEQ and nonlocal state are preallocated and reused. Complete 3D tensors are
reconstructed only for a final converged output or an explicitly requested
snapshot.

Optional bounded Aitken relaxation changes the convergence path, not the fixed
point. Its use, bounds, accepted/rejected updates and residual history are
diagnostics. The present global tangent is partitioned, not monolithic in
$(u,\chi)$.

## Architectural status

This is the canonical partitioned nonlocal coupling strategy. Production
solvers and robustness references reuse this implementation rather than
reproducing a simplified fixed-point loop.

The state hierarchy is:

```text
committed increment state
        ↓
mechanical Newton trial
        ↓
repeated constitutive/nonlocal trials
```

Every trial is evaluated from the committed state. `evaluate` and `revert`
belong to trial management; only acceptance of the complete global increment
permits `commit`.

### What is not equivalent

The following is not the reference algorithm:

```text
mechanical correction → one Helmholtz update → mechanical correction
```

The reference instead performs:

```text
mechanical Newton evaluation → converged constitutive/nonlocal fixed point
```

A simplified loop may be retained as an explicitly named experimental
candidate, but it must not be called production or reference staggered
coupling without a solution-equivalence qualification.
