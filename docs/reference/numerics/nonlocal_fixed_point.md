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
