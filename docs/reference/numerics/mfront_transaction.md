# MFront transaction protocol

**Category: Reference.**

Three state levels are distinguished:

1. committed state at the last converged increment;
2. trial state for the current global Newton evaluation;
3. repeated local evaluations used by condensation or micromorphic coupling.

`evaluate` always starts from the committed state and writes a trial state.
`commit` is called once after global increment convergence. `revert` restores
the committed state after failure or cutback.

Micromorphic fixed-point evaluations update the external $\chi$ field but do
not commit plastic state. The final constitutive evaluation at converged
$\chi$ supplies the stress and tangent assembled by Newton.

MGIS variable names, types, sizes and offsets are validated from behaviour
metadata during adapter construction. Missing native variables are errors
unless the selected behaviour explicitly declares an analytical completion
strategy.

## Nested coupling rule

A nested nonlocal fixed point may perform an arbitrary number of MFront trial
integrations. Every trial belongs to the current global increment and must be
revertible to the last committed state.

Neither a successful nonlocal iteration nor a successful mechanical Newton
iteration is a material commit. Only acceptance of the complete global load
increment permits `commit()`.

The following events must restore the accepted state before a retry:

- rejected line search;
- failed Newton or GMRES solve;
- failed nonlocal fixed point;
- constitutive inadmissibility;
- global cutback.

This transaction contract is shared by J2, SRIX and Méric–Cailletaud.
