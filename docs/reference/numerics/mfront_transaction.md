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
