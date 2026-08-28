# Run a FEMU identification

**Mode:** how-to  
**Domain:** identification

Declare the observable field, parameter perturbations, forward backend and
stopping criteria before launching FEMU. Archive the sensitivity matrix and
singular values; interpret weak modes before reporting parameter estimates.

For the currently registered smoke workflow, use
`scripts/srix_femu_smoke.py` with the preregistered case and the declared
`tau0`/`Q` perturbations. The driver performs complete forward evaluations and
records objective, parameter, backend and provenance data. Check that the
kinematics are imposed on the boundary rather than on every interior node;
the latter is a deliberate negative control because it removes constitutive
sensitivity.

Definitions and required provenance are in
{doc}`../../reference/numerics/femu_sensitivity_and_svd`; interpretation is in
{doc}`../../explanation/identification/femu_identification`.
