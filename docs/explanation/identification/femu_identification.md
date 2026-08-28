# FEMU and full-field identification

**Mode:** explanation  
**Domain:** identification

FEMU embeds the constitutive law in the equilibrium forward problem and
compares predicted displacement or strain fields with observations. Each
parameter perturbation may require a complete forward solve, so the cost of
the material and global Newton/Krylov layers directly controls the practical
identification budget.

The objective is evaluated after applying the observation operator to the
mechanical solution,

$$J(\theta)=\tfrac12\|W[O(u(\theta))-y^{obs}]\|_2^2.$$

The measured displacement is therefore a boundary datum or an observable,
not an interior constitutive state to impose everywhere. Full-field Dirichlet
imposition makes the interior displacement insensitive to the law and is a
useful negative control, not a valid identification experiment.

The native SRIX architecture reduces this cost while preserving an MFront
oracle; it does not make an unobservable parameter identifiable.

The recorded smoke driver is `scripts/srix_femu_smoke.py`. It demonstrates a
wiring limitation (full-field prescribed displacement), so its flat fit must
not be reported as parameter identification. Boundary-only FEMU and the
sensitivity/SVD definitions are documented in
{doc}`../../reference/numerics/femu_sensitivity_and_svd`.
