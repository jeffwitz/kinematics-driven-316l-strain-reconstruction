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

The relevant experimental sensitivity is the DIC-weighted map
{doc}`dic_observable_femu_modes`: `S_obs = W_DIC O S`.  It uses the measured
transfer function and repeated-frame uncertainty, rather than treating DIC as
an unspecified smoothing filter.

## What is already demonstrated

The direct sensitivity machinery has a positive synthetic qualification. In
`E-SRIX-P43-SYNTH-003`, a P43 M100 case with real EBSD, identity observation,
no noise and the registered 32-step path converges in three evaluations after
initialisation from the best M20 result. The recovered parameters reach the
synthetic truth to numerical precision, while the fourth singular direction is
weak and the `Q`/`b` combination remains strongly correlated. This demonstrates
the FEMU machinery and one synthetic scale-up; it does not identify experimental
316L parameters.

The evidence ladder is therefore:

```text
direct sensitivities       supported
synthetic M20 identification demonstrated
synthetic M20 -> M100      demonstrated for one registered run
experimental P43 modes     limited / under study
production boundary-only   not registered
```

The recorded smoke driver is `scripts/srix_femu_smoke.py`. It demonstrates a
wiring limitation (full-field prescribed displacement), so its flat fit must
not be reported as parameter identification. Boundary-only FEMU and the
sensitivity/SVD definitions are documented in
{doc}`../../reference/numerics/femu_sensitivity_and_svd`.
