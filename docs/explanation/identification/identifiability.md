# Identifiability and observable modes

**Mode:** explanation  
**Domain:** identification

The sensitivity matrix maps parameter perturbations to measured observables.
Its singular values separate strong, weak and null combinations of parameters.
An apparently good fit can therefore constrain only a low-dimensional
combination. SVD is used to report that observable subspace rather than to
claim independent identification of every parameter.

If (S=U\Sigma V^T), the right singular vectors are parameter combinations
and the left singular vectors are observable field patterns. Retaining only
the first (r) modes means optimising coordinates (q) in
(\delta\theta=V_rq), not pretending that every original parameter is
independently visible. Weak modes can reflect the observation mask,
normalisation, loading path or constitutive redundancy rather than a solver
failure.

Definitions and recorded thresholds belong in
{doc}`../../reference/evidence/validation_metrics` and the evidence
registry.
