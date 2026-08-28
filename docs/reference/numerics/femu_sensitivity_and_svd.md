# FEMU sensitivity and SVD reference

**Mode:** reference  
**Domain:** identification

For parameters \(\theta\), the weighted residual is

$$r(\theta)=W[O(u(\theta))-y^{obs}],$$

and the sensitivity matrix is \(S=\partial r/\partial\theta\). Runs must
record parameter ordering or log-parameterisation, scales, observation
operator, mask, whitening, finite-difference step and rejected evaluations.

The singular value decomposition is \(S=U\Sigma V^T\). Columns of (V) are
parameter combinations and columns of (U) are observable field patterns. A
retained rank (r) defines coordinates \(\delta\theta=V_rq\); weak/null modes
are reported rather than silently removed.

Required provenance includes singular values, (V), retained rank and
threshold, plus all forward and observation settings. Interpretation belongs
in {doc}`../../explanation/identification/identifiability`.
