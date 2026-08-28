# FEMU sensitivity and SVD reference

**Mode:** reference  
**Domain:** identification

For parameters $\theta$, the weighted residual is

$$r(\theta)=W[O(u(\theta))-y^{obs}],$$

and the sensitivity matrix is $S=\partial r/\partial\theta$. The residual
must state whether $\theta$ is dimensional or logarithmic, how the field is
flattened, which mask and observation operator are applied, and how the
whitening matrix $W$ is obtained. Runs must record the finite-difference
stencil, relative step, failed-forward policy and rejected evaluations.

The singular value decomposition is $S=U\Sigma V^T$. Columns of $V$ are
right-singular parameter combinations; columns of $U$ are observable field
patterns; $\Sigma$ contains the singular values. A retained rank $r$ defines
coordinates $\delta\theta=V_rq$. The threshold selecting $r$ must be recorded,
and weak/null modes must be reported rather than silently removed. A change of
parameterisation changes $S$ and therefore changes the observable subspace.

Required provenance includes singular values, $U$, $V$, retained rank and
threshold, plus all forward and observation settings. A full-field Dirichlet
control that prescribes the interior is not a valid FEMU sensitivity experiment:
it removes the constitutive response that the objective is meant to identify.
Interpretation belongs
in {doc}`../../explanation/identification/identifiability`.
