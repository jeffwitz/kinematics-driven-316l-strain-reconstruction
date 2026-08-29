# Sensitivity and gradient contracts

**Mode:** reference  
**Domain:** identification

This page defines what a sensitivity or gradient record must mean. Finite
differences, direct/tangent sensitivities and adjoints are different routes to
derivatives; the SVD interprets a matrix produced by one of those routes and is
not a fourth derivative algorithm.

## Residual definition

For a declared parameter vector $\theta$,

```{math}
r(\theta)=W[O(u(\theta))-y^{obs}],
\qquad
R(u,\theta)=0.
```

The record must identify the observation operator, crop, support, mask, units,
component order and whitening/noise convention. The operator is applied to
model predictions and sensitivity columns exactly once; already observed data
must not be transferred a second time.

## Parameter convention

Every run must record:

* dimensional or logarithmic parameters;
* parameter order, names and units;
* bounds, scaling and reference point;
* loading path, committed-state policy and scored frames;
* whether the derivative is local to a converged state or follows a complete
  history.

A change of parameterisation changes the sensitivity matrix and its observable
subspace. A singular direction is meaningful only with this convention.

## Route 1 — central finite differences

```{math}
\frac{\partial r}{\partial\theta_j}
\approx
\frac{r(\theta+h_je_j)-r(\theta-h_je_j)}{2h_j}.
```

Record the stencil, step, warm/cold-start policy, accepted path, failed-forward
policy and every rejected evaluation. Central differences are a general
oracle, but cost approximately $2p$ complete nonlinear forwards for $p$
parameters and can compare different histories if adaptive stepping is not
controlled.

## Route 2 — direct/tangent sensitivity

After a converged state, differentiate equilibrium as

```{math}
R_u\frac{\partial u}{\partial\theta_j}=-R_{\theta_j}.
```

Record the tangent definition, constitutive derivative source, linear solver
tolerance, boundary sensitivity and treatment of internal-variable history.
The repository has synthetic and shadow SRIX implementations and archived
direct Jacobians. The exact common-path M8 qualification gate remains blocked;
these records must not be promoted to a generic production SRIX derivative.

## Route 3 — adjoint

For a scalar objective, record the transpose convention and solve

```{math}
R_u^T\lambda=J_u^T,
\qquad
\frac{dJ}{d\theta}=J_\theta-\lambda^TR_\theta.
```

Record objective definition, adjoint tolerance, dot-product/transpose tests,
state-history backward treatment and the number of scalar objectives. The
repository's full-field eigenstrain operator $A^T$ is qualified at large scale;
this is distinct from a generic constitutive SRIX parameter adjoint. The causal
TANN sequence adjoint is a separate history-aware implementation.

## SVD interpretation

Given a recorded sensitivity matrix,

```{math}
S=U\Sigma V^T.
```

* $U$ contains observable field patterns;
* $V$ contains parameter combinations;
* $\Sigma$ contains their sensitivity in the declared metric.

A retained rank $r$ defines $\delta\theta=V_rq$. The report must preserve all
singular values, $U$/$V$ when available, the retained rank and its threshold;
weak/null directions must not be silently deleted. Relative singular-value
thresholds are descriptive, not an experimental confidence criterion.

SVD answers which combinations a particular observation and linearisation can
distinguish. It does not prove that the nonlinear inverse is unique, that a
latent field is physically correct, or that a material parameter is calibrated.

## Required provenance checklist

Each sensitivity/gradient artifact should identify:

```text
forward state and loading history
parameterisation and reference point
observation operator, mask and whitening
derivative route and numerical tolerances
singular values, vectors and rank rule
failed/rejected evaluations and claim boundary
```

The interpretation layer is in
{doc}`../../explanation/identification/identifiability`; the broader FEMU
strategy and route comparison are in
{doc}`../../explanation/identification/femu_identification`.
