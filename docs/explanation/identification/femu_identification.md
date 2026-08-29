# FEMU and full-field identification

**Mode:** explanation  
**Domain:** identification

## What FEMU does

Finite-element model updating (FEMU) embeds a constitutive law in an
equilibrium forward problem and adjusts parameters so that predicted
observables agree with measured fields. The generic structure is

```{math}
R(u,\theta)=0,
\qquad
r(\theta)=W[O(u(\theta))-y^{obs}],
\qquad
J(\theta)=\tfrac12\|r(\theta)\|^2.
```

The observation operator is part of the objective. A DIC displacement can be a
measured boundary history or an interior observable; it is not an interior
constitutive state to impose everywhere. A full-field Dirichlet control that
prescribes the interior is consequently a useful wiring negative control, not
a valid material-identification experiment.

## Why it is expensive

Crystal plasticity adds twelve slip systems, internal variables, history and a
local plane-stress closure to every global increment. A central finite-
difference Jacobian for $p$ parameters requires approximately $2p$ complete
nonlinear forward replays, in addition to optimizer iterations.

| Route | Nonlinear forwards | Linear/tangent or adjoint work | Scaling intuition | Repository status |
|---|---:|---:|---|---|
| Central finite differences | About $2p$ per Jacobian | None beyond each forward | $O(p)$ nonlinear | Implemented and used as an oracle; fixed-path SRIX gates are mixed/blocked. |
| Direct/tangent sensitivity | One converged trajectory | One tangent solve/right-hand side per direction, reusing $R_u$ | $O(p)$ linearised directions | Implemented in synthetic and shadow SRIX paths; exact common-path qualification is not passed. |
| Adjoint, scalar objective | One converged trajectory | Approximately one transpose solve per objective plus local contractions | Weak dependence on $p$ | Full-field linear/eigenstrain $A^T$ is strongly qualified; a generic production SRIX parameter adjoint is not claimed. |

The cost is not zero in either differentiated route. History-dependent laws
must retain the correct state trajectory, and several objectives require
several adjoints. The advantage is avoiding an unnecessary nonlinear replay for
every local coefficient or parameter direction.

## Three sensitivity routes

### Finite differences

For the log or dimensional parameter convention declared by a case,

```{math}
\frac{\partial r}{\partial\theta_j}
\approx
\frac{r(\theta+h_je_j)-r(\theta-h_je_j)}{2h_j}.
```

This is general and easy to audit, which makes it a useful oracle. Its limits
are the $2p$ forward cost, the step-size trade-off, numerical noise and
perturbed forwards that fail or accept a different history.

### Direct/tangent sensitivity

Differentiating the converged equilibrium equation gives

```{math}
R_u\frac{\partial u}{\partial\theta_j}=-R_{\theta_j}.
```

The converged state and mechanical tangent can therefore be reused. In this
repository the direct SRIX path uses constitutive shadow histories and the same
matrix-free tangent-action/GMRES conventions as the reference solver. The
synthetic P43 records demonstrate useful sensitivity machinery, but the
registered exact common-path M8 gate remains blocked because its
finite-difference oracle did not preserve a common accepted trajectory. No
generic analytic MFront parameter-derivative block is claimed.

### Adjoint sensitivity

For a scalar objective, the mechanical adjoint satisfies

```{math}
R_u^T\lambda=J_u^T,
\qquad
\frac{dJ}{d\theta}=J_\theta-\lambda^TR_\theta.
```

The repository contains three distinct adjoint levels:

1. the full-field linear/eigenstrain operator $A^T$, qualified at full field;
2. mechanical adjoints of converged inverse problems, present in selected
   inverse implementations but not a universal SRIX production interface;
3. a sequential history adjoint used by the causal TANN exploration.

These are related transpose constructions, not interchangeable claims. The
full-field gate demonstrates that one forward action and one transpose action
can feed many local gradient contractions. It does not establish a complete
constitutive SRIX adjoint for every objective.

## Why full-field data are interesting

The same observation operator can be applied to predicted displacement and to
sensitivity columns:

```{math}
S_\theta=W O\frac{\partial u}{\partial\theta}.
```

This makes measurement transfer, crop, mask, units and noise part of the
inverse contract. Comparing raw mechanical strain to differentiated DIC
strain bypasses that contract and can confuse spatial transfer with material
error.

## Why identifiability must still be checked

Sensitivity calculation does not imply unique recovery. With

```{math}
S_\theta=U\Sigma V^T,
```

$V$ gives parameter combinations, $U$ gives observable patterns and small
singular values expose weak directions. Four parameters do not imply four
identifiable directions, and an excellent observable fit does not establish a
unique latent constitutive state. The free-tensor, compact local and enriched
basis studies document these distinctions in
{doc}`observable_fit_vs_latent_identifiability`.

## What is demonstrated here

The evidence ladder is deliberately scoped:

```text
direct/synthetic sensitivity machinery       demonstrated on registered cases
synthetic M20 identification and scale-up    demonstrated for recorded cases
experimental P43 sensitivity geometry        limited / under study
production boundary-only FEMU                not registered
experimental 316L calibration                not claimed
```

The positive synthetic records `E-SRIX-P43-SYNTH-001`, `-002B` and `-003`
demonstrate the machinery and one M20-to-M100 scale-up, not an experimental
material calibration. P43 remains an experimental demonstrator whose
crystal-plasticity localisation result is limited by data provenance and
unproven physical DIC--EBSD co-registration.

See {doc}`../../reference/numerics/femu_sensitivity_and_svd` for technical
contracts and {doc}`srix_parametric_observability` for registered
parameter-space examples.
