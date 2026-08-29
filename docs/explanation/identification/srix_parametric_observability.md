# Parametric observability of the SRIX FEMU model

**Mode:** explanation  
**Domain:** identification

This page concerns the **parametric** sensitivity of the SRIX forward model.
For a parameter vector $\theta$, the residual is

$$
r(\theta)=W\,[O(u(\theta))-y^{\mathrm{obs}}],
\qquad
S_\theta=\frac{\partial r}{\partial\theta}.
$$

The columns of $S_\theta$ are finite-difference sensitivities in the
registered log-parameter coordinates.  The parameter order is

$$
\theta=(\tau_0, R, Q, b),
$$

with $\tau_0$, $R$ and $Q$ in MPa and $b$ dimensionless.  The archived
shadow finite-difference step is $h=0.0015$ in log coordinates.  Synthetic
cases use identity observation and no noise; the historical experimental
rank-three case uses measured displacement with scalar DIC whitening, while
the raw case uses the unweighted displacement mismatch.  A separate offline
report now applies the registered spectral transfer and spatial whitener to
the archived raw displacement Jacobians.

This is different from the free-field operator in
{doc}`dic_weighted_tensor_observability`.  That operator asks which arbitrary
tensorial eigenstrain-like fields can be seen.  Here the constitutive forward
maps four SRIX parameters to a displacement, which is then observed and
differentiated.  Field observability is a prerequisite or upper bound; it is
not a demonstration of parameter observability.

## What is archived

The registered records are:

| Evidence | Case | Observation and parameterisation | Sensitivity availability |
|---|---|---|---|
| `E-SRIX-P43-SYNTH-001` | synthetic P43 M20 smoke | identity observation; 32 path steps; $\theta_4$ log coordinates | full $S_\theta$, singular values, correlation and $V$ |
| `E-SRIX-P43-SYNTH-002B` | synthetic P43 M20, four starts | same path and log coordinates; `h=0.0015` | per-start sensitivity, singular values, $V$ and error projections |
| `E-SRIX-P43-SYNTH-003` | synthetic P43 M100 scale-up | identity observation; M20 initialisation; log coordinates | full $S_\theta$, singular values, correlation and $V$ |
| `E-SRIX-P43-EXP-001-M20` | experimental P43 M20 | identity displacement plus scalar DIC whitening | prior/final SVD, retained rank-three basis and $V$ |
| `E-SRIX-P43-EXP-RAW-001` | experimental P43 M20 raw control | identity displacement; no whitening/covariance | prior/final SVD, retained rank-three basis and $V$ |

All five records use the registered SRIX transposed parameter preset and 32
path steps.  No sensitivity is reconstructed when it is absent: the
multi-start record stores per-start matrices, while the experimental records
store their prior and final SVD summaries rather than a batch matrix file.

## Synthetic spectra and parameter directions

The normalised singular values are:

| Case | $\sigma_1,\sigma_2,\sigma_3,\sigma_4$ (normalised) | Condition number |
|---|---|---:|
| synthetic M20 smoke | `1.0000, 0.135725, 0.036116, 0.0001031` | 9696.6 |
| synthetic M100 scale-up | `1.0000, 0.416078, 0.055394, 0.0001431` | 6989.7 |
| experimental M20, whitened | `1.0000, 0.143072, 0.016464, 0.00001553` | not archived |
| experimental M20, raw | `1.0000, 0.143084, 0.016465, 0.00001553` | not archived |

The first two right singular vectors are primarily combinations of $\tau_0$
and $R$.  The third is approximately the same-sign $Q+b$ direction, while
the weak fourth is the opposite-sign $Q-b$ direction.  For the synthetic M20
smoke, for example, the archived fourth vector is approximately

$$
v_4\simeq(0.00009,\ 0.00012,\ 0.7023,\ -0.7119).
$$

The experimental whitened record gives the same structure,

$$
v_4\simeq(0.00002,\ 0.00002,\ 0.7056,\ -0.7086),
$$

and its parameter correlation reports $\rho(Q,b)=0.999999997$ for the
synthetic smoke and approximately $0.9999999999$ for the M100 scale-up.
This is why the records do not authorise separate identification of $Q$ and
$b$, even when the synthetic optimiser reaches the generating truth.

The energy fractions

$$
E_k=\frac{\sum_{i\le k}\sigma_i^2}{\sum_i\sigma_i^2}
$$

are approximately:

| Case | $E_1$ | $E_2$ | Modes for 90% / 95% / 99% |
|---|---:|---:|---|
| synthetic M20 | 0.980656 | 0.998721 | 1 / 1 / 2 |
| synthetic M100 | 0.850203 | 0.997391 | 2 / 2 / 2 |
| experimental whitened | 0.979681 | 0.999734 | 1 / 1 / 2 |
| experimental raw | 0.979677 | 0.999734 | 1 / 1 / 2 |

These are information fractions of the linearised residual, not guarantees
that the corresponding parameters can be estimated independently.

## Synthetic versus experimental records

The synthetic multi-start campaign has four deterministic starts.  All reduce
the residual to the numerical floor or to $4.98\times10^{-13}$, and the
archived $V$ directions are stable across starts.  The M100 scale-up reaches
the synthetic truth in three evaluations from the M20 solution.  These are
demonstrations of the machinery and of one synthetic scale-up, not a material
calibration.

The experimental whitened and raw records have almost identical normalised
spectra and right-singular subspaces because the recorded whitening is scalar.
If $S_{\mathrm{scalar}}=cS_{\mathrm{raw}}$, then $V$ and all normalised
singular values are mathematically unchanged.  This is therefore a consistency
check, not an independent robustness validation.  Comparing the experimental
rank-three subspace with the synthetic one gives principal angles of about
$0.265^\circ$ for synthetic M20 versus experimental, and $0.139^\circ$ for
synthetic M100 versus experimental.  These comparisons describe the archived
linearisation points; the experimental optimisations still stop at the
evaluation cap or parameter bounds and are explicitly NO-GO.

The word “experimental” also requires care.  The data vector
$y^{\mathrm{obs}}$ does not enter $S_\theta$ when $O$ and $W$ are fixed:

$$
S_\theta=W O\,\frac{\partial u}{\partial\theta}.
$$

The experimental record changes the evaluation point, crop, loading path and
observation convention, not the derivative through an additive data vector.

The scientifically supported conclusion is consequently narrow:

```text
synthetic machinery                 demonstrated
strong tau0/R combinations          observable in the registered model
Q+b direction                       weaker but present
Q-b direction                       nearly null / not separately constrained
experimental uniqueness             not established
316L parameter identification       not claimed
```

The field-observability analysis remains a distinct prerequisite.  The full
offline construction is recorded in
`validation/reference_data/p0043_parametric_dic_weighting_v1/report.json`.
It uses the registered conversion of the repeated-frame payload
(`noise[:512,:512]` to canonical millimetres), 256 seeded windows, the
one-node boundary support mask, and the archived sinusoidal transfer.  No
forward or finite-difference calculation was performed.

For the experimental M20 final Jacobian, the three successive levels are:

| Level | Normalised singular values | Condition number | Rank at $10^{-2}/10^{-3}/10^{-4}$ |
|---|---|---:|---:|
| raw displacement | `1, 0.143084, 0.016465, 0.00001553` | `6.44e4` | `3 / 3 / 3` |
| DIC transfer only | `1, 0.045725, 0.005725, 0.000002047` | `4.88e5` | `2 / 3 / 3` |
| transfer + spatial whitening | `1, 0.050649, 0.007111, 0.000002408` | `4.15e5` | `2 / 3 / 3` |

The registered DIC chain therefore leaves a useful third direction at the
$10^{-3}$ scale, but attenuates it substantially; the fourth opposite-sign
$Q-b$ direction remains effectively null.  The full-chain rank-three
subspace is close to the raw one (principal angle about $0.020^\circ$), while
its leading one-dimensional direction rotates by about $1.75^\circ$.  At the
prior point the full-chain spectrum is
`1, 0.077415, 0.012622, 0.00001777`; this difference is a change of evaluation
point, not a direct effect of the measured data vector.

The synthetic M20 control has the same full-chain spectrum as the experimental
prior because the archived displacement Jacobians coincide at that point.  The
synthetic M100 control retains three directions above $10^{-3}$ and has full
chain spectrum `1, 0.358349, 0.044036, 0.00012164`.  These controls do not
authorise experimental calibration.

The exact key/shape/dtype inventory is recorded in
{doc}`../../_audit/srix_parametric_fields_inventory`.

Exact files, values and claim boundaries are listed in
{doc}`../../reference/evidence/srix_parametric_observability`.
