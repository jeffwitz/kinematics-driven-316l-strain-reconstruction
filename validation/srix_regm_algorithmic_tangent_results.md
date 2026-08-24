# Algorithmic-tangent REGM diagnostic

This preregistered diagnostic tests whether the poor REGM/FEMU information
geometry is caused primarily by the constant elastic reconditioner `K0`.
On the exact M8 twin, the weak residual is reconditioned at each scored state
with the statewise consistent SRIX algorithmic tangent `K_alg`. No global
Newton solve is performed and no P43 data are used.

The machine-readable primary artifact is
`validation/reference_data/srix_regm_algorithmic_tangent_v1/report.json`.

| quantity | REGM with `K_alg` | observed FEMU reference |
|---|---:|---:|
| normalized singular values | `1, 0.37594, 0.03469, 8.62e-5` | `1, 0.54152, 0.40668, 0.06787` |
| condition number | `1.16e4` | `14.7` |
| leading principal angle | `73.9 deg` | reference subspace |
| rank-2 principal angles | `67.75 deg, 11.50 deg` | reference subspace |

The tangent reconditioner changes the REGM spectrum only modestly relative to
the fixed EBSD-cubic elastic `K0` (`1, 0.42199, 0.03240, 4.65e-5`, condition
number `2.15e4`). It does not recover the two weak FEMU directions; the
leading subspace angle even increases from `68.4 deg` to `73.9 deg`.

## Decision

Replacing `K0` by a statewise algorithmic tangent alone is **not sufficient**.
The hypothesis that the missing information is only an elastic-versus-plastic
preconditioner mismatch is rejected for this twin. This is a diagnostic
negative result, not a statement that SRIX or the complete FEMU is invalid.

The next and final low-cost diagnostic is a sequential one-correction replay:
one tangent correction per increment, followed by a constitutive re-evaluation
and commit. It tests the missing causal displacement--internal-state feedback
without converging a global Newton solve. P43 remains blocked until that gate,
and the independent forward validation, pass.

```{figure} _static/evidence/srix_regm_algorithmic_tangent.png
:alt: Information geometry for statewise algorithmic-tangent REGM and observed FEMU.
:width: 95%

The algorithmic tangent improves conditioning but does not align REGM with the
local FEMU sensitivity subspace.
```
