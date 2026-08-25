# E-SRIX-P43-SVD-SHADOW-001

This gate qualifies direct constitutive shadows in a fixed global SVD basis
against the existing nine-parameter displacement FD oracle. The first
diagnostic uses the two-point global smoke archive and rank 7; it is not an
identification campaign.

For a retained basis `V_r`, shadows use

```text
eta_plus/minus = eta +/- h_i V_r[:, i]
```

with fixed-current-strain forcing, the converged FEM tangent, causal shadow
history advancement, and per-mode clipped steps

```text
h_i = clip(1e-3 * s_1 / s_i, 5e-4, 5e-3).
```

The primary gate compares `Jz_shadow` to `Jeta_FD @ V_r`; target thresholds
are cosine >= 0.999 and relative column error <= 1% for strong modes, with a
3% allowance for the weakest retained mode when the `h`/`h/2` check is stable.

The global Sobol campaign must be complete before this local smoke result can
be promoted to a qualified global basis or used for experimental
identification.
