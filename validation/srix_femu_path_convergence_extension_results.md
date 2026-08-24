# E-SRIX-FEMU-PATH-002S — results

Status: **blocked at direct sensitivity evaluation** (2026-08-24)

The final preregistered refinement was attempted from the qualified L2 path in
`validation/reference_data/srix_femu_path_convergence_v3/`. The L3 path contains
784 mandatory midpoint steps and converges after 25 local repairs, for 809
steps in total.

The mechanical forward therefore remains converged. However, when constructing
the direct Jacobian, a persistent constitutive shadow replay fails with:

```text
MFrontIntegrationError: 3D MFront integration failed with status -1
```

Consequently, no L2→L3 sensitivity metric is available and the extension gate
cannot pass. This is recorded as a sensitivity/constitutive continuation
diagnostic, not as a forward failure and not as evidence that the parameters
are unidentifiable.

This was the final preregistered refinement. No L4 or blind refinement is
authorized. Identification and P43 remain blocked until the shadow sensitivity
failure is diagnosed.

Primary machine-readable artefact:
`validation/reference_data/srix_femu_path_convergence_v4/report.json`.
