# Inspect spectral convergence

Inspect the saved diagnostics for:

```text
equilibrium residual
post-revert verification residual
Newton and GMRES iteration counts
high-frequency energy
constitutive evaluations
```

For the registered Newton-GMRES comparison:

```bash
jq '{mesh, tolerance, errors, iterations, verification_residual}' \
  validation/_generated/ebi_tet/state_sharing_m12_reproduced.json
```

Accept a run only when the final solver residual and the independent
post-revert verification residual are both below the requested tolerance.
Compare fields at identical tolerances before attributing differences to the
spatial method.

The historical fixed-point script writes JSONL traces separately; those traces
are useful for diagnosing Anderson, but they are not Newton-GMRES evidence.
Do not infer accuracy from iteration count alone. A converged residual, an
independent verification residual and a field comparison at identical
kinematics are all required.
