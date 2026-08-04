# Inspect spectral convergence

Inspect the saved diagnostics for:

```text
equilibrium residual
post-revert verification residual
Newton and GMRES iteration counts
high-frequency energy
constitutive evaluations
```

For the command in the run guide:

```bash
jq '.variants | keys' validation/_generated/spectral2d_registered/report.json
tail -n 1 validation/_generated/spectral2d_registered/*trace.jsonl | jq .
```

Accept a run only when the final solver residual and the independent
post-revert verification residual are both below the requested tolerance.
Compare fields at identical tolerances before attributing differences to the
spatial method.

Do not infer accuracy from iteration count alone. A converged residual and a
field comparison at identical kinematics are both required.
