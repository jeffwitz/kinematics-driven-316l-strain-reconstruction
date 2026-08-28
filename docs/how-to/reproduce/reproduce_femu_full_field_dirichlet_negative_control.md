# Reproduce the full-field-Dirichlet FEMU negative control

**Mode:** how-to  
**Domain:** evidence

This procedure reproduces the registered limitation of the current FEMU
smoke driver; it is not a production identification workflow.

1. Check out the registered configuration and record its source commit.
2. Run `scripts/srix_femu_smoke.py` with the preregistered P43/M20 inputs.
3. Inspect the manifest and report to verify that displacement is prescribed
   on the full field (including the interior).
4. Confirm that the constitutive perturbation has no independent interior
   displacement response, and archive the objective, parameter ordering and
   report path.

The result is a negative control: full-field Dirichlet data remove the
mechanical sensitivity that FEMU would need. A future boundary-only driver
must be documented separately with its own sensitivity and SVD artefacts; this
page must not be used as evidence that such a driver exists.

See {doc}`../../reference/numerics/femu_sensitivity_and_svd` for the required
sensitivity provenance and {doc}`../../explanation/identification/femu_identification`
for the interpretation.
