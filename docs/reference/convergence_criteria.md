# Convergence criteria

**Category: Reference.**

## Global mechanics

Each increment must satisfy the configured residual criterion. Diagnostics
record attempted and converged increments, Newton iterations, cutbacks, final
absolute and relative residuals, and the reason for any failure.

## Constitutive state

MFront trial evaluations start from the last committed state. State is
committed only after global increment convergence. A failed Newton iteration
or cutback restores the committed state.

## Local 3D condensation

The transverse residual is
$[\sigma_{33},\sigma_{13},\sigma_{23}]$. Diagnostics record its maximum at
Gauss points, local iterations, failures and the maximum condition number of
$C_{bb}$. The public `S33_RESIDUAL_MPA` is the first component of the complete
residual.

## Micromorphic fixed point

The fixed point monitors the relative and absolute change in $\chi$, the
Helmholtz residual, finite values and the minimum yield-surface radius. Failure
does not commit MGIS state. Relaxation parameters and optional acceleration
must be recorded in the campaign manifest.

## Integrated section equilibrium

Saved plane-stress campaigns can be inspected with:

```bash
fem-inhouse diagnose-section-equilibrium \
  --campaign local results/my-local-campaign \
  --campaign coupled results/my-coupled-campaign \
  --partition-id 0 \
  --thickness-mm 2.0 \
  --output validation/section-equilibrium
```

For an interior DIC-Dirichlet partition, constancy of
`t integral sigma_yy dx` is not required: shear traction crosses the
artificial lateral cuts. The diagnostic reports the conservative balance
between the section-force increment and the lateral `sigma_12` flux. It also
reports the naive section-force dispersion, but that value is descriptive and
must not be interpreted alone as an equilibrium residual.

The current archived run is a baseline without an acceptance threshold. Its
source and claim boundary are recorded under evidence ID `E-EQ-001`.

## Interpretation

Failure at one parameter pair is a numerical censoring event unless a physical
admissibility check fails. It must not be presented as a material boundary.
