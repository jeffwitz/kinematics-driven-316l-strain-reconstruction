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

## Interpretation

Failure at one parameter pair is a numerical censoring event unless a physical
admissibility check fails. It must not be presented as a material boundary.
