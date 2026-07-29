# P43 DIC uncertainty propagation: pre-execution amendment

Date: 2026-07-29

This amendment was written after the first computational pilot and before any
uncertainty result was accepted or versioned.

## Detected defect

The preregistered periodic translations occasionally place the artificial
periodic join of the full measured residual inside the P43 solve support. The
opposite edges of the recorded crop do not have matching displacement values.
Their join therefore creates a non-physical displacement discontinuity and an
extreme EVM line. The pilot consequently produced a bimodal metric
distribution with an artificial high-error tail.

Those pilot intervals are invalid and must not be interpreted.

## Locked correction

Keep every other preregistered choice unchanged, including the measured
repeat-frame residual, component-mean removal, random sign, sample count,
seed, metrics and immutable FEM predictions.

Replace periodic translations by a uniformly sampled **contiguous window** of
the measured residual whose size is exactly the P43 nodal solve support. The
window origin is sampled independently and uniformly from every origin for
which the complete support remains inside the recorded crop. No wrap, padding,
interpolation or synthetic seam is allowed.

This correction preserves the measured spatial covariance over the complete
solve support while removing the numerical discontinuity introduced solely by
the periodic boundary convention.

## Audit trail

The failed pilot was not committed as scientific evidence. Its failure mode,
cause and replacement are fixed by this amendment before the corrected
campaign is executed.
