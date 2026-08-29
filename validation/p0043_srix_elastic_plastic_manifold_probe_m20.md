# P43 SRIX elastic–plastic manifold probe (M20)

## Status

Phase 0 (auditability) is complete locally. Phase 1 found no archived elastic
sensitivity compatible with the LOT E M20 displacement contract, so the gate
stops before any new forward. No elastic finite difference or nonlinear probe
was launched.

## LOT E audit

The observable-matched LOT E fixture uses eight scored states `[4, 8, 12, 16,
20, 24, 28, 32]`, eight `21 x 21 x 2` displacement blocks, and 7056 rows in
millimetres. Its local manifest records the fixture SHA256, shapes, source
commit, and reconstruction command:

`validation/reference_data/p0043_krylov_srix_intersection_m20_v1/manifest.json`

The local fixture gates pass: zero boundary defect, affine-extension error
zero, relative `A/A^T` dot-test error `7.0e-14`, and contribution reconstruction
errors below `2e-16`.

## Existing elastic sensitivity artifacts

The archived direct-sensitivity artifacts
(`srix_femu_direct_sensitivity_v1/v2/jacobian.npz`) have arrays of shape
`1296 x 4`, score states `[4, 36, 125, 275, 312, 316, 326, 338]`, and a
different observation/support contract. They cannot provide the three elastic
columns for LOT F.

Other elastic reports are M100, 10x10, or otherwise different supports and are
not reused by reshape, interpolation, or state relabelling.

## Forward gate

The permitted fallback would require up to six central finite-difference
M20 forwards in stable cubic coordinates. In the clean checkout, the repaired
P43 history payload is absent and the MGIS/TFEL runtime is unavailable; the
archived MFront library alone is not an executable Python constitutive path.
Therefore no forward was attempted and no elastic tangent was fabricated.

## Stable elastic coordinates (for a future compatible run)

\[
K=(C_{11}+2C_{12})/3,\qquad C'=(C_{11}-C_{12})/2,\qquad C_{44},
\]

with log coordinates and reconstruction
`C11 = K + 4 C'/3`, `C12 = K - 2 C'/3`. The historical set is
`(K,C',C44)=(149000,36000,122000) MPa`; the updated set is
`(169300,36750,125400) MPa`.

## Verdict

**LOT F blocked before elastic sensitivity construction.** The existing
LOT E conclusion remains unchanged: the SRIX plastic tangent has only partial
overlap with the observable Krylov correction. A valid elastic comparison
requires the missing compatible M20 forward payload/runtime or a future
archived `7056 x 3` elastic sensitivity.
