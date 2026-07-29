# P43 measured-boundary history: endpoint amendment

Date: 2026-07-29

This amendment is fixed after the single final-frame compatibility check and
before reconstructing the 40-state history or running mechanics.

## Compatibility result

Recomputing `000294 -> 000334` with OpenCV 4.14 and the source-derived
`legacy_script_2021` settings differs from the historically prepared P43
final displacement by:

- component RMS difference: \(7.764\times10^{-4}\) mm;
- maximum absolute component difference: \(1.209\times10^{-3}\) mm;
- relative vector norm: 1.583 %.

This is consistent with the uncertified historical OpenCV factory defaults.
Using the recomputed endpoint directly would confound loading-path sensitivity
with a changed final boundary condition.

## Locked endpoint anchoring

Let \(u_k^{\mathrm{new}}\) be the directly recomputed displacement at ordered
step \(k=1,\ldots,40\), and let \(u_{40}^{\mathrm{prepared}}\) be the immutable
prepared final displacement. Define:

\[
u_k^{\mathrm{anchored}} =
u_k^{\mathrm{new}} +
\frac{k}{40}
\left(u_{40}^{\mathrm{prepared}}-u_{40}^{\mathrm{new}}\right).
\]

This correction:

- is zero at the reference state;
- reaches the exact historical endpoint;
- changes smoothly and linearly with ordered pseudo-time;
- preserves every non-proportional deviation from the recomputed history;
- allows a controlled comparison with the existing proportional baseline.

Both raw recomputed and anchored histories, plus the correction field and
their hashes, must be archived. Results must be labelled as an
OpenCV-4.14 reconstruction anchored to the historical final field, not as a
bitwise reproduction of the unavailable historical history.
