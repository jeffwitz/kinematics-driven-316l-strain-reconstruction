# P43 global Newton replay diagnostic

The nested and coupled drivers were rerun with identical P43 M20 inputs.  The
runner now archives the global residual history and every linear solve.  This
does not alter either algorithm; it exposes where the global trajectories
diverge.

The first three increments have identical global Newton counts (2, 4, 3).
The first count difference occurs at increment 4, where nested uses 3 Newton
iterations and coupled uses 4.  From increment 9 onward, nested uses 4
iterations per increment while coupled uses 5.  The totals are therefore 121
versus 146.  GMRES per global Newton remains essentially unchanged (25.88
versus 26.18), so the extra 681 GMRES iterations are caused by the additional
global steps, not by a poorer Krylov solve at a fixed step.

The per-increment nonlinear residuals are equal to the displayed precision
through the shared iterations.  Around the stopping threshold, the two paths
straddle tiny differences: for example, at increment 4 the nested path stops
after the residual sequence `4.11e-2, 5.67e-4`, whereas coupled performs one
additional solve from `1.63e-7`.  Similar threshold crossings occur at
increments 8--32.  This is consistent with amplification of local roundoff
and stopping-test differences by the outer nonlinear iteration, rather than a
material tangent defect.

Independent same-strain local replay remains the stronger constitutive check:
the condensed tangents differ by at most `5.6e-10` relatively and the active
slip masks agree for all `1536` entries.  The full-forward displacement fields
still differ by only `1.94e-12 mm`.

The current evidence therefore supports keeping both implementations without
changing tolerances or forcing identical outer trajectories.  A literal
same-iterate global replay would require storing trial displacement fields at
each outer Newton callback; it is not needed to establish the present
diagnosis and should be added only if exact trajectory reproducibility becomes
a requirement.
