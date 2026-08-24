# Synchronized common-path gate — adaptive trajectory blocker

The synchronized gate is implemented in
`scripts/qualify_srix_femu_common_path_gate.py`. It first runs the nine
adaptive trajectories, takes the union of their accepted fractions, and then
replays all directions on that common partition with local synchronized
bisection when necessary.

The first M8 execution did not reach the common replay. The directions through
`b_plus` returned, but `b_minus` remained in the adaptive solver for more than
40 minutes without returning an accepted path. The run was interrupted rather
than allowed to remain unbounded. The result is recorded in
`validation/reference_data/srix_femu_common_path_gate_v1/report.json`.

This is a diagnostic of adaptive-path cost/branching, not a negative result for
the direct sensitivity method. No common-path FD, SVD, or parameter claim is
authorized from this run.
