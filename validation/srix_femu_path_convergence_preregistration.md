# E-SRIX-FEMU-PATH-002 — preregistration

This gate tests convergence with respect to the discrete loading path. It does
not identify parameters and does not authorize P43.

Starting from the strict 57-step common path archived by
`srix_femu_common_path_gate_v9`, construct nested paths with 114 and 228 steps
by bisecting every interval. The boundary history, SRIX preset, EBSD map,
plane-stress backend, observation transfer, scored physical endpoint fractions,
and solver tolerances remain unchanged.

At each level compute only the converged base forward and the direct FEMU
Jacobian. No full finite-difference Jacobian is required; the v9 gate already
qualified the direct method against a same-path FD oracle.

The primary comparison is the 114-to-228 change:

* observed forward relative L2 change `< 0.5%`;
* columns 1–3 relative change `< 2%` and cosine `> 0.999`;
* rank-3 principal angles `< 2 degrees`;
* report all singular-value ratios, including the fourth, without imposing a
  stability gate on the fourth mode.

If the fourth singular ratio remains near `1e-4`–`1e-5`, it is reported as a
near-null direction and not used to authorize a four-parameter identification.
