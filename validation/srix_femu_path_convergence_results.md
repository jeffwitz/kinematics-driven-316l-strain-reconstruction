# E-SRIX-FEMU-PATH-002 — historical result

> This document is retained for discovery history only. The corrected nested
> study is now in
> `validation/srix_femu_path_convergence_rebaseline_results.md` with primary
> artefact `validation/reference_data/srix_femu_path_convergence_v3/`.

> **Superseded for scientific interpretation.** This experiment depends on
> the pre-fix v9 common path and is retained only to document discovery of the
> initial-Dirichlet bug. The corrected re-baseline is
> `validation/reference_data/srix_femu_common_path_gate_v16/`, currently
> blocked by the strict oracle.

The nested-path experiment was run from the clean v9 common path. The 57-step
forward and direct Jacobian were recomputed successfully. Its normalized
spectrum is:

```text
(1, 0.187363, 0.040544, 5.354e-5)
```

The first strict bisection refinement (114 steps) does not converge with the
same oracle policy: the base trajectory fails at increment 34, on the interval
`[0.236328125, 0.23828125]`. Increasing the Newton cap from 80 to 120 and 160
in an isolated diagnostic did not change this failure. This is therefore not a
simple Newton-iteration budget issue.

The gate is recorded as `blocked_path_level`, not as a convergence result:

```text
forward convergence: not demonstrated
Jacobian convergence: not demonstrated
identification_authorized: false
p43_authorized: false
```

The machine-readable artifact is
`validation/reference_data/srix_femu_path_convergence_v2/report.json`, with
`path_convergence.npz` and `path_convergence.png`. The result means that the
57-step direct Jacobian is locally qualified against its same-path FD oracle,
but its spectrum cannot yet be interpreted as the limit of a refined loading
discretization. The next investigation must resolve the branch failure near
load fraction `0.237`, or construct a preregistered locally refined path that
actually converges before comparing spectra.
