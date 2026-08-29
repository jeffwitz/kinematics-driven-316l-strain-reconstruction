# P43 SRIX elastic–plastic manifold probe (M20)

## Scope and environment gate

This is a registered-case methodological diagnostic. It uses the archived
M20 SRIX displacement contract (eight scored states `[4, 8, 12, 16, 20, 24,
28, 32]`, eight `21 x 21 x 2` blocks, 7056 rows, millimetres) and does not
launch the nonlinear mixed-mode probe.

The historical runtime was recovered on the host:

* Python: `/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/.venv/bin/python`;
* MGIS Python modules: `/home/jeff/.local/lib/python3.12/site-packages`;
* TFEL/MFront: `/home/jeff/.local/bin/mfront`, TFEL 5.1.0;
* behaviour library: `build/mfront/src/libBehaviour.so`;
* repaired history SHA256: `8b0c6df9b8ac6235c87b0e5d60e5dee6a4e6e905980c879d50c595fe1d72c8a0`.

The 32-step baseline replay reproduces the archived M20 displacement exactly:
`max_abs_delta = 0`, `relative_delta = 0`, shape `(7056,)`. This validates the
runtime, payload and observation contract before perturbing elasticity.

## Local tangent geometry

Elasticity was perturbed in the stable cubic coordinates

\[
K=(C_{11}+2C_{12})/3,\qquad C'=(C_{11}-C_{12})/2,\qquad C_{44},
\]

using `log(K), log(C'), log(C44)` and central steps `±0.01`. The historical
constants are `(C11,C12,C44)=(197000,125000,122000) MPa`, hence
`(K,C',C44)=(149000,36000,122000) MPa`. All reconstructed perturbations pass
the cubic stability checks. The half-step check for `K` changes the derivative
by `4.06e-4` relatively.

Nine M20 forwards were executed: one baseline, six elastic central differences,
and two `K` half-step checks. No nonlinear ±20% probe was run.

The SVD threshold is `1e-4` relative to the largest singular value, so the
near-null `Q-b` direction is excluded from the numerical tangent rank.

| tangent | rank | normalized singular values |
| --- | ---: | --- |
| plastic `S_P` | 3 | `1, 0.1431, 0.01647` |
| elastic `S_E` | 3 | `1, 0.3652, 0.1533` |
| combined `S_EP` | 6 | `1, 0.8167, 0.2440, 0.1307, 0.0750, 0.01559` |

The elastic span has three independent directions outside the plastic span;
its singular values there are `3.589e-5, 1.487e-5, 8.018e-6`.

Principal angles between the fitted Krylov trajectory-contribution space and
the tangent spaces (degrees, three smallest) are:

| tangent | angle 1 | angle 2 | angle 3 |
| --- | ---: | ---: | ---: |
| plastic | 33.28 | 53.86 | 61.88 |
| elastic | 34.08 | 49.67 | 56.98 |
| combined | 28.35 | 41.77 | 47.81 |

Adding elasticity therefore improves the geometric overlap, but does not make
the Krylov correction a close match to the local combined manifold.

## Projection fractions

These are geometric squared projection fractions in the raw displacement
space, not likelihood or confidence measures.

| vector | `eta_P` | `eta_E` | `eta_EP` | new elastic `eta_{E|P}` |
| --- | ---: | ---: | ---: | ---: |
| Krylov raw r16 | 0.0640 | 0.1770 | 0.2169 | 0.1633 |
| Krylov dissipative r16 | 0.0627 | 0.1798 | 0.2054 | 0.1523 |
| final SRIX residual | 0.2100 | 0.3598 | 0.4131 | 0.2571 |

Per-state combined projection of the raw Krylov contribution is, for scored
steps 4–32, respectively:
`0.2105, 0.2926, 0.3271, 0.3287, 0.1942, 0.3891, 0.3950, 0.2778`.
Elastic directions contribute at every state; they are not confined to the
initial elastic part of the history.

## Interpretation

The result is **verdict A (elasticity supplies missing directions, at least
partly)**. The independent elastic contribution is substantial: it raises the
Krylov raw explained fraction from `0.0640` for plastic parameters alone to
`0.2169` for the combined local span, and raises the final-residual fraction
from `0.2100` to `0.4131`. The dissipative reconstruction shows the same trend.

This means elastic and plastic uncertainty should be considered jointly before
judging SRIX adequacy. It does not identify elastic constants or SRIX
parameters, and the remaining moderate angles/fractions leave open additional
plastic-manifold or model-form limitations. A nonlinear combined-manifold probe
is therefore a possible later experiment, but was deliberately not launched
in this lot.

The minimum-norm seven-coordinate maps are reported in the JSON as
`local_tangent_equivalent_parameter_factors`; their very large/small factors
are a conditioning diagnostic, never identified parameters.

All statements remain limited to the P43 registered-case diagnostic. Physical
DIC–EBSD co-registration is not independently proven (R2 is not satisfied).
The generated sensitivity arrays remain local/ignored; the committed JSON,
script, and this report provide their hash, shapes, provenance and replay
command without pushing the binary payload.
