# Why SRIX is the production law for the registered P43 reconstruction

This page records the constitutive decision for the full-resolution spectral
P43 calculation. It is deliberately separate from the numerical comparison
between TET2, CPS4 and EBI: the relevant question here is whether the chosen
crystal law is compatible with a quasi-static DIC reconstruction whose physical
time between images is not documented.

## The distinction that matters

Méric-Cailletaud is rate dependent. Its slip rule contains a physical or
pseudo-time rate, schematically,

```{math}
\dot{\gamma}_s
=
\left\langle\frac{f_s}{K}\right\rangle^n
\operatorname{sign}(\tau_s-X_s).
```

Consequently, changing the duration assigned to the same displacement path
changes the constitutive response. A Méric reconstruction requires the image
acquisition times, the interpolation of the displacement between images, and
parameters identified at that rate.

SRIX was introduced as the rate-independent FCC alternative. Its incremental
response depends on the strain path and its increments, not on an arbitrary
choice of elapsed seconds. The material-point campaign records bitwise
invariance when the duration assigned to the same path changes by a factor of
one million, while the Méric control changes by 9.1 MPa.

:::{admonition} Literature and model meaning
The rate-independent character is a property of the SRIX flow rule. It does
not make SRIX independent of path discretisation: too few strain increments
can still miss activation or reversal. Rate independence and temporal-path
resolution are different claims.
:::

## The registered P43 spectral result

The production calculation uses:

* 100x100 pixels and the registered P43 crop;
* two independent TRI2 constitutive histories per pixel;
* three-dimensional MFront condensation to plane stress;
* nonsymmetric LGMRES Newton iterations;
* eight proportional increments between zero displacement and the final
  boundary field;
* four MFront threads and one FFTW thread.

Three identical executions produced the same displacement, stress, reaction
and accumulated-slip SHA-256 values. The numerical result was:

| quantity | result |
|---|---:|
| Newton iterations | 46 |
| Jacobian matvecs | 637 |
| preconditioner applications | 599 |
| final verified residual | `6.13e-9` |
| median wall time | `78.12 s` |
| MAD of wall time | `0.74 s` |

The callback count is not confused with the mechanical operator count:
LGMRES reported 116 outer callbacks, while the actual Jacobian action was
called 637 times. The complete repeated report and its archive manifest are
the primary records for this result.

:::{admonition} Project numerical result
SRIX is qualified as the production law for the registered quasi-static P43
spectral reconstruction. This is a numerical and workflow qualification for
the declared case, not a new identification of the SRIX material parameters.
:::

## What happens with Méric-Cailletaud

With the same P43 crop and eight proportional increments, Méric-Cailletaud
does not complete the local plane-stress condensation. The local Newton reaches
its limit of 15 iterations with a transverse-stress residual of
`5.19e-5 MPa`.

This is not evidence that the spectral operator is defective. The bounded
diagnostic keeps the total pseudo-time equal to one and halves the increment
size by using 16 proportional increments:

| quantity | Méric, 16 increments |
|---|---:|
| status | converged numerically |
| verified residual | `8.59e-10` |
| Newton iterations | 94 |
| Jacobian matvecs | 1846 |
| total time | `304.5 s` |
| constitutive condensation | `269.2 s` |

The result shows sensitivity to temporal discretisation on this path. It does
not establish temporal convergence of the Méric fields: the 32- and
64-increment points were not retained, and the runner uses a proportional
synthetic path rather than a measured DIC chronology.

The comparison is nevertheless operationally decisive. The retained Méric
diagnostic costs about four times the qualified SRIX run while solving a
different, rate-dependent interpretation of a pseudo-time history that is not
physically documented.

:::{admonition} Scope boundary
The correct statement is not “Méric is non-convergent”. It is: “Méric is
sensitive to temporal discretisation on the registered P43 path; eight
increments fail locally, while 16 increments converge numerically but are not
temporally or operationally qualified.”
:::

## Decision

```text
SRIX P43 100x100
- production law for the registered quasi-static reconstruction
- independent of arbitrary pseudo-time duration
- converged and repeat-qualified at eight increments
- performance-qualified on the declared machine configuration

Méric-Cailletaud P43 100x100
- rate-dependent comparison law
- fails local condensation at eight increments
- converges numerically at 16 increments
- temporal field accuracy not demonstrated
- not selected for production
```

The decision therefore follows the information content of the experiment. The
DIC boundary sequence supplies an ordered displacement path, but not a
qualified physical time scale. SRIX removes that otherwise arbitrary rate
input while retaining the declared path discretisation and its measurable
limitations.

The material identification question remains separate: the SRIX parameter
`R` used in this campaign is an analytical transposition from a Méric pair at
a chosen reference rate, not a direct 316L identification. The production-law
decision does not change that status.

## Comparison of slip mechanisms

The archived 16-increment P43 fields show that the two laws share the same
principal system and the same top three systems, but not the same complete
distribution. The `S95` Jaccard index is `0.800`, the fraction-vector
variation distance is `0.2565`, and the normalized total-field cosine is
`0.9862`. The difference is therefore not adequately described as a single
global amplitude factor.

See the [complete system-level comparison](srix_meric_p43_slip_system_comparison)
for the spatial metrics, figures, scope and limitations.

## Evidence

* {doc}`../forest_rubin_srix` — rate independence, transposition limits and
  material identification scope.
* `E-SRIX-P43-001` — repeated SRIX P43 production evidence.
* `E-MERIC-P43-001` — eight-increment failure and 16-increment bounded
  refinement diagnostic.
