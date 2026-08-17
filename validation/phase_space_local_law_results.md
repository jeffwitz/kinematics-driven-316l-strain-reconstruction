# Phase-space local-law analysis — results

Against `validation/phase_space_local_law_preregistration.md`, thresholds
frozen before the run. The result is **dirty** — and, per the asymmetric
reading, that is the strong finding.

## Verdict against the frozen readings

| quantity | registered bar | measured (r=16, observable part) | reading |
|---|---|---|---|
| direction dispersion, best bins | `<= 15 deg` clean | **42.6 deg** | **dirty** |
| mean angle to `s`, best bins | `<= 10 deg` = J2-associated | **86.6 deg** | not J2 |
| non-J2 stable alternative | std `<= 15 deg`, mean `>= 20 deg` | std 42.6 deg | no stable alternative |
| amplitude structure | `R^2_cond >= 0.5` | **0.24** | unstructured |
| kernel exclusion | primary on observable projection | kernel removed by construction | held |

The cos-angle histogram is essentially **flat** (4.0–6.5 % per bin of width
0.1, slight U-shape): the deviatoric direction of the reconstructed inelastic
increment is isotropic with respect to the deviatoric stress. Robustness: the
top 10 % and top 1 % amplitude subpopulations are equally isotropic (mean
cos `+0.06` / `-0.15`, circular std `42.7 deg` / `34.3 deg`). No subpopulation
shows alignment with `s`, and the boundary mass is not the explanation —
only 304 of 400 000 samples sit at `|cos| < 1e-3` under this angle metric.

## What the dirty structure says

On the visited domain `Omega_P43`, with the displacement kernel removed, the
local state `(|s|, deviatoric angle, p_eq)` does **not** determine the
inelastic increment — neither its direction (42.6 deg dispersion) nor its
amplitude (`R^2_cond = 0.24`). The frozen interpretation is the strong one:
missing internal variables (crystallographic orientation, tensorial history,
gradients) **or** a reconstruction closure — and the second is the measured
candidate: the reconstructed field is the *effective* inelastic eigenstrain
(material plasticity + closure), and the closure is not a local function of
the predictor stress.

This is the expected and informative outcome of the pipeline the control
produced: before any law fit, the data themselves say that a law of the
minimal form `Delta eps^p = F(s, p_eq)` is not there to be found on this
reconstruction — exactly why the registered next step is the
`Delta eps_D / Delta eps_0` decomposition (dissipative component vs
zero-work closure), with the law fit attempted on the dissipative component
only.

## Coverage

251 of 256 bins populated, participation ratio 178: the single tensile path
covers the binning grid broadly but thinly — the coverage guard against
over-reading a single experiment remains in force.

## Caveats

* The state description tested is the minimal one. Richer descriptors
  (EBSD orientation, tensorial history) are the natural next test of the
  "missing variables" branch — the analysis is built to accept them.
* All numbers are the 100×100 qualification window; the trajectories at
  200×200 would multiply the samples by four but not by themselves change
  the reading.
* The f_0 boundary mass (47 % of active points, work metric) and the flat
  cos-angle histogram are different facts: the former is about the
  magnitude of `sigma : Delta eps^p`, the latter about the deviatoric
  direction. Both are reported; only the angle analysis is the object of
  this document's frozen bars.
