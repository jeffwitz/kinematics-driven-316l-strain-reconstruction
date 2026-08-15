# What Ludwik/J2 reproduces of the measured P43 kinematics

Run: `scripts/replay_ludwik_two_state_history_p43.py`, artefacts in
`results/ludwik-two-state-replay-p0043/`. Crop origin `(1580, 1030)`, 100x100
pixels, the repaired DIC Dirichlet history from state 0 to state 40, forty
increments, final equilibrium residual `8.95e-10`, 380 Newton iterations, 350 s.

## The comparison is exact, not approximate

No observation operator had to be invented and no solver had to be written. The
TwoSubcell chain already existed end to end:
`solve_two_state_dirichlet_plane_stress` builds `TwoSubcellDiagnostic2D` and
`TraditionalTwoStateTriangleBatch`, which carries **two independent
constitutive histories per pixel**, and `TensorPlasticObservabilityOperator`
builds the same kinematics. Measurement, simulation and inverse share one
`(nx, ny, 2, 3)` layout with no interpolation anywhere. What was missing was an
artefact, not a capability: the existing TRI2 benchmark drives that solver with
a proportional ramp on a different crop, never with the measured history.

The elastic denominator comes from the same solver with the yield stress set
out of reach, so `eps_el` and `eps_L` differ by the constitutive law alone.

The trajectory starts at the undeformed state 0, not at state 20: plasticity is
path dependent, and starting mid-history would hand the model a virgin material
the specimen no longer is.

## Result: Ludwik makes the strain agreement worse

`E_L = |eps_L - eps_DIC| / |eps_el - eps_DIC|`, so `E_L < 1` means the model
explains part of the elastic defect.

| state | E_L | eta_L | elastic error | Ludwik error |
|---|---|---|---|---|
| 25 | 1.636 | -0.636 | 0.466 | 0.763 |
| 30 | 1.697 | -0.697 | 0.504 | 0.855 |
| 35 | 2.066 | -1.066 | 0.390 | 0.805 |
| 40 | 2.912 | -1.912 | 0.292 | 0.850 |

Every component behaves the same way, shear worst in absolute terms
(`E_L` 1.33 to 2.25, with the Ludwik shear error exceeding the measured shear
itself at every state). Nodal displacement errors stay near 0.1 % for both
models, which is the reminder that a full-Dirichlet crop is almost determined
by its boundary: the discrimination lives entirely in the strain, a derivative.

## The amplitude is right. The distribution is wrong.

This is the answer to the binary question, and it is not the expected one.

| state 40 | mean | CV | corr. with DIC |
|---|---|---|---|
| DIC | 4.95e-3 | 0.22 | 1 |
| Ludwik | 5.04e-3 | 0.77 | 0.229 |
| elastic | 4.66e-3 | 0.15 | 0.645 |

The mean equivalent strain Ludwik produces lands within 2 % of the measured
mean -- the hardening amplitude is essentially calibrated. Its **heterogeneity
is 3.5 times too large**, and it is heterogeneous in the wrong places: the
elastic solution, which contains no plasticity at all, correlates almost three
times better with the measured field than Ludwik does.

The source is identifiable. Ludwik's equivalent strain correlates `-0.569` with
the per-pixel yield map, the measurement only `-0.196`. The model localises
into the soft pixels of that map; the specimen does not, or not nearly as much.
The maps show it directly: coarse 45-degree bands of saturated strain in the
simulation against a fine, comparatively uniform texture in the measurement.

## No rescaling can repair it

Writing `c = eps_L - eps_el` for the correction the model applies and
`g = eps_DIC - eps_el` for the defect to be closed:

| state | cos(c, g) | best scalar alpha | E_L at that alpha |
|---|---|---|---|
| 25 | +0.006 | +0.005 | 1.000 |
| 30 | -0.052 | -0.039 | 0.999 |
| 35 | +0.038 | +0.021 | 0.999 |
| 40 | +0.038 | +0.014 | 0.999 |

The correction is **orthogonal** to the defect. The optimal global rescaling is
to switch it off, and even then the residual is unchanged to three decimals. No
recalibration of the hardening amplitude, the exponent, or a uniform scaling of
the yield map can help, because scaling a vector orthogonal to its target
cannot shorten the residual. This closes the amplitude hypothesis at the global
level, and it closes it cheaply -- the test costs two inner products.

It does **not** close the pointwise amplitude hypothesis. A field `dp(x)` along
Ludwik's flow direction has twenty thousand degrees of freedom, and can zero
the correction where the model localises wrongly while raising it elsewhere.
That is now the decisive next experiment rather than a formality.

## The caveat that must be measured next

The elastic defect is 0.29 of the measured strain norm at state 40, while the
nodal agreement is 0.1 %. Much of that defect is fine, pixel-scale texture that
a smooth solution cannot carry. How much of it is *explainable at all* is not
established here: the propagated-noise reference exists (`(I - E P_b) n`, some
18 times smaller than the raw DIC noise) but was not applied to this
comparison. If a large share of the elastic defect is propagated noise, then
`cos(c, g) ~ 0` is partly a statement about noise rather than about Ludwik, and
the honest denominator for every ratio above is smaller than the one used.
Nothing in this note depends on that correction -- `E_L > 1` and the CV mismatch
are properties of the simulated field -- but the interpretation of "orthogonal"
does.

## What this rules out, and what it points at

Ruled out: that Ludwik needs its amplitude, hardening, or exponent retuned.
Ruled out: that the remaining defect is a modest perturbation of a broadly
correct plastic field. The model's plastic field is anti-correlated with the
measurement's structure at the scale where it matters.

Pointed at: the **heterogeneity source**. The per-pixel yield map, derived from
`el_thresh50.npy` with a 50 MPa floor and a 380 MPa hardening scale, drives
localisation the measurement does not show. Whether the true heterogeneity is
crystallographic and simply mislocated by that map, or whether J2 flow cannot
produce the measured pattern at any yield map, is exactly what the nested
inverse (amplitude along `n_L`, then the two transverse directions) is built to
separate.
