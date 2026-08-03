# SRIX 316L calibration — preregistration

Date: 2026-08-03
Written before any 316L parameter has been fitted to the SRIX law.
Section 12 of the 2026-08-03 specification.

## What this fixes

The order in which the parameters will be identified, and what each step is
allowed to use. It is written now, while no data has been looked at, because the
temptation it exists to prevent — adjusting a parameter after seeing a curve —
is invisible once the curve is on screen.

No calibration has been performed. Every registered parameter set today carries
`literature_prior` on its hardening and `analytical_transposition` or
`exploratory` on `R`; see `docs/reference/srix_parameter_sets.md`.

## The prohibition

> **Identifying all parameters simultaneously on one macroscopic curve is
> forbidden.**

Six quantities — the three cubic stiffnesses, `tau0`, `R`, `(Q, b)`, `(C, d)`
and seven interaction coefficients — cannot be recovered from a single monotonic
tensile curve. Several combinations of them produce the same curve to within any
realistic measurement error, so a simultaneous fit returns *a* solution, not
*the* solution, and its parameters carry no individual meaning even when the fit
looks excellent. Any result obtained that way is to be recorded as
`exploratory`, whatever its residual.

## The registered order

Each step uses only what the steps before it established, and each fixes what it
identifies before the next begins.

**1. Elasticity, `C11, C12, C44`.** From elastic data: single-crystal
measurements, or the elastic slope of several orientations if single-crystal
data are unavailable. Fixed before anything plastic is touched. Justification
from the sensitivity campaign: a 2 percent change in the elasticity moves the
stress by as much as doubling `R` does at `[001]`, so an error here is absorbed
by every later parameter.

**2. `tau0`, the slip threshold.** From the onset of slip, on the orientation
whose transition is sharpest. Not from a 0.2 percent offset yield stress, which
is a conventional construction and depends on the whole hardening curve.

**3. `R`, the overstress modulus.** From the **width of the elastic-plastic
transition**, not from a Méric-Cailletaud transposition. Two constraints follow
from the material-point campaign and are registered here:

- **`R` must be identified off-axis.** Its influence is smallest on `[001]` —
  7.7 percent of spread over the exploratory sweep, against 18.6 percent on
  `[123]` — and `[001]` is precisely where equation (16) makes it analytically
  redundant with `(K, n)`. Calibrating there and transferring is the specific
  error to avoid.
- **`R` and `tau0` are coupled** through `O_R = (sqrt(6)/8) R / tau_0`, which is
  the dimensionless quantity the transition width actually measures. Step 2 and
  step 3 must therefore be revisited jointly once both have a first value, and
  the revision recorded, rather than each being frozen in isolation.

`R` also changes the **active set**, not only the stress level: on `[123]` the
model runs on eight systems up to `R = 8` and on nine at `R = 18.78`. A step
that changes which systems carry the deformation cannot be treated as a scaling.

**4. `(Q, b)` and the interaction matrix.** From **several monotonic
orientations**, never one. Latent hardening is what the interaction matrix
describes, and a single orientation activates a single family of interactions.
The six published coefficients are converted through
`from_publication_coefficients`; see
`docs/reference/fcc_interaction_matrix_mapping.md`. If they are fitted rather
than adopted, the fit must respect the constraint that both glissile slots hold
the same value, or the hardening matrix stops being symmetric and the model
leaves the convention of the sources.

**5. `(C, d)`, kinematic hardening.** From **reversed and cyclic** loading only.
They are invisible on a monotonic path at the strains this project reaches: the
stored kinematic energy is 0.071 MPa against 2.1 of dissipation at 2 percent on
`[001]`. Fitting them monotonically would fit noise.

The step-size finding of the canonical qualification applies here and is
registered as a procedural constraint: **below about twenty increments over the
path, a reversal produces no reverse slip at all**. Any cyclic identification
must verify its increment count against that before reading a residual.

**6. Validation.** On orientations and paths **not used during identification**.
A model that reproduces its own calibration set has demonstrated nothing.

## What each step must record

For every parameter, at the moment it is fixed: its value, its unit, the data it
was fitted to, the residual, the orientations and paths used, and its status
promoted from `literature_prior` to `identified`. The machinery for this exists
— `SrixParameterSet` carries per-group provenance and
`srix_provenance` adds the run half — so a calibrated set is registered like any
other and cannot be reported without its attribution.

A set may be described as identifying 316L only when **every** group reads
`identified` or `literature_measurement`. `claims_material_identification`
computes exactly that, and a test asserts today that no registered set passes it.

## Falsifiers

**F1.** If step 3 cannot separate `R` from `tau0` on the available data — that
is, if a range of `(R, tau0)` pairs at constant `O_R` fits equally well — then
only `O_R` is identifiable and `R` must remain `exploratory`. This is a
publishable outcome and will be published.

**F2.** If step 4 requires an interaction matrix that is non-symmetric to fit
several orientations, the six-coefficient convention is inadequate for this
material and that must be stated rather than absorbed by writing different
values into the two glissile slots.

**F3.** If step 6 fails on unseen orientations while steps 1 to 5 each succeeded,
the parameters are fitted to the calibration set and not to the material. The
set stays `exploratory`.

## What this preregistration does not cover

The micromorphic length `ell` and the coupling `alpha` are out of scope, per the
prohibitions of section 16. So is any use of CPS4R for a scientific conclusion:
the reduced element is not qualified for elastoplastic campaigns, see
`validation/cps4r_qualification_results.md`.
