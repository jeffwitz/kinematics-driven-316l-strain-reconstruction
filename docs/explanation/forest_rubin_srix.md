# The SRIX rate-independent crystal law, and why it is not Méric-Cailletaud

**Category: Explanation.** This page says what the two FCC single-crystal laws
in `mfront/` compute, how one is derived from the other, and precisely how far
that derivation may be trusted.

## Two laws, one difference

Both laws are small-strain, tridimensional, orthotropic, and share everything
that describes the crystal:

- twelve octahedral slip systems `<0,1,-1>{1,1,1}`;
- the same seven-coefficient interaction matrix;
- saturating isotropic hardening `(τ₀, Q, b)` coupled through that matrix;
- Armstrong-Frederick kinematic hardening `(C, d)`;
- cubic elasticity from `C11 = 197000`, `C12 = 125000`, `C44 = 122000` MPa;
- the same implicit hardening update `Δp_s = |Δγ_s|`.

They differ in the flow rule alone.

`Fcc316LMericCailletaud`, rate dependent, from the TFEL gallery:

$$\dot\gamma_s = \left\langle\frac{f_s}{K}\right\rangle^{n}\operatorname{sign}(\tau_s - X_s)$$

`Fcc316LForestRubinSrix`, rate independent, from Forest and Rubin:

$$\dot\gamma_s = \dot{\bar\varepsilon}\left\langle\frac{f_s}{R}\right\rangle\operatorname{sign}(\tau_s - X_s),
\qquad
\dot{\bar\varepsilon} = \sqrt{\tfrac{2}{3}\,\dot{\boldsymbol\varepsilon}' : \dot{\boldsymbol\varepsilon}'}$$

with, in both cases, $f_s = |\tau_s - X_s| - g_s$ and
$g_s = r_0 + Q\sum_r h_{sr}\left(1 - \exp(-b p_r)\right)$.

Incrementally, over a step in which the strain varies linearly,

$$\Delta\gamma_s = \Delta\bar\varepsilon\,\frac{\max(f_s, 0)}{R}\operatorname{sign}(\tau_s - X_s),
\qquad
\Delta\bar\varepsilon = \sqrt{\tfrac{2}{3}\,\Delta\boldsymbol\varepsilon' : \Delta\boldsymbol\varepsilon'}.$$

Time enters only through $\Delta\bar\varepsilon$, which is itself a strain. The
response depends on the strain **path**, not on how fast it is traversed.

## The demonstration of time independence

`test_srix_is_time_independent` walks one strain path over total times of
`1e-2`, `1`, `1e2` and `1e4` seconds — a factor of a million — and requires
**bit-for-bit** equality of the stresses, of every internal variable and of the
consistent tangent. Not a tolerance: there is no `dt` in the law, so no
mechanism exists by which a difference could appear, and any non-zero deviation
means `dt` has crept back in.

Measured: `0.000e+00` on all three, at every ratio.

The control matters as much. `test_meric_cailletaud_is_not_time_independent`
runs the same path through the viscous law and requires the answer to move:
measured at 120.26 MPa against 111.16 MPa, a spread of 9.1 MPa across the same
range. Without that control, the first test could pass on a broken harness.

## Where R comes from, and what it is not

Equation (16) of Forest and Rubin equates the overstress of the two models for
the tension of a `[001]` single crystal at a chosen reference strain rate:

$$R = \frac{8}{\sqrt 6}K\left(\frac{\sqrt 6\,\dot\varepsilon_{\mathrm{ref}}}{8}\right)^{1/n}$$

`srix_overstress_modulus_from_meric` implements it. For `K = 12` MPa, `n = 11` and
`ε̇_ref = 1e-3 s⁻¹` it returns **18.7819100705 MPa**, which is the default of
the `R` parameter and the value used in every test on this page.

Three things this is not:

1. **Not an identification.** No 316L measurement was fitted to the SRIX law.
   `R` is an analytical transposition of a parameter set identified for a
   *different* flow rule. `provenance_record()` states this in a `status`
   field, and a test asserts the wording survives.
2. **Not orientation independent.** The equality is established for `[001]`
   tension only. Measured on the axial stress at 0.6 % strain: **0.32 %**
   difference for `[001]`, **7.1 %** for `[111]`, **14.2 %** for `[123]`. The
   test suite asserts both the agreement at `[001]` and the *disagreement*
   elsewhere, so the correspondence cannot be quietly overstated.
3. **Not tied to our experiment.** The reference rate of `1e-3 s⁻¹` is a
   placeholder chosen to make the number reproducible. The strain rate of our
   DIC test has not been documented. `srix_overstress_modulus_from_meric` therefore takes
   the rate as a required argument with no default: `R` carries the rate at
   which the viscous law was frozen, and a default would silently attach an
   unstated experimental condition to every result.

### Transposition is one route to R, not the definition of R

The function is named `..._from_meric` because equation (16) is a *bridge from a
rate-dependent law*, not the meaning of the parameter. In the SRIX model `R` is
the overstress modulus of the flow rule: it sets how much overstress
$|\tau_s - X_s| - r_s$ is needed to drive a given slip increment, and therefore
how abrupt the elastic-plastic transition is.

That makes it **directly identifiable**, with no Méric-Cailletaud law anywhere in
the chain: fit it to the *width of the measured transition* on a monotonic
curve. This is the route registered in
`validation/srix_316l_calibration_preregistration.md`, and it is the one that
would let a result claim `R` as an identified 316L parameter.

The distinction is carried in the manifest rather than left to the reader. A
value from this function is recorded with status `analytical_transposition`; a
fitted one would be recorded as `identified`. Only the second may support a
statement about the material.

## Why the equivalent strain increment is built from the unknowns

The obvious implementation reads $\Delta\bar\varepsilon$ straight off `deto`,
the imposed strain increment. It gives the right stresses and the **wrong
tangent**.

In the implicit DSL the unknowns are `deel` and `dg`; `deto` is data. Built
from `deto`, $\Delta\bar\varepsilon$ is a constant of the local system, and its
dependence on the imposed strain is invisible to the `StandardElasticity`
brick, which then assembles a consistent operator missing a rank-one term.
Measured against one-sided finite differences:

| state | relative deviation, from `deto` | from the unknowns |
|---|---:|---:|
| elastic | `3.2e-13` | `3.2e-13` |
| near the transition | `1.5e-11` | `1.5e-11` |
| established plasticity `[001]` | `7.7e-02` | `1.4e-07` |
| transverse perturbation | `2.0e-02` | `7.4e-08` |
| shear | `4.2e-01` | `7.0e-07` |
| orientation `[111]` | `1.1e-01` | `4.3e-07` |

Exact in the elastic range either way — which is why an elastic-only check
would have missed it entirely — and wrong by up to 42 % as soon as slip is
active.

The fix uses an identity rather than a hand-written `@TangentOperator`. The
elastic residual of the brick is `feel = deel - deto + Σ dg_s m_s`, so at
convergence `deto = deel + Σ dg_s m_s` exactly, and since the Schmid tensors
are deviatoric the two readings of $\Delta\bar\varepsilon$ are algebraically
identical. Building it from `deel` and `dg` routes the same dependence through
the implicit Jacobian, and the brick assembles the right operator by itself.
The residual deviation of `7e-7` is the truncation error of the finite
differences, not of the model.

## Two traps in the Jacobian, both measured

**The Macaulay bracket does not differentiate itself.** In Méric-Cailletaud the
slope `dv = n v / f` vanishes on its own as `f → 0`, so an inactive system
contributes nothing to the Jacobian without anyone arranging it. In SRIX the
slope is `Δε̄ / R`, a constant that vanishes for nobody. Writing it on an
inactive system corrupts the Jacobian by exactly the number of inactive
systems, and the local Newton then fails **non-monotonically** in the step
size: converging at `1e-3`, failing at `5e-4`. The law guards this explicitly.

### Semismooth linearisation at zero slip

The production law retains the sharp absolute value in the hardening and
backstrain updates. At `dg = 0`, its classical derivative is undefined; the
local Newton linearisation selects the symmetric Clarke generalized derivative
`d|dg|/ddg = 0`. This is a choice of generalized Jacobian, not a smoothing of
the constitutive law. The residual and committed state update are unchanged.
The rationale and qualification are documented in
{doc}`../reference/numerics/srix_semismooth_jacobian`.

**The overflow guard must not be copied.** Méric-Cailletaud rejects any step
whose overstress exceeds `1.1 K`, protecting its Norton power from overflow.
SRIX is linear in `f` and needs no such guard; carrying it over would discard
perfectly valid states.

Removing it has a practical consequence. SRIX integrates a single increment of
**5 % strain**, where Méric-Cailletaud declines at 1 %. For a project whose
runtime is close to linear in the increment count, a law that tolerates larger
steps is worth more than a marginally faster one.

## Degenerate increments

A null increment and a purely hydrostatic increment both have
$\Delta\bar\varepsilon = 0$. Below `deqeps` the law takes a guarded branch: no
division is reached, no slip appears, no internal variable moves, and the
hydrostatic case still returns its full elastic pressure. Verified as such.

The branch is a guarded block and not an early `return`, because the
`StandardElasticity` brick appends its own code after the user integrator and
returning would skip it.

## Unloading is smooth, and that is the point

SRIX has no loading-unloading switch. While `f` stays positive the systems keep
sliding, so a small reversal still produces a little slip: measured at 0.12 % of
the accumulated slip at 0.999 of the peak, 0.45 % at 0.99, and exactly zero by
0.95, once the overstress has collapsed.

This is the smooth elastic-plastic transition named in the title of the paper,
and it is what removes the slip indeterminacy of the classical rate-independent
formulation. It is a property of the model, not an artefact of the integration.

## Plane stress

There is no plane-stress version of either law and there must not be one. The
catalogue declares `native_plane_stress_behaviour=None`, so the 3D law is
condensed by the existing solver, which drives `σ_zz` to zero by Newton on the
transverse strain.

Building $\Delta\bar\varepsilon$ from the unknowns keeps that closure honest:
every update of the transverse strain changes `deel` and therefore
$\Delta\bar\varepsilon$. It is never frozen on the in-plane components alone,
which would be the natural mistake and would silently decouple the equivalent
increment from the very unknown being solved for.

## Cost

Measured at 62 to 99 µs per material point per increment, against roughly 6 µs
for the J2 Ludwik law the project runs today: a factor near 16, intrinsic to
integrating twelve slip systems with a full 12×12 interaction matrix rather
than a scalar radial return.

The consistent tangent of a single crystal is also **not symmetric**: relative
asymmetry measured at `4.0e-05` median against `1.4e-16` for J2. Both crystal
behaviours therefore declare `linear_system_matrix_type="nonsymmetric"` and
force the linear solver out of its symmetric mode, at roughly twice the memory
and under 10 % of added runtime.

For a controlled comparison with the rate-dependent Méric--Cailletaud law, use
the paired 316L backbone documented in the Reference page
`crystal_parameter_pairs`. It locks the elastic, FCC and hardening data before
the two flow rules are selected.

The [registered P43 slip-system comparison](spectral_mechanics/srix_meric_p43_slip_system_comparison)
tests the resulting distributions directly. Agreement in a `[001]` reference
transposition does not guarantee the same dominant systems, redistribution or
slip amplitude in a multiaxial field.

## References

- Samuel Forest and M. B. Rubin, *A rate-independent crystal plasticity model
  with a smooth elastic-plastic transition and no slip indeterminacy*, European
  Journal of Mechanics A/Solids **55**, 278–288, 2016.
  DOI [10.1016/j.euromechsol.2015.08.012](https://doi.org/10.1016/j.euromechsol.2015.08.012).
  Full text: <https://minesparis-psl.hal.science/hal-01251477>.
  Equation (7) is the flow rule, (8) the equivalent strain rate, (16) the
  correspondence with `(K, n)`.
- M. A. Nasri et al., *Proper Generalized Decomposition for the numerical
  simulation of polycrystalline aggregates under cyclic loading*, Comptes
  Rendus Mécanique **346**, 132–151, 2018.
  DOI [10.1016/j.crme.2017.11.009](https://doi.org/10.1016/j.crme.2017.11.009).
  Source of the 316L hardening set.
- TFEL/MFront, *Méric-Cailletaud single crystal plasticity*:
  <https://thelfer.github.io/tfel/web/MericCailletaudSingleCrystalPlasticity.html>.
