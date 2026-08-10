# Scientific contract

The exact manuscript used to define this contract is identified by size and
SHA-256 in `ArticleSource/manifest.json`. The manifest deliberately records a
null DOI because no DOI is present in the supplied manuscript; publication
metadata must not be invented.

Status: implementation baseline, 2026-07-24.

This document makes the conventions used by the code explicit. It is based on
`ArticleSource/ArticleAdil.pdf`; remaining ambiguities must be resolved against
the original Abaqus input generator and ODB extraction scripts.

## Scientific objective

The software reconstructs a mechanically admissible kinematic field from DIC
measurements. It aims to reproduce the spatial organization of microscale
strain localization. Pixel-wise elastoplastic parameters are effective
reconstruction descriptors conditional on the loading path, DIC resolution,
filtering, and constitutive assumptions. They are not intrinsic grain
properties.

Direct FE stress and strain-based reconstructed stress answer different
questions and must never be substituted for one another.

## Array and coordinate convention

All structured nodal and element arrays use:

- axis 0: physical `x`, transverse direction;
- axis 1: physical `y`, tensile/loading direction;
- `u_x`: displacement along `x`;
- `u_y`: displacement along `y`.

The historical DIC names map as follows in the cropped mesh frame:

- `V` is `u_x`;
- `U` is `u_y`.

For an array `a`, `numpy.gradient(a, dx, dy)` therefore returns derivatives
along `(x, y)` in that order.

## Units

The internal unit system is:

- length and displacement: millimetres;
- stress and elastic moduli: megapascals;
- strain: dimensionless;
- reaction force: newtons only after the plane-stress section thickness has
  been specified consistently with Abaqus.

The nominal DIC-to-mesh spacing is `0.001 mm × 1.84 = 0.00184 mm`.

The physical specimen thickness reported by the article is `2 mm`; this fact
justifies the plane-stress assumption but does not, by itself, establish the
section thickness used by the Abaqus CPS4 model. That value remains to be
verified from the original input file.

## Small-strain components

The tensorial in-plane strain components are:

```text
epsilon_xx = d(u_x) / dx
epsilon_yy = d(u_y) / dy
epsilon_xy = 0.5 * (d(u_x) / dy + d(u_y) / dx)
```

Engineering shear is:

```text
gamma_xy = 2 * epsilon_xy
```

FE element vectors use `[epsilon_xx, epsilon_yy, gamma_xy]`. DIC invariant
calculations use tensorial `epsilon_xy`; conversion must therefore be explicit.

## Complete tensors from the converged 2D state

The finite-element model remains strictly two-dimensional. It has no
out-of-plane displacement, element, or global equation. After convergence, the
accepted plane-stress state is completed locally for output only.

The mechanical constraints are:

```text
sigma_33 = sigma_13 = sigma_23 = 0
epsilon_13 = epsilon_23 = 0
```

Plane stress does not imply `epsilon_33 = 0`. For associative J2 plasticity:

```text
epsilon_p_33 = -(epsilon_p_11 + epsilon_p_22)
epsilon_e_33 = -nu / (1 - nu) * (epsilon_e_11 + epsilon_e_22)
epsilon_33 = epsilon_e_33 + epsilon_p_33
```

The public final result contains `S_3D`, `E_3D`, `EE_3D`, `PE_3D`, and
`PLANE_STRESS_RESIDUAL_MPA` in addition to the unchanged historical arrays.
The vector residual is `[S33, S13, S23]`; `S33_RESIDUAL_MPA` remains its first
component for compatibility. MFront native-plane-stress outputs use MGIS
`AxialStrain`, `ElasticStrain`, and complete `Stress`; no native residual is
replaced by an exact zero.

## Three-dimensional constitutive condensation

The constitutive layer supports three distinct routes. Native `PlaneStress` is
used for behaviours that directly provide the required two-dimensional
contract. `mfront-3d-condensed-plane-stress` is the independent reference for
small-strain three-dimensional behaviours: it solves locally for
`[epsilon_33, gamma_13, gamma_23]` such that
`[sigma_33, sigma_13, sigma_23] = 0`, then passes the corresponding Schur
tangent to the two-dimensional solver. The registered
`mfront-structural-plane-stress` backend applies the same three-traction
closure inside an `Implicit`/`StandardElasticity`-compatible MFront behaviour
and retains its complete three-dimensional constitutive state.

The local iterations always restart from the last globally committed material
state. The mesh, displacement unknowns, element formulation, global equations
and Newton algorithm are unchanged by the constitutive route. The structural
backend is qualified for the current small-strain crystal-plasticity
behaviours; finite-strain and multiplicative kinematics are outside this
contract. See
{doc}`reference/numerics/mfront_structural_plane_stress` and
{doc}`reference/configuration` for the detailed contract and configuration.

## Historical and reconstructed equivalent strain

The article-era operation

```text
epsilon_zz = -nu / (1 - nu) * (epsilon_xx + epsilon_yy)
```

applies an elastic closure directly to total in-plane strain. It is valid in
elasticity but not generally after plastic flow. It remains available only as
the explicitly named `EVM_HISTORICAL` comparison metric.

`EVM_RECONSTRUCTED_3D` is instead evaluated from the complete converged FEM
total-strain tensor:

```text
epsilon_vM = sqrt(2/3 * epsilon_dev : epsilon_dev)
```

A single final DIC image does not contain the plastic history required for this
completion. The historical closure must not be presented as a mechanically
identified transverse DIC strain after yielding.

## Constitutive convention

The local hardening law is:

```text
sigma_eq = sigma_y + K * epsilon_p**n
```

Nominal article values are `E = 205000 MPa`, `nu = 0.30`, `n = 0.245`,
`sigma_y = 124 MPa`, and `K = 380 MPa`. Local maps may replace `sigma_y` and
`K`; `E`, `nu`, and `n` remain homogeneous for the supported case.

The article states that Abaqus used a table over `0 <= epsilon_p <= 0.2`, with
1000 points and a minimum positive increment of `1e-6`. This is retained as a
historical reproduction mode, not as the production constitutive contract.

The default MFront law regularises only the first interval
`0 <= epsilon_p <= 1e-6`, then evaluates the analytical power law without an
upper PEEQ cap. This avoids introducing a non-physical plateau outside the
tabulated Abaqus range.

## Optional coupled micromorphic extension

The experimental coupled mode retains the same local J2 flow and regularized
Ludwik law. It introduces one element-centred scalar field `chi` and augments
the yield radius by

```text
Hchi * (PEEQ - chi)
```

where `chi` solves the existing discrete Helmholtz equation using the
element-average local PEEQ as its source. This field is solved inside every
mechanical Newton trial. It is not an output filter, a phase-field damage
variable, an additional displacement, or a replacement of PEEQ in the Ludwik
power law.

The MFront tangent is evaluated at fixed `chi`. Constitutive trials remain
uncommitted throughout the micromorphic fixed point and are committed only
after global convergence. The initial P154 campaign fixes
`ell = 0.05888 mm`; it selects only `Hchi`. Neither parameter is considered an
identified material property until a frozen pair transfers successfully to a
held-out partition.

The completed P154 sweep did not freeze a pair: the best tested candidate
passed seven of eight registered checks but exceeded the absolute DIC-q90
active-area limit (`21.85%` predicted versus `20%` maximum). The current
scientific status is therefore *partially supported*, with no confirmatory
transfer authorized by this contract.

## Optional reduced integration

The reference finite element remains `CPS4`, integrated at four Gauss points.
The optional `CPS4R` formulation uses one central constitutive point and
stiffness-based hourglass control,

$$K_{hg}=\beta\left(K_{ref}^{4pt}-K_{ref}^{1pt}\right),
\qquad 0<\beta\leq1.$$

The reference material operator is isotropic elastic for J2 and the rotated,
plane-stress-condensed cubic elastic operator for a crystal behaviour. It is
measured from the behaviour rather than reconstructed from nominal constants,
and an isotropic fallback is refused. The stabilisation remains elastic after
yielding and must never be interpreted as plastic dissipation, crystal
hardening, or nonlocal physics.

At $\beta=1$, CPS4R must recover CPS4 for constant linear elasticity, including
non-affine displacement fields that excite hourglass modes. This exact elastic
property does not extend to elastoplastic response.

Those comparisons have now been run, under
`validation/cps4r_qualification_preregistration.md`. **They failed.** The
plastic-strain error against CPS4 is 1.9 to 10 percent against a 0.5 percent
bound, on both a heterogeneous J2 case and a tilted-orientation crystal case.
**CPS4R is not authorised for scientific elastoplastic campaigns and no value of
$\beta$ is recommended.** CPS4 remains the reference formulation and the default.
Two qualifications belong with that verdict: the cost case did hold, at 1.9 to
2.9 times on total wall time, and the displacement difference is 30 to 200 times
below the DIC measurement noise, so the failure is one of numerical
self-consistency rather than of measurable physics.

After yielding, $\beta=1$ is the least accurate choice rather than the natural
one: the stabilisation keeps the elastic reference while the constitutive
tangent softens, so the hourglass modes stay elastically stiff while every other
mode yields.

For accepted equilibrium increments, internal work is accumulated by the
trapezoidal rule from the mechanical internal-force vector. Failed trials and
cutbacks contribute neither internal work nor diagnostic energy. A CPS4R
campaign records the global hourglass energy, its ratio to accumulated internal
work, and the spatial field `HOURGLASS_ENERGY_BY_ELEMENT`.

The ratio compares a state quantity, the stabilisation energy stored at the
final configuration, with a path quantity that includes plastic dissipation. It
therefore decreases as the loading path lengthens, at fixed element behaviour,
and is only comparable between runs with comparable paths. It is evaluated at
the final state alone, so a transient excitation that unloads leaves no trace in
it.

**The ratio must not be used as a validity gate.** The qualification campaign
measured a correlation of `0.033` between the element hourglass energy and the
CPS4-to-CPS4R plastic-strain error, and `0.066` between that energy and the
plastic strain itself; every configuration tested passed a one percent ratio by
an order of magnitude while missing the accuracy bound by four to twenty times.
The ratio reports how hard the stabilisation is working. This contract makes no
claim that it reports how wrong the answer is.

CPS4R and the micromorphic nonlocal extension are deliberately incompatible
until their interaction has been validated.

## Four macroscopic curves

The workflow must keep these four curves distinct:

1. measured macroscopic stress-strain curve;
2. DIC strain-based reconstructed equivalent stress;
3. FE strain-based reconstructed equivalent stress;
4. direct FE equivalent stress.

Curves 2 and 3 reapply the scalar constitutive law to spatially averaged
equivalent strain. They are consistency checks, not independent stress
predictions.

Curve 4 is computed by first spatially averaging `S11`, `S22`, and `S12`, then
applying the plane-stress von Mises formula. The direct curve may remain below
the measured response; this is a scientific result and must not be hidden by
switching to curve 3 after yielding.

## Shared DIC/FE comparison

For scientific field comparison, both DIC and FE strains are derived from
their nodal displacement fields with the same NumPy gradient convention. This
avoids mixing DIC differentiation with Abaqus visualization extrapolation.
