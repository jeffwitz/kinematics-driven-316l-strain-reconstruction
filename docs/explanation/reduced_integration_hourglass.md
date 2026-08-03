# Reduced integration and hourglass control

The reference element is the fully integrated bilinear plane-stress quadrilateral
`CPS4`, with four Gauss points. The optional `CPS4R` formulation uses one
central material point. It therefore divides the number of constitutive
integrations by four, which is particularly valuable for the SRIX crystal law.

## Missing modes

A one-point bilinear quadrilateral detects the three constant in-plane strain
components but leaves two non-rigid displacement modes without constitutive
energy. They are the two hourglass modes. Exposing the reduced element without
stabilisation would make its stiffness singular.

For the regular rectangular mesh used by this project, the stabilisation is

\[
K_{hg}=\beta\left(K_{ref}^{4pt}-K_{ref}^{1pt}\right),
\qquad 0<\beta\leq1.
\]

The same reference operator is used in both terms. Consequently, translations,
rigid rotation and affine displacement fields make no contribution to
\(K_{hg}\). At \(\beta=1\), a constant linear-elastic material recovers the
fully integrated element stiffness exactly.

The stabilisation is stiffness based. No viscous control is used because this is
a quasi-static problem and the answer must not acquire a dependence on increment
duration from the element formulation.

## Material reference

For isotropic J2 plasticity, the reference operator is the elastic plane-stress
matrix. For a crystal law it is the cubic elastic operator, rotated into the
global frame and condensed under the global plane-stress condition. It is
measured from the actual MFront behaviour. An isotropic fallback is forbidden.

The stabilisation remains elastic after constitutive yielding. It is therefore a
numerical energy, not crystal hardening, plastic dissipation or a nonlocal
interaction.

## Diagnostics

The final element field is stored as
`HOURGLASS_ENERGY_BY_ELEMENT.npy` for CPS4R campaigns:

\[
E_{hg,e}=\frac12u_e^T K_{hg,e}u_e.
\]

The global ratio is

\[
r_{hg}=\frac{\sum_e E_{hg,e}}{|W_{int}|},
\]

where \(W_{int}\) is integrated by the trapezoidal rule over accepted equilibrium
increments. Failed trials and cutbacks do not contribute.

A small global ratio is not sufficient evidence. The element field must also be
checked against plastic bands because a local concentration can be hidden by a
large domain average. The following are analysis guides, not universal
validity limits:

- below 1 percent: weak global influence;
- between 1 and 5 percent: inspect the spatial field;
- above 5 percent: the reduced result may be contaminated.

## Present validation boundary

The algebraic rank, affine patch tests, rotated cubic elasticity and a
non-affine elastic solve are covered by automated tests. Non-affine plastic J2
and SRIX comparisons are still required before CPS4R can replace CPS4 in a
scientific campaign. CPS4 remains the reference formulation.

CPS4R is deliberately refused with the micromorphic nonlocal coupling until
that interaction has been validated separately.
