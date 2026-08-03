# Reduced integration and hourglass control

The reference element is the fully integrated bilinear plane-stress quadrilateral
`CPS4`, with four Gauss points. The optional `CPS4R` formulation uses one
central material point. It therefore divides the number of constitutive
integrations by four, which is particularly valuable for the SRIX crystal law:
a crystal material point costs roughly sixteen times a J2 point, so the
constitutive evaluation dominates the wall time and dividing it by four is the
single largest saving available without changing the physics.

This page states the element algebra in full, because the stabilisation is
*defined* by that algebra rather than by a tuned parameter, and because the
diagnostic it produces can only be read correctly by someone who knows what it
measures. The bibliography is at the end.

## The element

Nodes are numbered counter-clockwise, and the parent square
$(\xi,\eta)\in[-1,1]^2$ carries the bilinear shape functions

$$N_1=\tfrac14(1-\xi)(1-\eta),\quad N_2=\tfrac14(1+\xi)(1-\eta),$$
$$N_3=\tfrac14(1+\xi)(1+\eta),\quad N_4=\tfrac14(1-\xi)(1+\eta).$$

The isoparametric map and the displacement interpolation use the same basis,
$\mathbf{x}=\sum_a N_a\mathbf{x}_a$ and $\mathbf{u}=\sum_a N_a\mathbf{u}_a$, with
the Jacobian $J_{ij}=\partial x_j/\partial\xi_i$ and $|J|>0$ enforced. Physical
derivatives follow from $\partial N_a/\partial x_j = J^{-1}\,\partial N_a/\partial\xi_i$.

Strains use the engineering-shear convention
$\boldsymbol{\varepsilon}=[\varepsilon_{xx},\varepsilon_{yy},\gamma_{xy}]^{T}$,
so that for the nodal vector
$\mathbf{u}_e=[u_{x1},u_{y1},\dots,u_{x4},u_{y4}]^{T}$,

$$\boldsymbol{\varepsilon}=B(\xi,\eta)\,\mathbf{u}_e,\qquad
B=\begin{bmatrix}
N_{1,x} & 0 & \cdots & N_{4,x} & 0\\
0 & N_{1,y} & \cdots & 0 & N_{4,y}\\
N_{1,y} & N_{1,x} & \cdots & N_{4,y} & N_{4,x}
\end{bmatrix}.$$

The element stiffness is the parent-domain integral

$$K_e=\int_{-1}^{1}\!\!\int_{-1}^{1} B^{T} C\, B\,|J|\;d\xi\,d\eta
\;\simeq\;\sum_{g} w_g\,|J(\xi_g,\eta_g)|\;B_g^{T} C\, B_g .$$

## The two quadrature rules

Both rules must integrate the parent area, $\sum_g w_g = 4$; this is checked when
a rule is constructed, because a rule that does not reproduce the area silently
rescales every stiffness built from it.

| rule | points | weights |
|---|---|---|
| `cps4` | $(\pm 1/\sqrt3,\pm 1/\sqrt3)$, four of them | $1,1,1,1$ |
| `cps4r` | $(0,0)$ | $4$ |

The $2\times2$ Gauss rule is exact for polynomials of degree $\le 3$ in each
variable. On the regular rectangular mesh this project uses, $J$ is constant, the
entries of $B$ are affine in $(\xi,\eta)$, and the integrand $B^{T}CB\,|J|$ is
therefore at most quadratic in each variable: **`cps4` computes $K_e$ exactly, not
approximately**. This is what makes the stabilisation below exact rather than
heuristic, and it is a property of the mesh regularity, not of the element.

The one-point rule is exact only for an integrand constant over the element,
which here means a displacement field of constant strain.

## The eight modes, and the two that go missing

The element has eight degrees of freedom, spanned by

- **3 rigid-body modes** — two translations and the infinitesimal rotation;
- **3 constant-strain modes** — one per component of $\boldsymbol{\varepsilon}$;
- **2 hourglass modes** — the remainder.

The hourglass modes are built from the node-ordered sign pattern
$\mathbf{h}=(1,-1,1,-1)$, applied to $u_x$ and $u_y$ independently:

$$\mathbf{h}_x=(1,0,-1,0,1,0,-1,0)^{T},\qquad
\mathbf{h}_y=(0,1,0,-1,0,1,0,-1)^{T}.$$

That pattern is exactly the $\xi\eta$ term of the bilinear basis,

$$\sum_{a=1}^{4} h_a N_a(\xi,\eta) = \xi\eta,$$

whose natural derivatives are $\partial(\xi\eta)/\partial\xi=\eta$ and
$\partial(\xi\eta)/\partial\eta=\xi$. **Both vanish at the centroid.** A single
central integration point therefore measures exactly zero strain for a
displacement field that is manifestly deformed — this is the whole problem, and
it is a property of where the point sits, not of the material.

The consequence is a rank deficiency. With $C$ positive definite, $B_0$ at the
centroid has rank 3, so

$$\operatorname{rank}\big(K^{1pt}\big)=3,\qquad
\operatorname{rank}\big(K^{4pt}\big)=5 .$$

The kernel of $K^{4pt}$ is exactly the three rigid-body modes, as it must be. The
kernel of $K^{1pt}$ is five-dimensional: the three rigid modes *plus the two
hourglass modes*. Assembled over a mesh, these zero-energy modes propagate and
the global stiffness is singular or nearly so. Exposing the reduced element
without stabilisation is not an option.

## The stabilisation

For the regular mesh used here, the stabilisation is the difference between the
fully integrated and the one-point stiffness of the **same** reference operator,

$$K_{hg}=\beta\left(K^{4pt}_{ref}-K^{1pt}_{ref}\right),
\qquad 0<\beta\leq1,$$

and the element operator actually assembled is
$K_e = K^{1pt}(C_{\text{tangent}}) + K_{hg}$, where the one-point term uses the
true constitutive tangent and $K_{hg}$ uses the fixed elastic reference.

This form is due to Flanagan and Belytschko, who introduced it for the uniform-strain
hexahedron and quadrilateral. Three properties follow from the definition alone,
with no hand-built projector and no tuned parameter:

**It is orthogonal to rigid and affine motion by construction.** Any field the
one-point rule integrates exactly — every rigid-body motion, every affine
displacement, since their strain is constant over the element — contributes
identically to both terms and therefore nothing to their difference. The element
passes the constant-strain patch test for any $\beta$. This is stronger than
orthogonality enforced by projection, which holds only to the accuracy of the
projector.

**At $\beta=1$ it is not an approximation.** With a constant tangent,
$K^{1pt}(C)+1\cdot(K^{4pt}(C)-K^{1pt}(C))=K^{4pt}(C)$ identically. In the linear
elastic range CPS4R *is* CPS4, to round-off, including when hourglass modes are
genuinely excited.

**$K_{hg}$ has rank 2**, is symmetric, and is positive semi-definite; it acts on
the two hourglass modes and on nothing else. All three are asserted, not assumed.

The control is stiffness-based. No viscous stabilisation is used: a viscous term
makes the answer depend on the increment duration, which is unacceptable for a
quasi-static solve whose increment count is itself under study. Viscous and
assumed-strain alternatives exist and are cited below; they are not used here.

The role of $\beta$ is to soften the hourglass modes only. It scales $K_{hg}$
linearly and leaves the response to every affine field untouched. Values below
one are therefore a deliberate trade: less artificial stiffness in the hourglass
modes, at the cost of losing the exact CPS4 equivalence.

**After yielding, $\beta=1$ is the worst choice, not the natural one.** The
assembled element is $K^{1pt}(C_{\text{tangent}}) + K_{hg}(C_{\text{elastic}})$.
Once the material yields, the constitutive tangent collapses while the
stabilisation keeps the full elastic reference, so the hourglass modes retain
elastic stiffness while every other mode softens: the element is over-stiffened
exactly where CPS4 would have softened. Measured on a heterogeneous J2 case,
$\beta=0.1$ lands six times closer to CPS4 on displacement and five times closer
on plastic strain than $\beta=1$, monotonically. No value is nevertheless
recommended, because none of them met the accuracy criterion; see
`validation/cps4r_qualification_results.md` and the sequence in
{doc}`../how-to/use_reduced_integration`.

## The reference operator

$K_{hg}$ is built on an **elastic** reference tangent, held fixed for the whole
solve. For isotropic J2 plasticity that is the plane-stress elasticity matrix.
For a crystal law it is the cubic elastic operator, rotated into the global frame
and condensed under the global plane-stress condition $\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0$.

Writing the 3D Kelvin tangent with the in-plane set
$a=\{xx,yy,xy\}$ and the transverse set $b=\{zz,xz,yz\}$,

$$\begin{bmatrix}\dot{\boldsymbol\sigma}_a\\ \dot{\boldsymbol\sigma}_b\end{bmatrix}
=\begin{bmatrix}C_{aa}&C_{ab}\\C_{ba}&C_{bb}\end{bmatrix}
\begin{bmatrix}\dot{\boldsymbol\varepsilon}_a\\ \dot{\boldsymbol\varepsilon}_b\end{bmatrix},
\qquad \dot{\boldsymbol\sigma}_b=0
\;\Longrightarrow\;
\dot{\boldsymbol\varepsilon}_b=-C_{bb}^{-1}C_{ba}\dot{\boldsymbol\varepsilon}_a,$$

so the condensed in-plane operator is the Schur complement

$$C^{ps}=C_{aa}-C_{ab}\,C_{bb}^{-1}\,C_{ba},$$

converted from Kelvin to the engineering-shear convention by the row and column
factors $(1,1,1/\sqrt2)$. The condition number of $C_{bb}$ is monitored, because a
badly conditioned transverse block would make the condensation meaningless
without making it fail.

The reference is **measured from the actual MFront behaviour** by a zero-strain
probe followed by a revert, not reconstructed from nominal constants. An
isotropic fallback is forbidden and refused rather than silently substituted: for
a 316L cubic crystal at a 30 degree in-plane rotation, an isotropic reference is
wrong by more than 10 percent on $K_{hg}$, which is asserted by the test suite.

Two consequences deserve to be stated plainly. The reference stays elastic after
constitutive yielding, so the stabilisation is a **numerical** energy — never
crystal hardening, plastic dissipation, or a nonlocal interaction. And because
the tangent softens on yielding while $K_{hg}$ does not, the exact CPS4
equivalence proved above is an *elastic* property that does not survive into the
plastic range. That is the technical reason CPS4 remains the reference
formulation.

## Force, energy, and the diagnostic

The stabilisation enters the residual as an ordinary internal force,
$\mathbf{f}_{hg,e}=K_{hg,e}\,\mathbf{u}_e$, assembled alongside the constitutive
internal force and included in the reactions and in every line-search trial. The
stored stabilisation energy of an element is the quadratic form

$$E_{hg,e}=\tfrac12\,\mathbf{u}_e^{T} K_{hg,e}\,\mathbf{u}_e \;\ge\; 0,$$

archived on the element grid as `HOURGLASS_ENERGY_BY_ELEMENT.npy` for CPS4R
campaigns. The global ratio reported by the solver is

$$r_{hg}=\frac{\sum_e E_{hg,e}}{\left|W_{int}\right|},$$

where the internal work is accumulated by the trapezoidal rule over the accepted
equilibrium path,

$$W_{int}=\sum_{n}\tfrac12\left(\mathbf{f}^{\,n}_{int}+\mathbf{f}^{\,n+1}_{int}\right)^{T}
\left(\mathbf{u}^{n+1}-\mathbf{u}^{n}\right),$$

with $\mathbf{f}_{int}$ the complete mechanical internal force, hourglass
contribution included. Failed Newton trials and cutback attempts contribute
nothing: the accumulation happens after the increment is accepted. The final
scalar product $\mathbf{f}_{int}(\mathbf{u}_{final})^{T}\mathbf{u}_{final}$ would
*not* do — it is twice the elastic strain energy on a linear monotonic path and
has no work meaning at all once the material yields.

### What the ratio does and does not say

Read $r_{hg}$ knowing three things about its construction.

It compares a **state** quantity with a **path** quantity. $E_{hg}$ is the elastic
energy stored in the stabilisation at the final configuration; $W_{int}$
accumulates along the whole loading history and includes plastic dissipation.
After appreciable yielding the denominator keeps growing while the numerator does
not, so the ratio decreases as the path lengthens. A longer loading history makes
the diagnostic look better without any change in the element behaviour. Compare
ratios only between runs with comparable loading paths.

It is evaluated **only at the final state**. A transient hourglass excitation that
appears mid-path and unloads is invisible in the ratio.

Most importantly, **it does not predict the error.** This was measured, not
assumed. On a pixel-wise heterogeneous J2 case at `beta = 1`, over 1024 elements,
the correlation between the element hourglass energy and the CPS4-to-CPS4R
plastic-strain error is `r = 0.033`, and between the energy and PEEQ itself
`r = 0.066`. Both are indistinguishable from zero: the stabilisation energy does
not concentrate in the plastic band, and it does not sit where the error is. At
the same time every configuration tested passed a `1 %` ratio by an order of
magnitude while missing the accuracy bound by four to twenty times. See
`validation/cps4r_qualification_results.md`.

**Read `r_hg` as a measure of how hard the stabilisation is working, not of how
wrong the answer is.** An earlier version of this page offered `1 %` and `5 %`
bands as analysis guides. They are withdrawn as a gate: the campaign above found
no relationship between the ratio and the error it was supposed to bound. The
`hourglass_energy_failure_ratio` setting remains available as a blunt guard
against a solve whose stabilisation energy runs away, which is a different
purpose — it protects against a broken run, not against an inaccurate one.

Inspecting the element field beside PEEQ, or beside accumulated slip for a
crystal law, is still worth doing: a visible concentration is informative when it
appears. What is not supported is the converse, that a flat field or a small
average certifies anything.

## What is verified, and what is not

Covered by automated tests, with the asserted tolerances:

- the two rules integrate the parent area, and an unknown formulation is refused;
- the unstabilised one-point stiffness has rank 3, the stabilised element rank 5,
  and its kernel is exactly the three rigid-body modes;
- $K_{hg}$ is symmetric, positive semi-definite, of rank 2;
- the hourglass mode produces zero strain at the central point ($<10^{-15}$) and
  strictly positive stabilisation energy;
- uniform tension, biaxial, shear, rigid rotation and rigid translation produce no
  hourglass force or energy (relative $<10^{-12}$);
- at $\beta=1$ the reduced element reproduces the full one, for anisotropic
  references too, to $10^{-13}$ relative;
- an anisotropic reference changes $K_{hg}$ by more than 10 percent against an
  isotropic one, so the fallback could not have passed unnoticed;
- an unsymmetric, singular, misshapen or zero reference tangent is refused, while
  round-off asymmetry is absorbed;
- through the whole solver: four times fewer material points, agreement with CPS4
  to $10^{-10}$ on an affine load, and — the sharp test — agreement to $10^{-10}$
  on a **non-affine** elastic load where the hourglass energy is genuinely nonzero;
- the reported ratio equals $E_{hg}/W_{int}$, and $W_{int}$ matches its analytical
  value on a linear path.

Measured by the qualification campaign, preregistered in
`validation/cps4r_qualification_preregistration.md` and reported in
`validation/cps4r_qualification_results.md`:

- **the elastoplastic accuracy criterion fails at every $\beta$**, on a
  pixel-wise heterogeneous J2 case and on a tilted-orientation SRIX case. The
  plastic-strain relative error against CPS4 runs from 1.9 to 10 percent against
  a 0.5 percent bound derived from the reproduction error the project already
  accepts;
- the cost case holds: constitutive time falls by 3.7 to 4.8 times and total
  wall time by 1.9 to 2.9 times, the larger figure being the crystal law;
- the displacement difference between the two formulations is 30 to 200 times
  **below** the DIC measurement noise. The failure is one of numerical
  self-consistency, not of measurable physics — which is a reason to state the
  result precisely, not a licence to ignore it.

**CPS4R is therefore not authorised for scientific elastoplastic campaigns, and
no value of $\beta$ is recommended. CPS4 remains the reference formulation and
the default.** What would change that verdict is listed in the results document:
a mesh-convergence study, a stabilisation built on the current tangent rather
than the fixed elastic reference, and an error estimator that actually predicts
the difference.

CPS4R is also deliberately refused in combination with the micromorphic nonlocal
coupling: a hourglass mode inside a localisation band would be indistinguishable
from the physics that coupling exists to capture, and the interaction has not been
validated.

## References

Numerical integration and hourglass control:

- D. P. Flanagan and T. Belytschko, *A uniform strain hexahedron and quadrilateral
  with orthogonal hourglass control*, International Journal for Numerical Methods
  in Engineering **17**(5), 679–706, 1981.
  DOI [10.1002/nme.1620170504](https://doi.org/10.1002/nme.1620170504).
  The origin of the stiffness control used here.
- T. Belytschko, J. S.-J. Ong, W. K. Liu and J. M. Kennedy, *Hourglass control in
  linear and nonlinear problems*, Computer Methods in Applied Mechanics and
  Engineering **43**(3), 251–276, 1984.
  DOI [10.1016/0045-7825(84)90067-7](https://doi.org/10.1016/0045-7825(84)90067-7).
  Stiffness against viscous control, and the consistency conditions a
  stabilisation must satisfy.
- D. Kosloff and G. A. Frazier, *Treatment of hourglass patterns in low order
  finite element codes*, International Journal for Numerical and Analytical
  Methods in Geomechanics **2**(1), 57–72, 1978.
  DOI [10.1002/nag.1610020105](https://doi.org/10.1002/nag.1610020105).
  Earlier identification of the spurious modes.
- T. Belytschko and W. E. Bachrach, *Efficient implementation of quadrilaterals
  with high coarse-mesh accuracy*, Computer Methods in Applied Mechanics and
  Engineering **54**(3), 279–301, 1986.
  DOI [10.1016/0045-7825(86)90107-6](https://doi.org/10.1016/0045-7825(86)90107-6).
  The quadrilateral-specific treatment.
- O. C. Zienkiewicz, R. L. Taylor and J. M. Too, *Reduced integration technique in
  general analysis of plates and shells*, International Journal for Numerical
  Methods in Engineering **3**(2), 275–290, 1971.
  DOI [10.1002/nme.1620030211](https://doi.org/10.1002/nme.1620030211).
  The introduction of reduced integration.

Alternatives not used here, for the record:

- T. Belytschko and L. P. Bindeman, *Assumed strain stabilization of the eight
  node hexahedral element*, Computer Methods in Applied Mechanics and Engineering
  **105**(2), 225–260, 1993.
  DOI [10.1016/0045-7825(93)90124-G](https://doi.org/10.1016/0045-7825(93)90124-G).
- J. C. Simo and M. S. Rifai, *A class of mixed assumed strain methods and the
  method of incompatible modes*, International Journal for Numerical Methods in
  Engineering **29**(8), 1595–1638, 1990.
  DOI [10.1002/nme.1620290802](https://doi.org/10.1002/nme.1620290802).

Textbooks, for the shape functions, the Gauss rules and the rank counting:

- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite
  Element Analysis*, Dover, 2000. ISBN 978-0-486-41181-1.
- T. Belytschko, W. K. Liu, B. Moran and K. I. Elkhodary, *Nonlinear Finite
  Elements for Continua and Structures*, 2nd ed., Wiley, 2014.
  ISBN 978-1-118-63270-3.
- O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, *The Finite Element Method: Its
  Basis and Fundamentals*, 7th ed., Butterworth-Heinemann, 2013.
  ISBN 978-1-85617-633-0.

Industrial precedent for stiffness hourglass control on a reduced-integration
continuum element, and for exposing its energy as a diagnostic:

- Abaqus Analysis User's Guide, *Section controls* and *Solid (continuum)
  elements*. `ALLAE` is the corresponding artificial strain energy; the practice
  of comparing it with the internal energy is the same one adopted here.

The crystal-plasticity references behind the reference operator are in
{doc}`forest_rubin_srix`.
