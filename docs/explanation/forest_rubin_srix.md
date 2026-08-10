# The Forest–Rubin SRIX crystal law

## Model scope

SRIX is a rate-independent crystal-plasticity law for FCC 316L. It uses twelve
slip systems and an additive small-strain split. The constitutive state retains
the twelve signed plastic slips and the corresponding isotropic and kinematic
hardening variables. It is therefore a genuinely crystal-resolved law, not a
J2 surrogate.

The law is appropriate for monotonic or incrementally prescribed loading where
the material parameters and orientation data are known. It does not identify
the 316L parameters from a DIC field by itself, and it does not represent
finite-strain lattice rotation or through-thickness heterogeneity.

## Kinematics and resolved shear

For each slip system \(s\), let \(M_s\) be the symmetric Schmid tensor in the
material frame. The inelastic increment is

$$
\Delta\varepsilon^p=\sum_s\Delta\gamma_s M_s.
$$

The resolved shear stress is

$$
\tau_s=\boldsymbol\sigma:M_s.
$$

The elastic strain and stress follow the cubic elastic law supplied by
`StandardElasticity`. Crystal orientations map the structural/global frame to
the material frame before these quantities are evaluated.

## Forest–Rubin flow rule

The slip resistance has an initial value \(\tau_0\) and an interaction term

$$
r_s=\tau_0+Q\sum_j m_{sj}\left(1-e^{-b p_j}\right),
$$

where \(m_{sj}\) is the FCC interaction matrix and \(p_j\) accumulates
\(|\Delta\gamma_j|\). The backstrain update is implicit; in the notation of
the implementation,

$$
\Delta a_s=
\frac{\Delta\gamma_s-d\,a_s|\Delta\gamma_s|}
     {1+\theta d|\Delta\gamma_s|}.
$$

The driving force is \(\tau_s-C(a_s+\theta\Delta a_s)\). The rate-independent
flow rule uses its sign and the Macaulay positive part of the overstress. The
inactive-system slope is explicitly zero; otherwise the constant SRIX flow
slope would inject a spurious coupling into inactive equations.

## Semismooth local Newton linearisation

The value of \(|\Delta\gamma|\) is continuous but not differentiable at zero.
The production Jacobian uses

$$
\frac{d|\Delta\gamma|}{d\Delta\gamma}=
\begin{cases}+1,&\Delta\gamma>0,\\-1,&\Delta\gamma<0,\\0,&\Delta\gamma=0.\end{cases}
$$

The last value is the symmetric element of the Clarke generalized derivative
of the absolute value. It changes the local linearisation only; it does not
smooth the residual or alter the committed state update. The precise
mathematical statement and references are in
{doc}`../reference/numerics/srix_semismooth_jacobian`.

## Implicit equivalent strain increment

The SRIX flow amplitude depends on the equivalent deviatoric strain increment
\(\Delta\bar\varepsilon\). In an implicit solve this quantity must depend on
the local unknowns, including the transverse relaxation variables when the law
is used with structural plane stress.

Writing it from the current unknown elastic strain and slip increments gives

$$
\Delta\bar\varepsilon
=\sqrt{\frac{2}{3}\,d:d},
\qquad
d=\operatorname{dev}(\Delta\varepsilon^e)+
\sum_s\Delta\gamma_s M_s.
$$

This makes the dependence of the flow amplitude visible to the consistent
Jacobian. Treating \(\Delta\bar\varepsilon\) as a fixed value taken only from
the imposed in-plane increment would omit that coupling.

## Orientation and structural plane stress

The law itself remains three-dimensional. In a two-dimensional calculation,
use `mfront-structural-plane-stress` or the independent external condensation
backend. Both relax \(\varepsilon_{zz},\gamma_{xz},\gamma_{yz}\) locally so that
the three transverse tractions vanish. The crystal orientation is supplied per
material point from the EBSD provider; no external rotation of the imposed
gradient is required.

## Interpretation and limits

The primary observables are signed system slips, accumulated absolute slip,
and system-wise hardening. A scalar equivalent plastic strain is not a native
SRIX quantity. Slip localisation depends on the orientation field, loading
path, elastic constants, interaction matrix, and boundary data; it should not
be interpreted independently of those inputs.

The law is small-strain, rate-independent, and local. Viscous regularisation,
finite-strain lattice evolution, damage, and explicit through-thickness
heterogeneity require a different constitutive model or an additional host
formulation.
