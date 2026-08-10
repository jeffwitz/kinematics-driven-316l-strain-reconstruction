# Structural plane stress for three-dimensional MFront behaviours

## Scope

`StructuralPlaneStress3D` is a constitutive closure for small-strain,
three-dimensional, `Implicit` MFront behaviours compatible with the standard
elastic split. The behaviour retains its complete three-dimensional internal
state. The closure is local to each material point and is suitable for a
two-dimensional membrane formulation; it does not resolve through-thickness
warping, layers, bending, or a three-dimensional displacement field.

The transverse kinematics are local constitutive unknowns, not additional FEM
degrees of freedom.

## Traction-free structural plane stress

For a thin body with structural normal \(n=e_z\), traction-free faces satisfy

$$
\boldsymbol\sigma n=0\quad\text{on the two faces}.
$$

The plane-stress approximation extends this condition to the material point
representing the thickness:

$$
\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0.
$$

Define the in-plane and transverse Kelvin components by

$$
a=(xx,yy,xy),\qquad b=(zz,xz,yz).
$$

The three components \(\varepsilon_b\) are relaxed locally so that the three
transverse tractions vanish. In an aligned isotropic special case the shear
conditions decouple, which is why conventional plane-stress formulas often
appear to solve only \(\sigma_{zz}=0\). For an arbitrarily oriented anisotropic
or crystal law, the coupling is generally nonzero and imposing only
\(\sigma_{zz}=0\) is insufficient.

## Relation to standard MFront hypotheses

MFront's `StandardElasticity` already supports the standard `PlaneStress`
hypothesis by introducing an axial strain and enforcing the corresponding
axial stress condition. It also provides axial-stress relaxation machinery for
the axisymmetric generalised-plane-stress case. Those mechanisms relax an
axial component; they do not provide the three-component structural relaxation
needed here while retaining a `Tridimensional` constitutive state and an
arbitrary structural/material orientation.

`StructuralPlaneStress3D` complements, rather than replaces, these standard
hypotheses.

## Tensor and Kelvin notation

Let \(Q\in SO(3)\) map global components to material components:

$$
A_m=Q A_g Q^T.
$$

Let \(\mathcal R_K(Q)\) be the induced \(6\times6\) Kelvin operator. Then

$$
[A_m]_K=\mathcal R_K(Q)[A_g]_K,
\qquad
[A_g]_K=\mathcal R_K(Q)^T[A_m]_K.
$$

Kelvin strain components use \(\sqrt2\,\varepsilon_{ij}\) for shear terms;
engineering components use \(\gamma_{ij}=2\varepsilon_{ij}\). All internal
closure equations use Kelvin components. Engineering components appear only at
the FEM/configuration interface.

For a raw three-dimensional tangent, both its input and output frames change:

$$
C_g=\mathcal R_K(Q)^T C_m\mathcal R_K(Q).
$$

The structural closure constructs its constrained operator directly in the
structural frame.

## Generic residual transformation

For a `StandardElasticity`-compatible behaviour, the first six elastic rows
have the standard form

$$
f_e=K_m-g,
$$

where `g` is the structural gradient stored in the standard `deto` slot and
\(K_m\) is the constitutive elastic kinematics reconstructed by the behaviour.
The current implementation uses the algebraic access point

$$
K_m=f_e+g.
$$

It then forms \(K_g=\mathcal R_K(Q)^T K_m\) and replaces the six elastic rows
with

$$
F_a=(K_g)_a-g_a,
\qquad
F_b=\frac{(\sigma_g)_b}{S_{ref}},
$$

where \(S_{ref}>0\) is a residual scaling modulus. All other constitutive
rows remain untouched. The transformation acts on every Jacobian column, so
it does not need to know whether those columns represent slips, hardening
variables, or another implicit state.

The V1 contract is deliberately explicit:

- `@DSL Implicit` and `@ModellingHypothesis Tridimensional`;
- small-strain symmetric tensor gradient;
- standard elastic/inelastic additive split;
- the first six rows form the standard elastic block;
- stress is controlled by the elastic strain block and known point properties,
  without a direct dependence on another implicit variable;
- the behaviour exposes the structural orientation used by the closure.

The registered full-field generator currently adds a crystal-output contract
for the SRIX and Méric behaviours. This packaging restriction is narrower than
the mathematical transformation itself.

## One-step consistent tangent

At convergence let \(z\) contain every local implicit unknown and

$$
A=\frac{\partial F}{\partial z}.
$$

Only the three in-plane imposed components are independent. Define
\(E_a\in\mathbb R^{n\times3}\) by

$$
(E_a)_{0,0}=(E_a)_{1,1}=(E_a)_{3,2}=1
$$

with all other entries zero. Since \(F_a=K_{g,a}-g_a\), the sensitivities
solve

$$
A X=E_a,\qquad X=\frac{\partial z}{\partial g_a}.
$$

No explicit inverse is formed. If \(X_e\) denotes the six elastic rows of
\(X\), the full stress response is

$$
\widehat C^{PS}=\mathcal R_K(Q)^T D_m X_e\in\mathbb R^{6\times3}.
$$

Let \(S_a\in\mathbb R^{3\times6}\) select the in-plane stress components.
The two-dimensional consistent tangent is

$$
\boxed{C^{PS}=S_a\widehat C^{PS}}
$$

and is a \(3\times3\) operator. The transverse rows of
\(\widehat C^{PS}\) provide a direct check that the traction constraints remain
zero under in-plane perturbations.

## Equivalence with external condensation

The raw three-dimensional local equations can be written schematically as

$$
K_a(z)-g_a=0,\qquad K_b(z)-g_b=0,\qquad G(z)=0,
$$

where \(G\) contains the remaining constitutive equations. External
condensation solves the transverse equation for \(g_b=K_b(z)\), then imposes
the three traction equations. Eliminating this same equation before Newton
gives

$$
K_a(z)-g_a=0,\qquad G(z)=0,\qquad \sigma_b(z)=0,
$$

which is exactly the structural closure system. Therefore external 3D
condensation and the monolithic transformation are two eliminations of the
same local equations, provided they use the same state, orientation, branch,
and one-step increment.

For a smooth linear tangent partitioned into \(a,b\), the resulting operator
is the usual Schur complement

$$
C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
$$

## Host integration and composite tangents

The closure supplies the tangent of one constitutive step. A host may still
integrate difficult points through a sequence of substeps. If the substep
targets depend on the final target strain, the derivative is propagated through
the implicit substep equations; it is not simply the last one-step tangent.
The host composite finite-difference option approximates the derivative of
that complete algorithmic map. Adaptive partition changes make this map
piecewise smooth, so the result is a local algorithmic secant when the
perturbed branches differ.

Substepping, failure caches, composite finite differences, and shadow
diagnostics are host concerns. They are not part of the `StructuralPlaneStress3D`
constitutive closure.

## Verification scope

The transformation has been verified at material-point level for rotated
anisotropic elasticity, J2 plasticity, Forest–Rubin SRIX, and
Méric–Cailletaud. Full-field registration uses the same host adapter for SRIX
and Méric. The current V1 implementation is therefore qualified for the
explicit contract above, not for arbitrary MFront behaviours.

The external three-dimensional condensation backend remains the independent
reference for new behaviours and for checking the structural closure.
