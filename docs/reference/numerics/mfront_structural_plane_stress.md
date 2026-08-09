# Structural plane stress from generic three-dimensional MFront behaviours

**Category: scientific reference and upstream design note.**

This note describes a structural plane-stress closure for a genuinely
three-dimensional constitutive behaviour. It is not a plate or shell theory:
the finite-element model is a two-dimensional membrane model with a local
constitutive closure at each material point.

The demonstrated V1 domain is deliberately narrow:

$$
\boxed{\text{small strain} + \text{Implicit} + \text{Tridimensional}
       + \text{StandardElasticity-compatible elastic split}.}
$$

The constitutive state remains three-dimensional. The generic mathematical
transformation has been demonstrated on rotated elasticity, J2, SRIX and
Méric--Cailletaud; the currently registered full-field implementation still
uses SRIX-specific source-generation scaffolding, documented in Section 12.

## 1. Structural assumptions and the meaning of plane stress

For a body with structural thickness coordinate \(z\), a free face has the
boundary condition

$$
\boldsymbol\sigma\mathbf n=\mathbf 0
\qquad\text{on }z=+h/2\text{ and }z=-h/2.
$$

The two-dimensional plane-stress approximation extends this surface condition
to the representative constitutive state through the thickness of a
sufficiently thin membrane. Under that approximation, with
\(\mathbf n=\mathbf e_z\), the local closure is

$$
\boxed{\sigma_{zz}=0,\qquad \sigma_{xz}=0,\qquad \sigma_{yz}=0.}
$$

This is an approximation to a three-dimensional boundary-value problem, not a
claim that the traction condition holds at every point of a finite-thickness
body. Bending, transverse shear and through-thickness variation are outside
this formulation.

The FEM unknowns remain only

$$
u_x(x,y),\qquad u_y(x,y).
$$

The three transverse strains

$$
\varepsilon_b=(\varepsilon_{zz},\gamma_{xz},\gamma_{yz})
$$

are local constitutive unknowns. They are solved and condensed at a material
point; they are not additional FEM degrees of freedom.

## 2. Relation to existing MFront plane-stress hypotheses

MFront and `StandardElasticity` already provide powerful support for standard
`PlaneStress` and `GeneralisedPlaneStress` modelling hypotheses. In the usual
case, the extra local unknown is the axial strain \(\varepsilon_{zz}\), and the
additional equation is \(\sigma_{zz}=0\). This is exactly the right
abstraction for an isotropic material, or an aligned material for which the
transverse shear components are decoupled.

The problem addressed here is different:

| Existing standard hypothesis | Structural 3D closure in this note |
|---|---|
| additional local unknown typically \(\varepsilon_{zz}\) | \(\varepsilon_{zz},\gamma_{xz},\gamma_{yz}\) |
| closure typically \(\sigma_{zz}=0\) | \(\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0\) |
| behaviour represented under a plane-stress hypothesis | behaviour remains `Tridimensional` |
| suitable for the standard decoupled case | suitable for arbitrary 3D material orientation |

The proposal does not replace MFront's existing plane-stress mechanisms. It
targets a three-dimensional law with a complete three-dimensional internal
state and a structural surface orientation that is independent of the crystal
orientation. In particular, an EBSD crystal orientation is not generally
aligned with the structural normal.

## 3. Tensor and Kelvin notation

The scientific derivation uses Kelvin components. The structural engineering
interface uses engineering shear components. With component order

$$
(xx,yy,zz,xy,xz,yz),
$$

the engineering strain and Kelvin strain vectors are

$$
[\varepsilon]_{eng}=
\begin{bmatrix}\varepsilon_{xx}&\varepsilon_{yy}&\varepsilon_{zz}&
\gamma_{xy}&\gamma_{xz}&\gamma_{yz}\end{bmatrix}^{T},
$$

$$
[\varepsilon]_{K}=
\begin{bmatrix}\varepsilon_{xx}&\varepsilon_{yy}&\varepsilon_{zz}&
\sqrt2\varepsilon_{xy}&\sqrt2\varepsilon_{xz}&\sqrt2\varepsilon_{yz}
\end{bmatrix}^{T},
\qquad \gamma_{ij}=2\varepsilon_{ij}.
$$

The stress Kelvin vector is

$$
[\sigma]_K=
\begin{bmatrix}\sigma_{xx}&\sigma_{yy}&\sigma_{zz}&
\sqrt2\sigma_{xy}&\sqrt2\sigma_{xz}&\sqrt2\sigma_{yz}\end{bmatrix}^{T}.
$$

Let \(Q\in SO(3)\) map global/structural tensors to material tensors:

$$
\boldsymbol A_m=Q\boldsymbol A_gQ^T,
\qquad
\boldsymbol A_g=Q^T\boldsymbol A_mQ.
$$

The corresponding Kelvin operator is a distinct six-dimensional object,
\(\mathcal R_K(Q)\in\mathbb R^{6\times6}\):

$$
[A_m]_K=\mathcal R_K(Q)[A_g]_K,
\qquad
[A_g]_K=\mathcal R_K(Q)^T[A_m]_K.
$$

Thus no 3-by-3 matrix is ever multiplied directly by a six-component vector.
The raw 3D bridge applies \(\mathcal R_K(Q)\) to the input gradient and its
transpose to returned stresses and tangents. GPS behaviours receive the
structural gradient and own the crystal rotation through per-point `Qij`
properties. Applying both rotations would rotate the gradient twice.

## 4. Why \(\sigma_{zz}=0\) alone fails for rotated anisotropy

Partition the structural Kelvin components as

$$
a=(xx,yy,xy),\qquad b=(zz,xz,yz).
$$

The linearized response is

$$
\begin{bmatrix}\sigma_a\\\sigma_b\end{bmatrix}
=
\begin{bmatrix}C_{aa}&C_{ab}\\C_{ba}&C_{bb}\end{bmatrix}
\begin{bmatrix}\varepsilon_a\\\varepsilon_b\end{bmatrix}.
$$

For a rotated anisotropic material, \(C_{ba}\ne0\) in general. A purely
in-plane strain therefore produces transverse shear traction. Fixing
\(\gamma_{xz}=\gamma_{yz}=0\) and adjusting only \(\varepsilon_{zz}\) can make
\(\sigma_{zz}=0\) while leaving \(\sigma_{xz}\) and \(\sigma_{yz}\) non-zero.

Crystal plasticity makes the same point independently of elasticity:

$$
\Delta\boldsymbol\varepsilon^p=
\sum_s\Delta\gamma_s\,\mathbf M_s.
$$

After an arbitrary EBSD orientation, each Schmid tensor can have all six
structural components. The three transverse strains must therefore be solved
so that the complete traction vector vanishes.

## 5. External 3D condensation: the independent reference

The backend `mfront-3d-condensed-plane-stress` leaves the behaviour unchanged.
It solves the three local equations

$$
\sigma_b(\varepsilon_a,\varepsilon_b)=0
$$

with an outer Newton iteration. Every trial is restored from one complete
committed snapshot. After convergence, the constrained tangent is

$$
\boxed{C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.}
$$

The inverse is not formed: a 3-by-3 linear system is solved. The bridge can
use either a committed or tangent transverse predictor, monitors the
conditioning of \(C_{bb}\), and records the local closure iterations. This is
the independent reference because it can use an unchanged compatible 3D law.

## 6. Exact equivalence of external and monolithic closure

The equivalence is easiest to see before differentiating. Let \(x\) contain
the local constitutive unknowns and write the raw 3D system as

$$
K_a(x)-\varepsilon_a=0,\qquad
K_b(x)-\varepsilon_b=0,\qquad
G(x)=0,
$$

where \(G\) denotes all other constitutive equations. Plane stress adds

$$
S_b(x)=0.
$$

The second raw equation gives \(\varepsilon_b=K_b(x)\). Eliminating it leaves

$$
K_a(x)-\varepsilon_a=0,\qquad G(x)=0,\qquad S_b(x)=0,
$$

which is exactly the monolithic structural closure. Therefore, on the same
committed state, increment, orientation and differentiable constitutive
branch,

$$
\boxed{\text{external 3D condensation}
\equiv\text{monolithic structural closure}.}
$$

The Schur tangent and the monolithic implicit tangent are two eliminations of
the same local equations, not two different physical models.

## 7. Generic MFront residual transformation

The specialised SRIX behaviour writes the closure equations explicitly. The
generic prototype instead uses the standard elastic residual as an algebraic
access point. In the current generated hook, `StandardElasticity` has already
assembled a block of the form

$$
f_e^{std}=K_m-g,
$$

where `g` is the six-component gradient stored in the standard `deto` slot.
Here `g` is the structural gradient supplied by the host; it must not be
renamed \(\Delta\varepsilon_m\). Before the standard rows are consumed by
Newton, the hook recovers

$$
K_m=f_e^{std}+g.
$$

This is an implementation-level algebraic extraction, not a statement that a
structural gradient is a material-frame tensor. A future upstream interface
should expose \(K_m\) or the elastic block directly instead of relying on the
generated residual layout.

The recovered tensor is transformed with \(\mathcal R_K(Q)^T\). The six
elastic rows are replaced by

$$
F_a=(K_g)_a-g_a,
\qquad
F_b=\frac{(\sigma_g)_b}{S_{ref}},
$$

while all other constitutive rows are left unchanged. All columns of the
elastic-row Jacobian are transformed, including columns belonging to unknowns
whose names and physical meanings are not known to the closure.

The current V1 Jacobian transformation additionally assumes

$$
\sigma_m=D_m e_m^{el},
$$

with no direct dependence of stress on another implicit variable. If, for
example, \(\sigma=\sigma(e^{el},d)\) with an implicit damage variable \(d\),
then \(\partial\sigma/\partial d\ne0\) and the currently zeroed transverse
columns would not be correct. This is part of the contract, not an accidental
detail.

## 8. Residual scaling

The traction rows are scaled as

$$
F_b=\sigma_b/S_{ref},\qquad S_{ref}>0.
$$

The root is unchanged, but the scaling affects local Newton conditioning.
The current generated prototype uses the fixed value
\(S_{ref}=210000\) MPa. This is a prototype limitation, not a generic physical
constant. An upstream implementation should derive the scale from a
representative elastic modulus or expose it as a well-documented behaviour
parameter.

## 9. One-step consistent tangent

At convergence, let

$$
F(x,g_a)=0,\qquad A=\frac{\partial F}{\partial x}.
$$

Let \(P\) contain the dependence on the three imposed structural components;
its active columns are `xx`, `yy` and `xy`, with the sign convention taken
from the actual transformed residual. Solve

$$
AX=P,
\qquad X=\frac{\partial x}{\partial g_a}.
$$

The implementation solves this system and never forms \(A^{-1}\). If \(X_e\)
denotes the elastic-state rows and \(D_m\) is the material-frame
stress/elastic-strain tangent, then

$$
C^{PS}=\mathcal R_K(Q)^T D_m X_e.
$$

The inactive structural columns are not independent input columns and are set
to zero in the returned operator used by the 2D solver. This is the tangent of
one constitutive step. The raw 3D route gives the same derivative through

$$
C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
$$

## 10. Host substepping and the composite tangent

The implementation has three distinct derivative layers:

```text
3D constitutive physics
        ↓
one-step structural closure and tangent
        ↓
host substepping policy
        ↓
derivative of the composed algorithm
        ↓
global FEM Newton
```

For a fixed sequence of maps, a simplified notation is

$$
\Phi=\Phi_N\circ\cdots\circ\Phi_1,
\qquad D\Phi=D\Phi_N\cdots D\Phi_1.
$$

The actual driver distributes the target increment over the sub-steps,

$$
\Delta g_k=\frac{1}{N}(g_{target}-g_n),
$$

so each map also depends directly on the final target. With

$$
F_k(z_k,S_{k-1},\Delta g_k)=0,
$$

the sensitivity obeys the recurrence

$$
A_k\frac{dz_k}{dg_{target}}
=-
\left(F_{S,k}\frac{dS_{k-1}}{dg_{target}}
 +F_{\Delta g,k}\frac{d\Delta g_k}{dg_{target}}\right).
$$

The product formula is therefore only an illustration; the direct target
terms and state propagation are part of the true composite derivative.

The qualified adapter reconstructs this derivative by central finite
differences for points that actually sub-step. If the plus and minus
perturbations follow the same partition, the result approximates the local
derivative of the algorithmic map. If the adaptive partition changes, the map
is only piecewise smooth and the finite difference is a secant/generalised
Jacobian diagnostic rather than a classical derivative. The partition and any
branch mismatch are recorded.

## 11. Numerical controls and algorithmic choices

| Control | Mathematical role | Production/diagnostic |
|---|---|---|
| `local_transverse_predictor=committed` | starts \(\varepsilon_b\) from the last committed state | production option |
| `local_transverse_predictor=tangent` | uses the previous transverse tangent to predict the new state | qualified production option |
| absolute/relative closure tolerances | scales the three traction residuals | production |
| `local_condition_check_mode=always` | checks \(C_{bb}\) after every local solve | production option |
| `on_failure` | checks conditioning only after a failed local solve | production default in GPS diagnostics |
| `diagnostic_sample` | samples conditioning without paying for every point | diagnostic/performance option |
| maximum local iterations | bounds the transverse Newton | production safety bound |
| `maximum_substeps=256` | maximum GPS path subdivision | production safety bound |
| `minimum_substep_span=1` | prevents further subdivision of one-point spans | production safety bound |
| failure-span bisection/cache | isolates failing points and reuses known spans | production GPS policy |
| `gps_composite_fd_tangent` | replaces last-substep tangent on selected points | qualified production GPS option |
| `gps_composite_fd_step=1e-6` | absolute engineering-strain central-FD step | production numerical option |
| `gps_shadow_tangent` | compares with a raw full-step diagnostic tangent | diagnostic only |
| `gps_shadow_tangent_scope` | selects all, substepped or non-substepped points | diagnostic only |
| MGIS thread count | parallelizes constitutive batches | performance setting |

The shadow tangent must not be enabled in production. The composite FD is
sparse: on the qualified M100 run it touched 192 points and 1152 trajectories.

## 12. What is generic, and what is still prototype-specific

Three levels must be distinguished:

1. **Generic mathematical transformation.** The residual/Jacobian idea does
   not refer to SRIX, Méric, slips or hardening and has been demonstrated on
   rotated elasticity, J2, SRIX and Méric.
2. **Generic prototype closure/tangent.** The probes transform all local
   Jacobian columns and derive the local system size from the Jacobian type.
3. **Registered full-field backend.** The current generated SRIX backend still
   uses source-rewriting scaffolding, an exposed
   `StructuralJacobian[324]` auxiliary buffer and SRIX-oriented output
   extraction. Thus the industrialised field backend is not yet arbitrary-law
   generic. In particular, `324=18^2` is a current implementation limitation
   and a law with a different local system size is not yet covered by this
   production transport path.

The external Behaviour Brick route was investigated, but the installed TFEL
5.1 ABI does not export the required `BehaviourBrickBase` symbols. No TFEL fork
is used. The honest current genericity claim is therefore:

> The transformation is mathematically generic for the demonstrated V1
> contract; the registered full-field implementation is currently qualified
> for SRIX.

## 13. Verification and qualification

The point-material evidence is:

| Behaviour | transverse traction | in-plane kinematic error | tangent check | what it proves |
|---|---:|---:|---:|---|
| rotated anisotropic elasticity | \(2.3\times10^{-14}\) MPa | \(4.9\times10^{-19}\) | Schur \(6.95\times10^{-16}\) | frame transformation and elastic closure |
| J2 plastic | \(7.99\times10^{-14}\) MPa | — | FD \(3.12\times10^{-11}\) at \(h=10^{-7}\) | all local columns can be transformed |
| SRIX | \(9.30\times10^{-15}\) MPa | \(1.74\times10^{-18}\) | FD \(2.48\times10^{-10}\) at \(h=10^{-7}\) | crystal plasticity closure and one-step tangent |
| Méric--Cailletaud | \(6.77\times10^{-15}\) MPa | \(1.42\times10^{-19}\) | FD \(O(10^{-13})\) | second implicit crystal law, same closure code |

The generic SRIX and Méric one-step tangents have been checked by finite
differences. A dedicated, archived same-state comparison of those two generic
variants against a live raw-3D Schur oracle is not yet available; the Schur
identity is therefore a required next validation artifact, not a claim made by
this table.

The P43 M100 single-run comparison used the same EBSD crop, eight increments,
four MFront threads, one BLAS thread and one FFTW thread:

| strategy | Newton | sequence | elapsed |
|---|---:|---|---:|
| 3D + Python condensation | 57 | `[6,6,7,7,7,8,8,8]` | 56.72 s |
| specialised GPS + composite FD | 58 | `[6,6,7,7,7,8,8,9]` | 51.65 s |
| generic `StructuralPlaneStress3D` + composite FD | 58 | `[6,6,7,7,7,8,8,9]` | 54.56 s |

The generic and specialised GPS fields agree at approximately `1e-12` relative
L2 for stress, reactions and crystal slips, and at `1e-16` for displacement.
Both used 192 substepped points. These are single-campaign measurements, not
universal speed guarantees. The complete comparison is archived in
`validation/_generated/performance/p43_m100_backend_comparison_latest.json`.

## 14. Current source notes and upstream design

The generated prototype uses \(S_{ref}=210000\) MPa and the fixed 18-by-18
transport buffer described above. These should be replaced by a dimension-safe
private transport and an explicit residual-scale contract before claiming a
fully distributable generic backend.

The specialised source contains historical diagnostic comments about a
three-per-thousand GPS/Schur tangent discrepancy. Those comments referred to a
same-state transplant bug in an earlier diagnostic and are invalid as a
qualification claim. The corrected result is that the one-step GPS tangent and
the raw Schur agree when they start from the same physical state; the remaining
Newton issue was the tangent of the host substepping composition.

Possible upstream designs are a Behaviour Brick, an extension of
`StandardElasticity`, or a dedicated structural-plane-stress transformation in
the `Implicit` DSL. The appropriate interface should expose the elastic block
and its Jacobian before gradient subtraction, rather than relying on generated
`fzeros` positions. It should also make the stress-dependence contract and
residual scaling explicit.

## 15. Reproducibility and references

The reports record TFEL/MGIS and Python versions, source hashes, the EBSD source
hash, orientations, parameter set, thread settings, Newton/Krylov diagnostics
and field hashes. The independent condensation route, the specialised GPS
route and the generic route are retained as separate validation artefacts.

Relevant MFront concepts are the `StandardElasticity`/Hooke stress-potential
infrastructure, Behaviour Bricks, the generic behaviour interface and MGIS's
batch material-data interface. The repository's feasibility report records
the TFEL 5.1 ABI probe and the generated-hook experiments:
`validation/structural_plane_stress_mfront_feasibility.md`.

This note does not claim support for arbitrary MFront behaviours, finite
strains, cohesive-zone behaviours, non-standard multi-gradient systems, or a
distributed external Behaviour Brick for TFEL 5.1.
