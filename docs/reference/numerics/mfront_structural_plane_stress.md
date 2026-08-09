# Structural plane stress from generic three-dimensional MFront behaviours

**Category: scientific reference.**

This document describes a structural plane-stress closure for a genuinely
three-dimensional constitutive behaviour. It is intended for small-strain,
implicit MFront behaviours using the standard elastic-strain split provided by
`StandardElasticity`. The constitutive state remains three-dimensional; plane
stress is a local closure, not a reduction of the finite-element kinematics.

## 1. Plane stress is a traction-free condition

Let the structural surface normal be \(\mathbf n=\mathbf e_z\). A free surface
has zero traction:

$$
\mathbf t=\boldsymbol\sigma\mathbf n=\mathbf 0.
$$

For a symmetric stress tensor this is the three-component condition

$$
\boxed{\sigma_{zz}=0,\qquad \sigma_{xz}=0,\qquad \sigma_{yz}=0.}
$$

The familiar one-equation formulation is a special case. For isotropic
elasticity, or for an aligned orthotropy, transverse shear is decoupled and
\(\gamma_{xz}=\gamma_{yz}=0\) follows. A crystal orientation measured by EBSD
is generally not aligned with the structural frame, so that simplification is
not valid.

The structural model still has only

$$
u_x(x,y),\qquad u_y(x,y)
$$

as finite-element unknowns. The three transverse strains

$$
\varepsilon_b=(\varepsilon_{zz},\gamma_{xz},\gamma_{yz})
$$

are local constitutive unknowns at each material point; they are condensed and
are not additional FEM degrees of freedom.

## 2. Why crystal plasticity needs the full three-dimensional state

Crystal plasticity evolves

$$
\Delta\boldsymbol\varepsilon^p
=\sum_s \Delta\gamma_s\,\mathbf M_s.
$$

After an arbitrary three-dimensional orientation, each Schmid tensor may have
`xx`, `yy`, `zz`, `xy`, `xz` and `yz` components in the structural frame.
Consequently, both the elastic anisotropy and the plastic strain can generate
transverse shear traction. Setting \(\sigma_{zz}=0\) while prescribing
\(\gamma_{xz}=\gamma_{yz}=0\) does not impose a free surface.

Partition engineering components in the structural frame as

$$
\varepsilon_a=(\varepsilon_{xx},\varepsilon_{yy},\gamma_{xy}),\qquad
\varepsilon_b=(\varepsilon_{zz},\gamma_{xz},\gamma_{yz}),
$$

and similarly \(\sigma=(\sigma_a,\sigma_b)\). In a linearized state,

$$
\begin{bmatrix}\sigma_a\\\sigma_b\end{bmatrix}
=
\begin{bmatrix}C_{aa}&C_{ab}\\C_{ba}&C_{bb}\end{bmatrix}
\begin{bmatrix}\varepsilon_a\\\varepsilon_b\end{bmatrix}.
$$

For a rotated anisotropic material, \(C_{ba}\ne0\) in general. The three
transverse strains must therefore be solved so that \(\sigma_b=0\).

## 3. Structural and material frames

The implementation uses the convention

$$
Q:\text{ global/structural}\rightarrow\text{material},\qquad
T=Q^T:\text{ material}\rightarrow\text{global}.
$$

The finite-element gradient is supplied in the structural frame. The raw
three-dimensional bridge rotates it to the material frame before calling MGIS
and rotates stresses and tangents back. The GPS behaviours instead receive the
global gradient and own the crystal rotation through per-point `Qij`
properties. These two paths must not both rotate the same gradient.

All six-component operations use the repository's engineering/Kelvin
conventions. The structural closure is always expressed in the structural
frame, regardless of the material's crystal axes.

## 4. Three-dimensional reference: external condensation

The backend
`mfront-3d-condensed-plane-stress` keeps the law unchanged. It solves

$$
\sigma_b(\varepsilon_a,\varepsilon_b)=0
$$

with an outer local Newton method. Every trial is restored from the same
committed state, and the converged three-dimensional tangent is partitioned
and condensed:

$$
\boxed{
C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.}
$$

The inverse is never formed explicitly: linear systems with \(C_{bb}\) are
solved. This route is the independent numerical reference because it accepts
any compatible three-dimensional behaviour without modifying its MFront
source. Its cost is the repeated MGIS integration and the host-side closure.

## 5. Specialised monolithic GPS behaviour

The historical SRIX route is
`mfront-native-generalised-plane-stress`, backed by
`Fcc316LForestRubinSrixGps`. The local MFront system retains the three-
dimensional SRIX state and replaces the appropriate elastic residual rows by
the in-plane kinematics and the three traction-free equations.

This formulation is algebraically equivalent to external condensation for one
smooth constitutive step. It is specialised: a new law requires a corresponding
GPS variant and its own qualification. Its robust integration policy is partly
host-side, as described below.

## 6. Generic `StructuralPlaneStress3D`

The new backend is
`mfront-structural-plane-stress`. It applies the same closure transformation
to a `Tridimensional`, small-strain, `Implicit` behaviour compatible with
`StandardElasticity`, without referring to a law-specific variable such as a
slip, a hardening variable, or a Schmid tensor.

The key contract is the standard elastic block. If \(x\) denotes all local
implicit unknowns, the elastic residual has the form

$$
f_e(x)=K_m(x)-\Delta\varepsilon_m,
\qquad K_m(x)=f_e(x)+\Delta\varepsilon_m.
$$

Thus the complete constitutive kinematics can be recovered from the first six
elastic residual rows without knowing the inelastic variables. The generic
transformation computes

$$
K_g=T K_m,
$$

then replaces only the elastic rows by

$$
F_a=(K_g)_a-\Delta\varepsilon^{global}_a,
\qquad
F_b=\frac{(T\sigma_m)_b}{G_{ref}}.
$$

All remaining constitutive rows are left unchanged. For every local Jacobian
column, the elastic rows are transformed with the same material-to-structural
operator; no knowledge of the number or meaning of the inelastic unknowns is
required. The generated source currently provides this mechanism through
MFront fragments and a generated behaviour, because the installed TFEL 5.1
ABI does not export the symbols needed to distribute a new external
`BehaviourBrick` implementation.

The demonstrated V1 domain is therefore deliberately limited to:

- small strain and symmetric stress;
- `@DSL Implicit` and `@ModellingHypothesis Tridimensional`;
- the `StandardElasticity` elastic split;
- a six-component `deel`/`feel` elastic block;
- constitutive laws whose imposed gradient enters through that standard block.

This is not a claim about arbitrary MFront behaviours, finite strains, cohesive
laws, or multi-gradient formulations.

## 7. Consistent one-step tangent

At convergence, let the transformed local system be

$$
F(x,\varepsilon_a)=0,
\qquad A=\frac{\partial F}{\partial x}.
$$

Let \(P\) contain the dependence on the three imposed structural components;
its active entries are the columns corresponding to `xx`, `yy` and `xy`.
The local sensitivity is obtained from

$$
AX=P,
\qquad X=\frac{\partial x}{\partial\varepsilon_a}.
$$

The implementation solves this system and never forms \(A^{-1}\). Since the
stress is obtained from the elastic state,

$$
\boxed{C^{PS}=T\,D_{\sigma e}\,X_e,}
$$

where \(X_e\) denotes the elastic-state rows. This is the tangent of the
one-step structural constitutive map. In the raw three-dimensional route,
eliminating \(\varepsilon_b\) first gives the Schur complement above; the two
expressions are the same implicit derivative written after different
eliminations.

The tangent is a constitutive contract. It is distinct from the tangent of a
host integration algorithm that may later compose several sub-steps.

## 8. Substepping and the composite-map tangent

The layers are:

```text
3D constitutive physics
        ↓
StructuralPlaneStress3D or specialised GPS
        ↓
one-step constitutive map and tangent
        ↓
optional host substepping
        ↓
optional composite-map tangent
        ↓
global FEM Newton
```

If a host replaces one increment by

$$
\Phi=\Phi_n\circ\cdots\circ\Phi_1,
$$

the global Newton needs

$$
D\Phi=D\Phi_n\,D\Phi_{n-1}\cdots D\Phi_1,
$$

not merely the tangent of the last sub-step. The qualified GPS adapter detects
the few points that require substepping and reconstructs this composite
tangent by central finite differences when
`gps_composite_fd_tangent=true`. The absolute engineering-strain perturbation
is controlled by `gps_composite_fd_step` (default `1e-6`). The diagnostic shadow
tangent is not a production mechanism.

This distinction explains the M100 change from 85 Newton iterations for the
unrepaired GPS tangent to 58 with the composite tangent. The correction belongs
to the host composition layer; it is not a modification of the constitutive
law's one-step tangent.

## 9. Implemented strategies and selection

| Strategy | MFront law | Closure | Generality |
|---|---|---|---|
| Native `PlaneStress` | 2D behaviour | MFront standard hypothesis | laws designed for that hypothesis |
| External 3D condensation | unchanged 3D law | host Python | any compatible 3D law |
| Specialised structural GPS | SRIX GPS variant | law-specific MFront system | one qualified law per variant |
| Generic `StructuralPlaneStress3D` | generated 3D variant | generic residual/Jacobian transformation | V1 `StandardElasticity`-compatible implicit laws |

For the qualified SRIX + EBSD workflow, use the specialised GPS route with
composite FD or the generic structural backend. Use external condensation as
the independent reference and for a new three-dimensional law.

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6
    crystal_orientation:
      mode: ebsd
```

The independent reference is:

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
```

Other important options control the transverse predictor, closure tolerances,
conditioning checks for \(C_{bb}\), local iteration limits, the EBSD orientation
provider, and the MGIS thread pool. They exist because the closure is a local
nonlinear solve: the predictor controls its initial distance to the root,
conditioning checks prevent unstable transverse corrections, and explicit
transactional snapshots make rejected trials reproducible. These are numerical
controls, not changes to SRIX parameters.

## 10. Qualification hierarchy

The evidence was built in the following order:

$$
\text{rotated anisotropic elasticity}
\rightarrow J2
\rightarrow \text{SRIX}
\rightarrow \text{Méric--Cailletaud}
\rightarrow \text{P43 M20}
\rightarrow \text{P43 M100}.
$$

The point-material probes verify transverse traction, in-plane kinematics and
the one-step tangent. SRIX and Méric use the same generic transformation; no
law-specific closure equation appears in `StructuralPlaneStress3D`.

On the single-run P43 M100 EBSD campaign (8 increments, crop
`[1570:1670] × [1035:1135]`, four MFront threads, one BLAS and one FFTW
thread), the measured results were:

| Strategy | Newton | elapsed |
|---|---:|---:|
| 3D + Python condensation | 57 | 56.72 s |
| specialised GPS + composite FD | 58 | 51.65 s |
| generic `StructuralPlaneStress3D` + composite FD | 58 | 54.56 s |

The generic and specialised GPS fields agree at approximately
`1e-12` relative L2 for stresses, reactions and crystal slips, and at
`1e-16` for displacement in this run. Both used 192 substepped points and the
same Newton sequence `[6, 6, 7, 7, 7, 8, 8, 9]`. The generic route is about
5.7% slower than the specialised route in this single campaign, but remains
faster than external condensation. These are reproducibility results, not a
universal speed guarantee.

The complete artefact is
`validation/_generated/performance/p43_m100_backend_comparison_latest.json`.
The condensation, specialised GPS and generic GPS reports and field files are
stored alongside it.

## 11. Current implementation limits and upstream question

The generic mechanism is currently implemented as generated/local MFront
fragments. The external Behaviour Brick route was investigated, but the
installed TFEL 5.1 ABI does not export the required `BehaviourBrickBase`
symbols. No TFEL fork is used.

The result is a demonstrated V1 capability, not yet an upstream MFront feature.
The natural upstream question is:

> Can this transformation be represented as a first-class MFront mechanism for
> `Implicit`, small-strain, `StandardElasticity`-compatible, tridimensional
> behaviours?

Possible designs include a Behaviour Brick, an extension of
`StandardElasticity`, or a dedicated structural-plane-stress transformation in
the `Implicit` DSL. Whichever design is chosen, it should preserve the
separation between the constitutive one-step map and any host-side substepping
policy.

## 12. Reproducibility and scope

The M100 reports record the source hashes, EBSD source hash, orientations,
parameter set, thread settings, Newton/Krylov diagnostics and field hashes.
The external condensation route remains the independent oracle; the specialised
GPS route remains a second implementation; the generic route is the proposed
reusable mechanism.

This reference does not claim that all MFront behaviours support the closure.
It does not cover finite strains, cohesive-zone behaviours, non-standard
multi-gradient systems, Méric-specific GPS source code, or a distributed
external Behaviour Brick for TFEL 5.1.
