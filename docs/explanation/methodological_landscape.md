# Methodological landscape

This repository is a research platform for full-field computational mechanics
and inverse methods.  Its numerical methods have different maturity levels and
must not be confused with the maturity of the current experimental
demonstrator.  P43 is the main registered case used to exercise the stack; it
is not the definition of the scientific contribution.

The guiding question is:

> Which mechanical and inverse-method capabilities can turn measured fields
> into defensible information about heterogeneous constitutive behaviour?

## One composable architecture

```text
EXPERIMENTAL INFORMATION
DIC boundary/interior fields, EBSD orientations, loading history
                              │
                              ▼
DATA AND OBSERVATION CONTRACTS
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       MECHANICAL FORWARD              OBSERVATION OPERATOR
       full-Dirichlet spectral          crop, transfer, mask, noise
       matrix-free equilibrium
       3-D constitutive law
       structural plane stress
              │                               │
              └───────────────┬───────────────┘
                              ▼
                     PREDICTED OBSERVABLE
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       DIRECT / TANGENT                    ADJOINT
       SENSITIVITIES                       GRADIENT
              │                               │
              └───────────────┬───────────────┘
                              ▼
                       INVERSE METHODS
                 FEMU, SVD, reduced and learned
                         surrogates
```

These are composable capabilities, not one mandatory end-to-end workflow.  A
spectral solve can be used without FEMU; the mechanical adjoint can support a
field optimisation; plane stress can wrap another three-dimensional law; and
the observation operator can be applied to other full-field datasets.

## I. Mechanics at field scale

### Full-Dirichlet spectral mechanics

Measured boundary displacement is lifted into the domain and decomposed as

```{math}
u=u^\ast+u^f,
\qquad u^f|_{\partial\Omega}=0.
```

The fluctuation has homogeneous boundary values, so a DST-I reference operator
can be used inside a Krylov method without pretending that the physical
problem is periodic.  The actual residual still comes from local stresses and
the actual matrix-free Jacobian from local algorithmic tangents:

```{math}
Jv=-B^T C_{\rm alg}Bv.
```

No global stiffness matrix is assembled.  The DST/FFT action is the inverse of
a simple elastic reference operator used as a preconditioner; it does not
diagonalise the nonlinear heterogeneous constitutive problem.  The formulation
and Newton--GMRES pipeline are described in
{doc}`spectral_mechanics/index`.

### Three-dimensional constitutive behaviour in a two-dimensional solve

Structural plane stress keeps the constitutive law three-dimensional and
relaxes local transverse strains until

```{math}
\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0.
```

Crystal slip, internal variables and anisotropic couplings remain three-
dimensional; only the local closure is condensed before the global solver sees
the in-plane response.  This makes the same structural machinery reusable for
crystal plasticity, MFront behaviours and future three-dimensional laws.  See
{doc}`constitutive/structural_plane_stress`.

## II. Differentiation and inverse mechanics

The generic inverse problem is

```{math}
R(u,\theta)=0,
\qquad y_{\rm pred}=O(u),
\qquad J(\theta)=\tfrac12\|O(u(\theta))-y_{\rm obs}\|^2.
```

The observation operator is part of the problem, not plotting after the fact:

```{math}
y=O_{\rm DIC}(u)+n.
```

For a parameter sensitivity, the repository uses three conceptually distinct
routes:

| Route | Main global cost | Repository status |
|---|---|---|
| Central finite differences | Approximately $2p$ nonlinear forwards for $p$ parameters | Implemented and used as an oracle; fixed-path SRIX gates remain mixed/blocked. |
| Direct/tangent sensitivity | One converged trajectory plus tangent solves/right-hand sides | Implemented in synthetic and shadow SRIX paths; the exact common-path qualification gate is not passed. |
| Adjoint | Approximately one transpose solve per scalar objective, plus local contractions | Full-field linear/eigenstrain $A^T$ is strongly qualified; a generic production SRIX parameter adjoint is not claimed. |

Finite differences are general and easy to audit, but expensive and sensitive
to step size and failed forwards.  Direct sensitivities reuse the converged
state and tangent, avoiding a complete nonlinear replay for every perturbation.
Adjoints reverse the dependency of a scalar objective:

```{math}
R_u^T\lambda=J_u^T,
\qquad
\frac{dJ}{d\theta}=J_\theta-\lambda^TR_\theta.
```

An adjoint is not free: history-dependent constitutive states and multiple
increments still require correct storage and backward propagation.  Its
advantage is that the number of global transpose solves is driven mainly by
the number of scalar objectives rather than by the number of local
coefficients.

### The qualified full-field operator

For an eigenstrain-like perturbation, the linear operator is

```{math}
A:\delta\varepsilon_p\mapsto\delta y,
\qquad
A^T:g_y\mapsto g_{\varepsilon_p}.
```

The scalable pattern is to assemble one complete perturbation field, apply
$A$ once, apply $A^T$ once to the resulting dual, and then form all local
coefficient contractions.  It is not one global solve per coefficient.  The
registered full-field gate covers 22,293,208 interior unknowns, with adjoint
dot-product discrepancy about $4.4\times10^{-17}$, $A\approx52$ s,
$A^T\approx53$ s and peak memory about 1.8 GB.  This is a numerical
feasibility and transpose-consistency result, not a P43 material validation;
details are in {doc}`spectral_mechanics/plastic_inverse_reuse` and
`validation/full_field_operator_gate.md`.

The same matrix-free mechanics plus transpose actions provide ingredients for
high-dimensional field inversion, efficient gradient-based FEMU and future
material or topology optimisation.  They do not mean that those future
applications are already implemented.

## III. Identifiability is a separate question

Once a sensitivity matrix exists,

```{math}
S=\frac{\partial r}{\partial\theta}=U\Sigma V^T
```

$U$ describes observable field patterns, $V$ describes parameter combinations,
and $\Sigma$ describes their relative sensitivity.  Four material parameters
do not imply four identifiable directions.  The free-tensor, compact local,
enriched-basis and FCC experiments show why an excellent observable fit can
coexist with a wrong or non-unique latent state.  SVD is therefore an
interpretation and reduction tool, not a fourth sensitivity-generation route.

The current registered SRIX records show strongly unequal parameter directions
and a weak $Q-b$ combination.  This illustrates the method on one experiment;
it is not an experimental calibration claim.  The specialised discussion is
in {doc}`identification/srix_parametric_observability` and
{doc}`identification/observable_fit_vs_latent_identifiability`.

## Exploratory and complementary branches

Several other branches were experiments on the inverse architecture, not
failed competitors to FEMU.  They address different questions and expose
conditions under which a method would be worth revisiting:

| Method | Scientific question | Current status |
|---|---|---|
| FEMU | Which constitutive parameters best explain the observations? | Synthetic/registered demonstrations; experimental use remains open |
| SVD | Which parameter combinations can the observations distinguish? | Geometric and synthetic/registered demonstrations |
| REGM | Can an equilibrium proxy accelerate or screen inversion? | Exploratory; registered negative transfer to the current DIC observable |
| DIC-driven dissipative reconstruction | What plastic correction is required by measured kinematics, and which constitutive directions are missing from a candidate law? | Reduced/full-field reconstruction, dissipative projection and registered-case tangent/curvature diagnostics demonstrated; constitutive enrichment open |
| Reduced basis | How can a high-dimensional latent field be represented? | Synthetic demonstrations; basis richness and nullspaces documented |
| TANN | Can a causal constitutive evolution law be learned from data? | Pipeline and sequential adjoint developed; no qualified learned law |
| Nonlocal coupling | Is spatial interaction missing from a local material model? | Spatial redistribution demonstrated; physical length scale uncalibrated |

### What these branches taught us

* **REGM** replaces repeated constitutive solves by an equilibrium-gap proxy.
  It can be useful for screening or initialisation, but the recorded proxy
  ranking did not transfer convincingly through the present DIC observation.
  It should be revisited only when that connection is demonstrated for a
  curated observable.
* **Reduced representations** can regularise an otherwise invisible latent
  field: a compact synthetic representation recovered its twin, whereas an
  enriched basis introduced algebraic null directions.  Adding basis
  richness is therefore not automatically adding information.  Revisit when
  bases are designed or orthogonalised against the observable nullspace.
* **DIC-driven dissipative reconstruction** uses $A^Tr$ to build modes guided by
  the observed mechanical defect, then enforces positive dissipation without
  choosing a full constitutive law first.  The observable increment is exactly
  in the FCC tensor span, but the dissipative constraint still leaves a large
  zero-work boundary; this is a route to constitutive hypotheses, not a
  recovery of true slip histories.  Its newer role is to compare the
  reconstructed correction with a candidate constitutive tangent and identify
  observable directions that the current law does not generate.  Revisit it on
  curated multi-case data to screen structured descriptors such as grain size,
  boundary proximity, misorientation and slip-transfer compatibility.
* **TANN** established a causal state-propagation and sequential-adjoint
  pipeline, but the registered learning configuration did not establish a
  useful scientific constitutive law.  The old runs are provenance and
  diagnostics, not a general failure claim.  Revisit with physically grounded
  variables and curated multi-case data that justify the model flexibility.
* **Nonlocal or micromorphic coupling** provides a way to represent a missing
  interaction length and has produced substantial spatial redistribution in
  the recorded campaigns.  Its length scale is not physically calibrated,
  so this remains a secondary modelling branch.  Revisit when a curated
  localisation dataset demonstrates that a qualified local law is
  insufficient.

These branches can be summarised by their triggers rather than by their
campaign history:

| Branch | Revisit trigger |
|---|---|
| REGM | A demonstrated link between the equilibrium proxy and the measured observable |
| DIC-driven dissipative reconstruction | Curated multi-case full-field data with independently supported registration, allowing transverse reconstructed directions to be compared across experiments and used to screen physically structured constitutive enrichments |
| Richer reduced bases | Observable-aware basis design or orthogonalisation |
| TANN | Physically grounded parameterisation and curated multi-case data |
| Nonlocal coupling | Curated localisation data showing a qualified local model is inadequate |

The common lesson is methodological: proxy agreement may not survive the
measurement operator, richer latent spaces can create nullspaces, history
requires causal state propagation, and spatial coupling can change
localisation.  Further discrimination between these options now depends more
on curated DIC/EBSD/loading cases than on another layer of optimisation of
the present P43 record.

## IV. Constitutive and measurement inputs

J2/Ludwik is a robust isotropic baseline for checking boundary conditions,
equilibrium and constitutive plumbing.  Méric--Cailletaud provides a
rate-dependent FCC comparison branch.  SRIX provides the current
rate-independent FCC candidate with twelve slip systems, hardening memory and
EBSD-dependent orientation.  MFront is the qualified reference implementation;
native SRIX exposes the same registered formulation for explicit coupled
plane-stress algebra and acceleration.  Their formulations and boundaries are
independent of whether P43 ultimately supports a material conclusion.

EBSD supplies a local crystallographic frame and rotated slip systems.  Its
scientific value depends on reliable spatial registration to the mechanical
measurement.  The numerical assignment exists for registered-case studies,
but the P43 physical co-registration remains unproven.  DIC supplies measured
image motion through an observation operator; it does not directly measure
stress, slip or plastic strain.  The detailed contracts are in
{doc}`measurement/dic_observation_limits` and
{doc}`measurement/ebsd_registration_and_orientation`.

## Research maturity

| Capability | Current status |
|---|---|
| Spectral mechanics and matrix-free equilibrium | Demonstrated on registered numerical cases |
| Full-field mechanical/eigenstrain adjoint | Demonstrated and full-field qualified |
| Structural plane stress | Demonstrated numerically and in registered closures |
| FEMU, direct sensitivities and SVD | Synthetic/registered demonstrations; experimental workflow open |
| Constitutive CP implementations | Numerically and registered-case demonstrated |
| DIC observation transfer | Synthetic/algorithmic demonstration; P43 absolute noise model limited |
| EBSD-driven experimental CP conclusion | Open because physical registration is not proven |
| Experimental material identification | Open |

“Open” means that the evidence for that scientific claim is insufficient; it
does not mean that the numerical method failed.  Conversely, code or a
successful numerical qualification does not by itself establish an
experimental material hypothesis.

## What remains to consolidate

* **Experimental/data:** DIC--EBSD co-registration, acquisition metadata,
  loading synchronisation and multiple curated cases; the readiness levels
  are defined in {doc}`../reference/data/experimental_dataset_readiness`.
* **Method:** a generic user-facing full-field runner, a broader constitutive
  adjoint route and robust multi-experiment inverse orchestration.
* **Science:** determine which physical ingredients control localisation, then
  identify only parameters supported by curated observations.

## Potential beyond the demonstrator

The spectral solver and adjoint provide ingredients for large-field inverse
mechanics, local field optimisation and topology/material optimisation.  The
3-D plane-stress closure can support other anisotropic constitutive laws in
2-D experiments.  The observation contract can be reused with better-curated
full-field measurements.  Sensitivity/SVD methods could support experiment
design, parameter reduction and multi-experiment identification, while the
native constitutive backend could support CPU/GPU acceleration.  These are
capabilities the architecture enables, not completed claims about P43.

P43 is therefore best read as an imperfect experimental demonstrator that
exercises a set of more general methods: mature numerical mechanics,
partially mature inverse machinery and data curation still in progress.
