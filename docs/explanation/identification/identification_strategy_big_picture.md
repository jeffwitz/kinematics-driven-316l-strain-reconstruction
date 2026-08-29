# Identification strategy: the scientific big picture

This page is the short overview of the project's current scientific strategy.
It is intended for a reader who knows numerical mechanics but has not followed
the successive reconstruction, constitutive and identification studies.  The
detailed derivations, commands and evidence remain in the linked pages.

## 1. What the experiment provides

The experiment provides images, a loading history, DIC displacement fields,
specimen geometry and EBSD orientations.  DIC is an observation of motion, not
a direct measurement of stress, plastic strain or slip activity.  A useful
abstraction is

$$
y^{\mathrm{obs}} = O(u) + \text{noise},
$$

where $O$ includes registration, spatial transfer, masking and the conventions
of the measurement chain.  Its resolution and noise determine which changes in
the mechanical field can actually be seen.  The temporal path matters as well:
a constitutive law with memory must be driven through the measured sequence,
not only through its final displacement.

EBSD supplies a crystal orientation at a material point.  It determines how
the cubic elasticity and slip systems are rotated into the specimen frame, but
it does not provide the active slips, stresses or constitutive parameters.
These distinctions are developed in [DIC observation limits](../measurement/dic_observation_limits),
[the temporal loading path](../reconstruction/temporal_loading_path) and
[EBSD registration and orientation](../measurement/ebsd_registration_and_orientation).

## 2. Why a rich forward problem is necessary

The current forward chain is:

```text
DIC boundary history
        ↓
full-Dirichlet mechanical forward
        ↓
3-D constitutive update
        ↓
structural plane stress
        ↓
EBSD orientation at each point
        ↓
SRIX, MFront or native SRIX backend
        ↓
predicted interior fields
```

The full-Dirichlet lifting carries the measured boundary data while the
interior remains a prediction of equilibrium and constitutive response.  This
is why prescribing the measured displacement at every node is a useful
negative control, but not an identification experiment.

J2/Ludwik is retained as a historical and numerical baseline: it verifies the
mechanical chain and shows the limits of an isotropic local law.  SRIX is the
current crystal-plasticity production direction because it combines FCC slip,
anisotropic elasticity, rate-independent incremental evolution and a local
consistent Newton formulation.  Méric--Cailletaud provides a rate-dependent
comparison branch.  MFront remains the constitutive oracle used to qualify the
native implementation.  The native SRIX backend exists because coupled
plane-stress closure and point-local kernels make the many-forward budget more
manageable, while preserving a path toward a GPU implementation.  The
matrix-free spectral solver and its Krylov preconditioner serve the same
purpose at the global level: make repeated, full-field forward solves
practical.

See [Forest--Rubin SRIX](../constitutive/forest_rubin_srix),
[structural plane stress](../constitutive/structural_plane_stress), and
[native SRIX optimisation](../native-srix/optimization_strategy) for the
technical detail.

## 3. Three different questions

The central methodological point is that reconstruction and identification
are not the same inverse problem.

```text
A. Field reconstruction
   Which latent field reproduces the measured kinematics?

B. Field observability
   Which perturbations of a latent field can pass through the DIC chain?

C. Parametric identification
   Which combinations of SRIX parameters can the experiment distinguish?
```

The first question can have many answers.  In the archived tensor-inverse
study, the displacement observable was reproduced to essentially machine
precision while the latent tensor field could still be wrong by nearly 100%.
Agreement of a displacement field therefore does not validate a unique hidden
constitutive state.

The second question is studied by the free tensor/eigenstrain observability
operator.  Its leading modes are statistically detectable through the
registered DIC analysis, but their anatomy is edge-dominated and does not
recover the plastic localisation.  Removing the boundary band does not repair
that conclusion.  This is a statement about which field perturbations the
measurement can see; it is not a statement that SRIX parameters have been
identified.

The third question is the actual FEMU problem.  It asks how the constitutive
forward changes when the parameter vector changes.  The synthetic studies show
that this machinery works: a registered P43 synthetic M100 case recovered its
generating parameters after transfer from M20.  The experimental records,
however, do not establish a unique four-parameter calibration.

## 4. Why the SVD is central

For a parameter vector $\theta$, the weighted residual sensitivity is

$$
S_\theta = \frac{\partial}{\partial\theta}
 W\,[O(u(\theta))-y^{\mathrm{obs}}].
$$

When $O$ and $W$ are fixed, the data vector itself drops out of the
derivative: $S_\theta = WO\,\partial u/\partial\theta$.  The experimental
record still matters through the evaluation point, loading path, crop, mask
and observation convention.

Its singular value decomposition,

$$
S_\theta = U\Sigma V^T,
$$

separates observable combinations from weak or nearly null combinations.  The
columns of $V$ are directions in parameter space, not individual parameter
certificates.  In the registered four-parameter SRIX records,
$\tau_0/R$-dominated directions are strong, a $Q+b$ direction is weaker and
the opposite-sign $Q-b$ direction is almost null.  This is a conclusion for
the recorded parameterisation and loading path, not a universal property of
all SRIX experiments.  It means that a sound future FEMU should optimise in
the supported subspace before claiming separate recovery of every parameter.

The archived raw and scalar-whitened controls have almost identical spectra
because the recorded whitening is scalar.  Multiplying a Jacobian by a scalar
changes singular-value magnitudes but not $V$ or normalised singular values;
that agreement is therefore an algebraic consistency check, not an independent
validation of the spatial DIC chain.  The full transfer-plus-spatial-whitener
parametric SVD remains blocked until the qualified repeated-frame noise
payload is available.

Definitions and recorded values are collected in [SRIX parametric
observability](srix_parametric_observability) and its [evidence
reference](../../reference/evidence/srix_parametric_observability).

## 5. Why alternatives were tested and set aside

Several alternatives were scientifically useful precisely because they showed
what an apparently good proxy cannot establish.

* Local tensor inversion and TANN-style closures can fit an observable while
  leaving a large latent nullspace.  The issue is not merely the optimiser;
  the inverse state is not uniquely determined by the kinematics.
* REGM ranks candidates well on an exact mechanical twin, but its ranking is
  destroyed after the registered DIC observation.  A surrogate must be
  validated after the same observation operator, not only in the mechanical
  state space.
* One-point/TET2/EBI variants, reduced integration and other inexpensive
  proxies can improve a numerical symptom while changing the constitutive
  history or failing the scientific gate.  Their negative results are why the
  production strategy keeps the qualified full forward visible.

These results are not discarded side experiments.  They define the boundary
between an observable agreement, a useful screening proxy and an actual
constitutive claim.  The detailed analyses are linked from the [negative
results](../evidence/negative_results) and [REGM explanation](regm_screening).

## 6. The strategy now

```text
                    EXPERIMENT
             DIC + history + EBSD
                       │
                       ▼
              observation model
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
 mechanical forward             field observability
 SRIX + plane stress       transfer + noise + field SVD
        │                             │
        └──────────────┬──────────────┘
                       ▼
                  FEMU residual
                       │
                       ▼
                parametric SVD
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       observable modes      weak/null modes
             │
             ▼
      identify only in the
      supported parameter subspace
```

The field-observability branch is a prerequisite and a diagnostic.  It must
not be substituted for the parametric FEMU branch.  Conversely, a successful
synthetic FEMU fit does not prove that the experimental data contain the same
information.

## 7. What is demonstrated, and what is open

**Demonstrated**

* The SRIX forward machinery and its MFront/native qualification path work for
  the registered cases.
* Synthetic FEMU machinery works, including the registered M20-to-M100
  scale-up.
* The archived parametric sensitivities have a strong low-dimensional
  structure.
* DIC transfer and uncertainty limitations have been quantified.

**Supported but limited**

* Some SRIX parameter combinations are observable in the registered records.
* The $Q/b$ separation is extremely weak in those records.
* REGM is useful as an exact-twin screening result, not after the current DIC
  transfer.

**Negative**

* Kinematics alone do not uniquely recover a latent constitutive field.
* REGM transfer to the observed DIC data fails its registered ranking gate.
* Experimental four-parameter SRIX identification is not qualified.

**Open**

* Reconstruct the full DIC-weighted parametric SVD once the qualified spatial
  whitener is available.
* Establish a genuinely boundary-only experimental FEMU workflow.
* Assess experimental SRIX calibration only in the supported parameter
  subspace.
* Transfer these conclusions to larger or independent observations.

The intended order is therefore deliberate: qualify the forward, quantify the
observation, determine observable parameter modes, and only then spend the
cost of a full experimental identification campaign.  This keeps the next
scientific step falsifiable and prevents a good-looking displacement fit from
being mistaken for a unique constitutive identification.

