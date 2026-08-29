# Methodological capabilities inventory

## Scope and reading rule

This inventory is the first LOT A audit for a methodological presentation of
the repository.  It inventories methods that can be reused beyond the current
P43 demonstrator, rather than treating P43 as the product that every page must
complete.

The maturity scale is deliberately orthogonal to an experimental claim:

* **M0** — concept or exploratory proposal;
* **M1** — implemented;
* **M2** — numerically qualified by declared tests;
* **M3** — demonstrated on a synthetic or registered case;
* **M4** — experimentally demonstrated with the relevant data provenance.

An M3 method can therefore be strong and reusable while the P43 material
interpretation remains open.  The current inventory contains no M4 claim:
the physical DIC--EBSD co-registration and a production experimental inverse
workflow are not independently established.

## Capability inventory

| Method | Scientific purpose | Core equations / operator | Implementation | Evidence | Current maturity | Main remaining gap |
|---|---|---|---|---|---|---|
| Full-Dirichlet spectral mechanics | Solve heterogeneous small-strain mechanics from arbitrary measured boundary displacements without imposing periodicity | $u=u^\ast+u^f$, $u^f|_{\partial\Omega}=0$; equilibrium $B^T\sigma=0$ | `src/fem_inhouse/spectral2d`, `spectral_mechanics` workflow, harmonic lifting and DST-I reference operator | `E-SPECTRAL-003`; `docs/explanation/spectral_mechanics/full_dirichlet_formulation.md`; `validation/full_field_operator_gate.md` | **M3** — registered full-Dirichlet cases and full-field operator gate | A generic production DIC runner and broader independent experimental demonstrations are not registered. |
| Matrix-free Newton--GMRES with spectral preconditioning | Retain the actual nonlinear constitutive residual and tangent while avoiding a global stiffness assembly | $Jv=-B^TC_{alg}Bv$; $B_0^{-1}$ applied by DST-I inside GMRES | `solver_pipeline`, `newton_gmres_contract`, TET2 and constitutive adapters | `E-SPECTRAL-003`; `validation/reference_data/ebi_tet/state_sharing_m12.json` and `state_sharing_m24.json` | **M3** — residual/transaction checks pass on registered grids | Broader performance/evidence coverage and a stable user-facing generic runner remain gaps. |
| Full-field plastic operator and FFT adjoint | Map a perturbation field to displacement/strain and return objective duals without one global solve per local coefficient | $A\,\delta\varepsilon_p\mapsto\delta u$ and $A^T$; local coefficient gradients are contractions after one adjoint | `src/fem_inhouse/identification/plastic_observability.py`, `scripts/qualify_full_field_plastic_operator.py` | `validation/full_field_operator_gate.md`: 22,293,208 unknowns, adjoint discrepancy $4.445\times10^{-17}$, $A\approx52$ s, $A^T\approx53$ s | **M3** — full-field adjoint identity and coefficient-scaling milestones are qualified | The explanation is buried in a reconstruction report; a general adjoint/gradient contract and wider objective coverage are not foregrounded. |
| FEMU direct sensitivities and SVD | Determine which constitutive parameter combinations a measured field can constrain before optimization | $r(\theta)=W[O(u(\theta))-y]$; $S_\theta=\partial r/\partial\theta=U\Sigma V^T$ | Synthetic direct-sensitivity drivers, archived prior/final Jacobians, log-parameter SVD reports | `E-SRIX-P43-SYNTH-001`, `E-SRIX-P43-SYNTH-002B`, `E-SRIX-P43-SYNTH-003`, `E-SRIX-PARAM-OBS-001`--`008` | **M3** — synthetic/registered sensitivity and scale-up demonstrations | No production boundary-only driver; absolute experimental detectability and four-parameter P43 identification remain open. |
| DIC observation operator | Compare mechanics to what the measurement chain can actually transmit, rather than to raw strain fields | $y=O_{DIC}(u)+n$; transfer, crop, mask, interpolation and differentiation are explicit | DIC measurement-chain contracts, synthetic transfer tests and `DICSpectralTransfer`/whitening code | `E-DIC-001`, `E-DIC-004`, `E-DIC-006`; `docs/explanation/measurement/dic_observation_limits.md` | **M3** — algorithmic/synthetic transfer is qualified | The P43 noise covariance and temporal correlation are not fully qualified; image-level experimental claims remain bounded. |
| 3-D constitutive law under structural plane stress | Use a genuinely three-dimensional material response in a two-dimensional structural equilibrium solve | Solve $\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0$ for $\varepsilon_{zz},\varepsilon_{xz},\varepsilon_{yz}$; $C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}$ | Nested, external-condensation, structural-MFront and native-coupled closures | `docs/reference/evidence/srix_qualification.md`; `docs/reference/evidence/native_srix_qualification.md`; registered plane-stress reports | **M3** — numerical equivalence/registered cases | Wider constitutive coverage and independent experimental validation of the underlying material model are not established. |
| Forest--Rubin SRIX crystal plasticity | Provide an orientation-aware, rate-independent FCC constitutive candidate with slip interaction and memory | $\Delta\varepsilon^p=\sum_s\Delta\gamma_sM_s$; $r_s=\tau_0+Q\sum_jm_{sj}(1-e^{-bp_j})$; incremental transition controlled by $R$ | MFront behaviour `Fcc316LForestRubinSrix.mfront` and native NumPy/Numba implementation | `E-SRIX-P43-001`; `validation/srix_canonical_qualification_report.md`; source/unit tangent tests | **M3** — formulation and registered robustness/equivalence cases | `R` is an analytical transposition, not an experimental calibration; P43 morphology and registration do not validate the material hypothesis. |
| Méric--Cailletaud crystal-plasticity comparator | Test a rate-dependent FCC formulation and separate physical rate effects from numerical refinement | $\Delta\gamma_s=\Delta t\langle(|\tau_s-X_s|-r_s)/K\rangle^n\operatorname{sign}(\tau_s-X_s)$ | Registered MFront/reference branch and comparison workflow | `E-MERIC-P43-001`; `docs/explanation/constitutive/meric_cailletaud.md` | **M2** — law and implementation are defined, but the registered P43 production path is not qualified | Temporal convergence and a physically calibrated rate scale are still open; the eight-step failure is not a viscosity measurement. |
| MFront/reference versus native SRIX implementation | Separate a trusted constitutive oracle from an explicit, optimisable implementation for coupled closure and future accelerators | Same SRIX map, different implementation path; nested/coupled local solves and condensed tangent | MFront/MGIS bridge plus NumPy, Numba-fused and adaptive native kernels | `docs/reference/evidence/native_srix_qualification.md`; `validation/p0043_*coupled*results.md` | **M3** — registered field/state/tangent equivalence and performance cases | End-to-end reproduction depends on TFEL/MGIS and archived payload availability; GPU remains unclaimed. |
| EBSD orientation and material-frame assignment | Supply heterogeneous crystal frames and Schmid tensors to the constitutive law | $A_m=Q_{global\to material}A_gQ^T$; assignment $F$ is separate from storage layout $C$ | EBSD contracts, orientation readers and rotated cubic/FCC constitutive inputs | `E-EBSD-001`; `docs/reference/scientific/ebsd_orientation_contract.md`; P43 registration report | **M2** — orientation mapping and numerical controls are verified | `registration_proven=false`, global geometry/axis metadata are missing; no M4 co-registration claim is allowed. |
| Observability and latent-state diagnostics | Distinguish an observable fit, field observability and unique constitutive recovery | Free-field $A$ nullspaces; parametric $S_\theta=U\Sigma V^T$; latent error reported separately from displacement error | Tensor/local inverse, FCC decomposition, DIC-weighted field and parametric SVD analyses | `docs/explanation/identification/observable_fit_vs_latent_identifiability.md`; `validation/observability_atlas_v1/report.json`; `E-SRIX-PARAM-OBS-*` | **M3** — synthetic/registered counterexamples and positive constrained twin | The results are not a material identification and should remain a general diagnostic framework rather than a P43 conclusion. |
| Exploratory inverse, reduced and learned surrogates | Test cheaper or learned alternatives and identify when their proxy is not connected to the target observable | REGM equilibrium gap $f=B^TW\sigma$; reduced-basis projections; causal sequence-adjoint/TANN contracts | REGM scripts, reduced-basis campaigns, `tann_fcc_*` sequence/adjoint code, nonlocal branch | `E-SRIX-REGM-001`/`002`; `validation/tann_fcc_recovery_strategy.md`; reduced-basis validation reports | **M1--M3, method-dependent** — implemented diagnostics range from exploratory to registered twin results | A compact cross-method explanation is missing; observed-DIC transfer, trained scientific TANN and production reduced models are not established. |

## P43 demonstrator boundary

P43 is the main registered experimental demonstrator used to exercise these
methods.  It is not the maturity definition for the methods themselves.  The
current P43 chain supports numerical and synthetic demonstrations, but crystal
plasticity has not shown a convincing localisation improvement over the
simpler baseline and the physical DIC--EBSD registration is not independently
proven.  Those limits prevent an M4 material conclusion; they do not erase an
M3 solver, adjoint, observation or sensitivity capability.

## Methods currently under-described

The inventory identifies a small number of real narrative gaps rather than a
need to rewrite every page:

1. **The matrix-free adjoint is not presented as a first-class method.**  Its
   full-field qualification is currently easiest to find in
   `validation/full_field_operator_gate.md` and
   `docs/explanation/spectral_mechanics/plastic_inverse_reuse.md`.
2. **Sensitivity routes are split across FEMU, SVD and historical reports.**
   The distinction between finite differences, direct/tangent sensitivities
   and mechanical/eigenstrain adjoints is technically present but not yet a
   single methodological story.
3. **The relationship between the observation operator and the inverse
   methods is clear locally but not visible as a repository-wide capability.**
   DIC transfer, full-field adjoints and parametric FEMU should be shown as
   composable operators, while P43 covariance details remain secondary.
4. **The reduced/learned alternatives have heterogeneous maturity.**  REGM,
   TANN, reduced bases and nonlocal coupling should be labelled by question,
   evidence and boundary rather than presented as one competing product path.
5. **There is no short landing page for the methodological landscape.**  The
   homepage and Explanation index expose the components, but not yet the
   common architecture or the potential beyond P43.

## Proposed LOT B page set

No changes are made by this audit.  If LOT B is opened, the smallest useful
set is:

* create `docs/explanation/methodological_landscape.md` as the central
  one-page route;
* revise `docs/explanation/identification/femu_identification.md` around
  finite-difference, direct/tangent and adjoint sensitivity costs;
* extend `docs/reference/numerics/femu_sensitivity_and_svd.md` with the
  sensitivity/gradient contracts and their boundaries;
* revise `docs/explanation/spectral_mechanics/plastic_inverse_reuse.md` so
  the qualified $A/A^T$ result is a reusable full-field capability, not only a
  historical reconstruction note;
* add only the links needed from `docs/index.rst` and
  `docs/explanation/index.md` to make the landscape discoverable.

Constitutive, DIC and EBSD pages already contain the necessary scientific
building blocks; LOT B should link and reposition them rather than duplicate
their detailed equations.

## Audit conclusion

The repository contains a mature numerical core, partially mature inverse
methods, and an experimental demonstrator whose data provenance is still being
consolidated.  The main documentation need is therefore a methodological
landscape and a first-class adjoint/sensitivity narrative, not another P43
qualification campaign.
