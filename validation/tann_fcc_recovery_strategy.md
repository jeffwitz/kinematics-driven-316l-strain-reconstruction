# TANN-FCC recovery: corrected status and learning strategy

Date: 2026-08-23  
Status: **implementation repaired; no trained scientific model exists yet**

This document is the cold-start entry point for any agent resuming the
TANN-FCC work.  It supersedes the operational conclusions of
`tann_fcc_preregistration.md`, `tann_fcc_primary_run_results.md` and
`tann_fcc_amended_run_status.md`.  Those files remain historical evidence and
must not be used to justify a constitutive claim.

## 1. What the archived campaign actually established

The archived primary 100 x 100 run used `sigma_ref = 2 mu`.  Its response was
almost elastic and its displacement loss was nearly insensitive to the
network.  The amended `sigma_ref = 200 MPa` campaign made the untrained law
active, but the 100 x 100 trajectory stopped before completing one full
training step.  No qualified trained weights were saved.

The historical EVM figure was additionally invalid: the DIC column was an
absolute strain from state 0 while the elastic and TANN columns were increments
from state 20.  Reconstructing all columns from the incremental displacement
fields gives the following diagnostic correlations:

| state | corr(DIC,TANN) | corr(DIC,elastic) | std(TANN)/std(DIC) |
|---:|---:|---:|---:|
| 25 | 0.504 | 0.537 | 0.72 |
| 32 | 0.221 | 0.248 | 0.50 |
| 40 | 0.711 | 0.735 | 0.64 |

The archived TANN therefore added no demonstrated spatial information beyond
elastic equilibrium and remained too smooth.  This is a diagnostic of the old
pipeline, not a negative result about the TANN concept.

The corrected view is archived as
`validation/figures/tann_fcc_p43/EVM_incremental_diagnostic.png`; its title
explicitly prevents reuse as model evidence.

## 2. Corrected defects

The following defects were present in the checkout at `fb585a7` and are fixed
in the recovery commit that references this document.

1. **Optimizer trajectory reset.** A rollout committed state through the last
   frame and the next Adam step continued from that state.  The adjoint instead
   assumed a zero start.  `TannFCCSequence` now captures its exact initial
   state and strain and restores them before every rollout.  Every record also
   stores its exact previous state, so restarts and adjoints cannot silently
   disagree.
2. **Observation asymmetry.** The measured DIC displacement was filtered a
   second time through `O(u_fem-u_dic)`.  The loss is now
   `O(u_fem)-u_dic`, and the adjoint uses the exact transpose of `O`.
3. **Wrong transfer provenance.** The old training used the declared MEDIUM V4
   transfer although the history was produced with the historical
   `legacy_script_2021` profile.  The differentiable approximation now reads
   the legacy-profile/corrected-warp transfer campaign.
4. **Wrap-free adjoint.** `apply_without_wrap` is `P + F(I-P)`, where `P` is
   the affine projector and `F` the spectral transfer.  Its exact adjoint is
   `P + (I-P)F`, not another forward application.  It is implemented and dot
   tested.
5. **Kelvin/engineering boundary.** The spectral mechanics solver exchanges
   engineering strain and Voigt stress, while the TANN is Kelvin internally.
   `evaluate_in_plane` now converts strain, stress and tangent explicitly; the
   tangent committed for the mechanical adjoint is stored in engineering form,
   and the material VJP applies the corresponding Kelvin cotangent conversions.
6. **Transverse plane-stress closure.** The previous force added a transverse
   compensation equivalent to constraining total `xz/yz` shear while reporting
   zero transverse stress.  The reduced energy now eliminates the unobserved
   transverse total strains: elastic `xz/yz` are zero and the total transverse
   shears follow the plastic slips.  A 3D Hooke re-evaluation and an energy
   gradient test guard the result.
7. **History origin.** State 20 was treated as virgin despite already carrying
   appreciable deformation.  The training driver now replays states 1--40 from
   state 0.  States 1--20 are causal warm-up states.
8. **Interpolated observations.** Repaired states 31 and 32 remain in the
   mechanical path but carry no loss and are not holdout evidence.
9. **Training persistence.** Model and Adam states are saved atomically after
   every optimizer step.  Per-increment trajectory archives remain diagnostic
   only: resuming a partial trajectory for training is forbidden because the
   restart state depends on the network parameters and its omitted sensitivity
   would invalidate the gradient.

## 3. What remains scientifically unidentifiable

P43 supplies boundary and interior displacements but no synchronized force.
Consequently the absolute stress scale, mobility scale and hardening scale
cannot all be learned uniquely from this trajectory.  In particular,
`sigma_ref` changes the physical rate of slip along the strain path; it is not
just a harmless input normalization.

T0 is strictly local (`context_dim=0`) and uses isotropic elasticity.  It can
express orientation-dependent slip through the EBSD Schmid tensors, but it
cannot represent grain-neighbour interactions, the unobserved through-thickness
organization or a spatial internal length.  A smooth T0 response is therefore
possible even when the implementation is correct.

The current full-rank softplus mobility also has no exact elastic domain: every
non-zero force produces some slip.  Its latent quadratic energy has no
independently identified scale.  These choices make a completely free TANN
harder to identify than a conventional FCC law from the available data.

## 4. Adopted learning strategy

The next objective is **not** another long P43 Adam run.  The TANN must first
pass a falsification ladder in which every added freedom answers a measured
failure.

### Gate A — corrected numerical contract

Required and now covered by unit tests:

- engineering/Kelvin shear patch and tangent finite differences;
- condensed free-energy force derivative;
- 3D Hooke plane-stress residual;
- repeated-rollout identity;
- observation/adjoint dot product;
- trajectory-adjoint finite difference with the observation operator.

A 4 x 4 real-history smoke test replayed states 1--24 with 24/24 converged
increments and no cutback. Its machine-readable, deliberately limited verdict
is `validation/tann_fcc_recovery_qualification.json`.

### Gate B — constitutive digital twin

Generate small trajectories with a known FCC reference law (the existing SRIX
or Méric implementation), then pass its displacements through the qualified DIC
observation operator.  Train without exposing stresses or internal variables.
The test asks whether the inverse procedure recovers held-out displacement
morphology and a transferable material response.  It must include at least:

- several orientations and loading paths, including shear;
- a reversal or non-proportional path if latent/kinematic memory is claimed;
- noise and spatial transfer consistent with `legacy_script_2021`;
- an unseen orientation/path holdout;
- comparison of stresses and slips to the known truth **only for scoring**.

Failure here closes free-law discovery from P43; no experimental training is
allowed to compensate for it.

### Gate C — physically anchored TANN

The preferred model is a qualified FCC slip law plus a bounded learned
correction, not an unconstrained mobility learned from zero:

1. keep the FCC systems, plane-stress reduction and a conventional positive
   slip resistance/hardening law explicit;
2. fix elastic and stress-scale parameters from independent information;
3. let the network learn only a dimensionless residual mobility or hardening
   interaction, bounded around the physical baseline;
4. preserve non-negative dissipation and an explicit near-elastic domain;
5. add spatial context only after the local anchored model fails a morphology
   metric that the observation operator can resolve.

This reduces the inverse problem and gives a meaningful zero-network baseline.
SRIX, the anchored TANN and elasticity must be run on exactly the same history.

### Gate D — small experimental crop

Use a band-containing crop small enough for repeated optimization.  Replay the
full state-0 history, score only valid states 21--40, and keep
`{24, 28, 36, 39}` as temporal holdout.  Report separately:

- observed-displacement error;
- EVM amplitude metrics;
- band width, directional spectrum and localization metrics;
- reaction/resultant only if an experimental force becomes available;
- slip activity and dissipation as model diagnostics, never as DIC truth.

Advancement requires a gain over both elasticity and the physical FCC baseline
on held-out morphology, not only on a low-pass displacement norm.

### Gate E — P43 and transfer

Only after Gates B--D pass may a full P43 run be authorized.  A second
band-containing ROI or a distinct loading path is then evaluated without
retraining.  A model that only improves P43 is a reconstruction surrogate, not
a learned constitutive behavior.

## 5. Observation policy

The image-level V3 replay is the scientific observation operator.  The
legacy-profile spectral transfer is retained only as a differentiable surrogate
for rapid training.  Before using its gradient in a real campaign, compare its
predicted loss and gradient ranking against finite differences through V3 on a
small set of perturbations.  If rankings disagree, use a differentiable image
surrogate trained against V3 or a derivative-free outer loop; do not tune the
spectral transfer against FEM-DIC agreement.

## 6. Stop/go rules

- **Stop** if the digital twin cannot distinguish the known FCC law from
  elasticity on held-out paths.
- **Stop T0** if the corrected local model remains smoother than both DIC and
  the physical FCC baseline while spatial frequencies are observable.
- **Open T1** only if T0 passes amplitude/history gates but fails resolved
  morphology in a way correlated with grain neighbourhoods.
- **No material claim** without transfer to another ROI/path and, for stress
  scale, a force measurement or an independently fixed scale.
- **Never reuse** the archived primary/amended artifacts as trained-model
  evidence.  They are regression/provenance records only.

## 7. Cold-start file list

Read in this order:

1. `Claude.md`;
2. this file;
3. `src/fem_inhouse/constitutive/tann_fcc.py`;
4. `src/fem_inhouse/identification/tann_fcc_sequence.py`;
5. `src/fem_inhouse/identification/tann_fcc_adjoint.py`;
6. `src/fem_inhouse/identification/dic_whitening.py`;
7. `scripts/train_tann_fcc_p43.py`;
8. `tests/unit/constitutive/test_tann_fcc.py`;
9. `tests/unit/constitutive/test_tann_fcc_sequence.py`;
10. `tests/unit/constitutive/test_tann_fcc_adjoint.py`;
11. `validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json`;
12. the three superseded historical TANN documents named at the top.

The next implementation ticket is Gate B, not a P43 production run.
