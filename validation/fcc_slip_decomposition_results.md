# FCC slip decomposition — results

Against `validation/fcc_slip_decomposition_preregistration.md`, thresholds
frozen before the runs. The ceiling is exact; the dissipation constraint
rejects half; the slip phase space shows no law structure.

## Verdict against the frozen bars

| bar | registered | measured | |
|---|---|---|---|
| representability, constrained | `e_FCC <= 0.5` median | **0.546** (q25 0.305, q75 0.766) | fail by 0.046 |
| represented share | `rho >= 0.5` | **0.836** | pass |
| variant stability | correlation `>= 0.9` | **1.0000** | pass |
| slip-law structure | per-system LOSO `R^2 >= 0.5` | **-0.03** weighted | fail |

## Three measured facts

1. **The observable increment is exactly in the FCC span.** The
   unconstrained decomposition represents the observable-projected
   increment at machine precision (`e_FCC = 2e-12`, `rho = 1.000`):
   nothing non-FCC is needed for the observed kinematics. The raw field is
   *not* exactly representable (`e_FCC = 0.353`): the component the
   displacement kernel carries — the reconstruction closure — is precisely
   the part FCC slip cannot express. The kernel and the slip span are
   nearly complementary in these data.
2. **Per-system dissipation rejects half the increment.** Under
   `tau^alpha Delta gamma^alpha >= 0` on every system, the median residual
   jumps to **0.55** and the represented share falls to 0.84: the
   tangential mass measured as `f_0 ~ 0.47` reappears in slip coordinates
   as tensorially *compensating* slip combinations — systems with positive
   work whose tensors cancel, which the per-system sign constraint
   forbids. This is the registered interpretation of the tangential
   mystery, now quantified: the boundary plasticity is unrepresentable by
   dissipative slips, not by slip kinematics.
3. **The two decompositions coincide exactly** (correlation `1.0000`): the
   sign cone determines the decomposition uniquely — the L1/L2 choice the
   preregistration asked to compare is not a degree of freedom.
4. **No slip law structure.** The per-system conditioning
   `(tau^alpha, Gamma^alpha) -> Delta gamma^alpha` in leave-one-state-out
   kNN is at or below zero for every system (`R^2 in [-0.25, +0.12]`,
   weighted `-0.03`): the resolved shear and its accumulated history do
   not predict the slip activity — the effective field's direction isotropy
   survives the change to slip coordinates, as it must.

## What this says

The FCC experiment answers the user's question with precision: the
effective field is kinematically FCC (ceiling exact), the thermodynamic
per-system constraint rejects ~half of it (the closure/tangential mass),
and the slip phase space contains no local law under the tested state —
the same conclusion as the tensor phase space, now stated in the
coordinates a crystal plasticity law would use. The slip activities
recovered here remain **2-D-projection-compatible activities**, not
claimed true slips, per the registered caveat.

The next discriminator stays the one the analyses point to: separating the
dissipative-representable part from the closure before any law fit —
equivalently, decomposing the increment into its FCC-dissipative component
and the rest, which this experiment now computes at every point.

## Slip-coordinate geometry (the figures)

The slip fields were saved and plotted as the counterpart of the
tensor-phase figures (`fcc_geometry_*.png` beside the phase-space
artifacts): per-system and pooled hexbins of `tau^alpha` vs
`Delta gamma^alpha` and `Gamma^alpha` vs `Delta gamma^alpha`, plus the
pooled view colored by state.

Measured, not eyeballed:

* **The driving force organises the activity — the strongest relation in
  any coordinate system so far.** `Spearman(tau^alpha, Delta gamma^alpha)`
  is `+0.70..+0.83` per system, `+0.76` pooled, against `0.35` for the
  tensor-phase `sigma_eq -> Delta p`. Absorbing the orientation into the
  resolved shears works exactly as proposed: the material's response is far
  better organised in slip coordinates than in Euler angles.
* **But it remains a thick band, not a law.** The within-`tau`-bin
  coefficient of variation of the activity stays `1.8-3.4`, and the kNN
  LOSO score is `~0`: the `tau -> gamma` relation drifts across increments
  (the missing hardening variable again), so the structure is
  in-sample real and cross-increment weak.
* **The accumulated history does not organise the activity**
  (`Spearman(Gamma^alpha, Delta gamma^alpha) ~ 0` pooled): the scalar
  history is not the missing variable.
* **The dissipation splits 50/50 between positively and negatively
  resolved systems.** Half the dissipative activity sits on systems whose
  `tau^alpha < 0` — the compensating-systems picture of the tangential
  mass, confirmed from the driving-force side.
