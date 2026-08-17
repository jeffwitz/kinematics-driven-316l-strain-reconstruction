# Phase-space clustering of the admissible trajectories — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The conditional analysis showed that `(s, p_eq)` does not determine the
inelastic increment. Before any law is fitted, look for **families**: do the
400 000 admissible state–response pairs `(S_n(x), Delta eps_n(x))` organise
spontaneously into regimes, and **which variables make the structure
sharpest and most reproducible**? A regime found this way is the empirical
discovery of internal variables; a clean continuous manifold would say
"continuous vector field" instead — both are outcomes.

## Data

* Trajectories: the predictive projected-Krylov lines at r=8 and r=16
  (`validation/_generated/shared_tensor_generator/krylov_trajectories.r{8,16}.npz`),
  raw and observable-projected fields both stored.
* Orientation: `essais/9_numerical/CP_dataset.h5`, Euler angles and max
  Schmid factor on the full 3600×3100 grid, cropped at the campaign window
  `(1580, 1030)` x 100, broadcast to the two material subcells per cell.
* Population: all points with `|s| > 1e-2 * max(|s|)` (loaded), 20 states.

## Feature sets, in increasing richness

| set | features | question |
|---|---|---|
| F1 | `q`, `p`, `(sin, cos)` of the deviatoric angle | stress alone |
| F2 | F1 + `p_eq` (observable cumulative) | + hardening level |
| F3 | F2 + max Schmid factor | + orientation summary |
| F4 | F3 + `(sin, cos)` of the three Euler angles | + full orientation |

All features standardised. The response is measured separately and is never
a clustering feature in the primary runs: amplitude `Delta p = |Delta eps|_Gp`
and the deviatoric direction angle to `s`.

## Clustering procedure

HDBSCAN on a 40 000-point subsample stratified by state (2 000 per state),
`min_cluster_size = 200`, `min_samples = 50`; the full population is assigned
to the nearest cluster medoid on the standardised features; clusters smaller
than 0.5 % of the assigned population are merged into noise. Labels are
compared with the adjusted mutual information (AMI).

> **Amendment, recorded before any clustering result was interpreted.** The
> registered parameters produced an all-noise labeling on every feature set
> (the smooth path-shaped data have no stability contrast at that scale), so
> the procedure would have been uninformative. They are replaced, pre-reading,
> by `min_cluster_size = 50`, `min_samples = 5` — the exploratory setting for
> unknown regime count and density, which is exactly the question this
> analysis asks. All frozen bars are unchanged.

## Frozen bars

1. **Robustness to the reconstruction.** AMI between the labelings of the
   same feature set from the r=8 and r=16 trajectories `>= 0.5`. Below it,
   the clusters are reconstruction artefacts, whatever their silhouette.
2. **Kernel exclusion.** AMI between the F2 labeling with the raw `p_eq` and
   with the observable-projected `p_eq` `>= 0.5`. Below it, the kernel
   carries the cluster structure.
3. **Time-mixing.** No cluster with more than `80 %` of its mass inside a
   single state — otherwise the clustering merely re-found the loading
   levels.
4. **Response conditioning.** A feature set is declared *promising* if it
   passes 1-3 and at least one of: the within-cluster circular standard
   deviation of the flow-direction angle is at most `1/1.4` of the global
   one, or the within-cluster amplitude variance gives `R^2_cluster >= 0.5`.
   Passing 1-3 but failing both conditioning bars is a named result: the
   variables do not determine the response — the missing-variable search
   continues.

## The sharpest-variable comparison

The silhouette and the DBCV-equivalent validity index are reported per
feature set; the feature addition that raises them the most is the
empirical internal-variable discovery, registered as such.

## Out of scope

No law fitting. No CNN. No claims beyond the P43 window and the admissible
reconstruction it produced. Cluster count is reported, never asserted as a
physical truth — only the frozen bars above speak.
