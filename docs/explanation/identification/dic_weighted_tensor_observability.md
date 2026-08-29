# DIC-weighted observability of inelastic fields

**Mode:** explanation  
**Domain:** identification

This analysis concerns **field observability**, not parametric FEMU
observability.  The script
`scripts/project_dic_residuals_on_observable_modes.py` uses
`TensorPlasticObservabilityOperator` to study a free tensorial
eigenstrain/plastic-strain-like field.  In the registered construction the
operator has the form

$$
A = W_D\,M_D\,K^{-1}B^T C H^{-1/2}.
$$

Its singular vectors describe tensor fields that can produce displacement
patterns visible through the registered DIC observation, transfer model and
noise whitener.  They are not the right singular vectors of the parametric
FEMU sensitivity

$$
S_\theta = \frac{\partial r}{\partial\theta}.
$$

The distinction is essential:

1. **Field observability:** which free inelastic/eigenstrain fields can be
   detected by the DIC chain?
2. **Parameter observability:** which combinations of SRIX parameters change
   the observed residual through $S_\theta$?

The first question is addressed by the archived post-processing below.  The
second remains a subsequent FEMU/SVD task.

## Registered M20 and M100 analysis

The post-processing applies the measured DIC transfer function and repeated-
frame uncertainty to 40 archived P43 states, then projects the residuals onto
the left singular vectors of $A$.  The registered M20 crop retains 12 modes;
its expected pure-noise norm is 26.8701 and its early maximum coefficient is
0.4016 noise standard deviations.  The registered M100 crop retains 20 modes;
its expected norm is 140.0071 and its final first and third coefficients are
-118.0552 and +72.3165 noise standard deviations.  The correct conclusion is
not a general scaling law: in these registered analyses M20 remains below the
measured noise scale early, whereas M100 contains strongly excited observable
residual modes.

The report also fits an **empirical rank-three subspace to states 3--20**.
That subspace explains 99.70% of the early-state variance.  It is a model of
the early heterogeneity/measurement pattern used for a later subtraction; it
is not the first three singular vectors of $A$.

## Mode anatomy and boundary masking

The leading modes are dominated by the crop boundary.  With a 15-pixel border,
the interior covers 49% of the area, while the interior energy share ranges
from 0.094 at worst to 0.197 at the median; the modes are also shear-dominated
(mean component shares 0.24 / 0.22 / 0.54 for $e_{11}/e_{22}/g_{12}$).  Their
correlation with the archived DIC equivalent-strain map is only +0.149 and
their top-decile overlap is 0.134.

Masking a 15-node boundary band does not rescue localisation: the correlation
changes from approximately +0.149 to -0.150 and the top-decile overlap falls
to about 0.112.  The leading modes therefore represent mathematically
observable boundary-leveraged directions, not a reconstruction of the
plastic localisation band.

## EBSD elastic-reference test

The archived EBSD test replaces homogeneous isotropic elasticity by rotated
cubic FCC elasticity at each material point, followed by exact plane-stress
condensation.  The controls pass: isotropic rotation invariance is $4\times
10^{-16}$ relative, cubic symmetry is $7\times10^{-17}$, and the shuffled
EBSD control is indistinguishable from the registered arrangement in the
residual norms.

On the polycrystalline crop, the residual norms at states 20/30/40 are
1.180/2.948/5.875 for EBSD, versus 1.180/2.944/5.873 for the isotropic
reference.  The EBSD correction is 2.6--3.0% of the residual but nearly
orthogonal to it (cosines +0.0015 and +0.0057).  This does **not** show that
EBSD is irrelevant; it shows only that this crystallographic-elasticity
hypothesis does not explain the registered residual.

## Transfer-model boundary

The numbers above belong to the registered transfer-model analysis.  The same
report records later transfer diagnostics: removing the periodic-wrap artefact
reduces the residual by 57--71% on the tested states, and an identity transfer
keeps the residual below the pure-noise norm.  Consequently the modal
excitation is a valid statement about the registered observation model, not a
standalone proof of a plastic signal in the specimen.  Any future parameter
observability analysis must state which $M_D$ is used.

This result is therefore a useful field-observability prerequisite and a
warning about latent-state claims.  It does not establish SRIX parameter
identification, slip-system recovery, or a production boundary-only FEMU
workflow.

See the exact evidence records in
{doc}`../../reference/evidence/dic_observable_modes`, the general parametric
FEMU definitions in {doc}`../../reference/numerics/femu_sensitivity_and_svd`,
and the source report
`validation/dic_excitation_of_observable_plastic_modes.md`.
