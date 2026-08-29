# DIC-weighted FEMU observability

**Mode:** explanation  
**Domain:** identification

The sensitivity that matters for an experiment is not the raw mechanical
sensitivity alone.  The DIC pipeline observes a transformed and noisy version
of the mechanical field, so the relevant linearised map is

$$
S_{\mathrm{obs}} = W_{\mathrm{DIC}}\,O\,S,
$$

where (S) is the mechanical sensitivity, (O) is the registered DIC
observation operator and (W_{\mathrm{DIC}}) is the spectral whitener inferred
from the repeated-frame uncertainty.  This separates a parameter combination
that changes the simulated field from one that changes it at spatial scales
that the experiment can actually observe.

## Registered post-processing

The existing post-processing script
`scripts/project_dic_residuals_on_observable_modes.py` applies the measured
transfer function and repeated-frame noise to the archived P43 histories.  It
then projects the whitened residuals onto the leading singular modes.  This is
an algebraic analysis of existing artefacts; it does not run a new mechanical
forward or identify experimental material parameters.

The registered M20 and M100 artefacts use 40 history states, a fixed whitener
seed of 42, and respectively 12 and 20 retained modes.  The M20 crop has an
expected pure-noise norm of 26.8701 and an early maximum modal coefficient of
0.4016 noise standard deviations.  The M100 crop has an expected norm of
140.0071; its final leading coefficients include -118.0552 and +72.3165 in
the first and third modes.  The signal therefore grows with the larger crop,
while the M20 early response remains below the registered noise scale.

The associated report also shows that the first three early modes explain
99.70% of the early response.  After the registered heterogeneity correction,
the post-yield component reaches about 21.6 noise standard deviations in the
M100 analysis.  These numbers describe excitation of observable residual
modes, not recovery of a hidden plastic field.

## Interpretation and limits

The leading modes are edge-dominated and their correlation with the archived
DIC equivalent map is only about 0.149.  Elastic heterogeneity contributes to
the residual as well.  The corrected plastic signal is consequently a lower
bound on what can be attributed to plasticity from this analysis.

The result supports a more precise statement than “the parameters are
sensitive”: some parameter combinations excite modes above the measured DIC
noise at M100, whereas the registered M20 crop does not.  It does not establish
experimental 316L parameter identification, latent slip recovery, or a
production boundary-only FEMU workflow.  Those remain separate claims with
their own evidence boundaries.

See the exact artefacts and provenance in
{doc}`../../reference/evidence/dic_observable_modes`, and the general FEMU
definitions in {doc}`../../reference/numerics/femu_sensitivity_and_svd`.
