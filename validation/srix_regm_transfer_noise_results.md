# SRIX-REGM transfer/noise twin — Gate 4 result

Date: 2026-08-23  
Primary artefact:
`reference_data/srix_regm_transfer_noise_v1/report.json`  
Pre-registration: `srix_regm_transfer_noise_preregistration.md`

## Result in one sentence

The measured DIC transfer and noise preserve four non-zero local sensitivity
directions, but they destroy the exact-twin optimum: parameter sets far below
the true SRIX strengths produce a smaller REGM residual than the truth.

## Quantitative result

| Level | truth RMS (mm) | initial RMS (mm) | lowest reached RMS (mm) | reached / truth | condition number |
|---|---:|---:|---:|---:|---:|
| exact T0 | `1.474e-13` | `3.143e-8` | `1.412e-13` | `0.958` | `2.15e4` |
| transfer T1 | `2.132e-7` | `2.310e-7` | `1.300e-7` | `0.610` | `7.90e4` |
| transfer + noise T2 | `1.741e-3` | `1.894e-3` | `1.346e-3` | `0.774` | `9.75e4` |

The optimizations were deliberately stopped at the pre-registered 16 function
evaluations and did not declare numerical convergence. That does not weaken
the negative observation: a single admissible point with lower residual is
already sufficient to show that the true parameters are not a minimizer of
the stated transferred/noisy objective.

The lowest points reached were:

| Level | tau0 / truth | R / truth | Q / truth | b / truth | log-error RMS |
|---|---:|---:|---:|---:|---:|
| T1 | `0.410` | `0.347` | `0.443` | `0.366` | `0.947` |
| T2 | `0.655` | `0.527` | `0.680` | `0.487` | `0.560` |

These are not reported as identified parameters. They are counterexamples to
the hypothesis that the current REGM observation objective is centred on the
truth.

## Sensitivity

Using the frozen relative rank threshold `1e-6`, all levels retain rank four.
The weakest normalized singular value nevertheless falls from `4.65e-5` for
T0 to `1.27e-5` for T1 and `1.03e-5` for T2. The condition number grows by
roughly a factor four. Thus the formal rank survives, but the weakest
combination becomes less robust and should not be interpreted as a precise
four-parameter determination.

## Interpretation

The test separates two facts:

1. The SRIX replay and weak-equilibrium implementation remain sensitive to the
   constitutive parameters after transfer and whitening.
2. Sensitivity is not sufficient for unbiased identification. The observation
   operator changes the equilibrium-gap objective so that the generating
   parameter vector no longer minimizes it.

The most likely mechanism is not a constitutive integration failure: T0
recovers the truth on the same history and backend. Rather, applying a spatial
measurement transfer to a mechanically equilibrated displacement does not in
general preserve mechanical equilibrium. The transferred displacement is
therefore not a mechanically exact SRIX trajectory, even when its unobserved
source was one.

This is a **negative Gate-4 result for direct parameter recovery with the
current transferred/noisy REGM objective**. It does not yet close the whole
method: the independently pre-registered Gate 5 asks the weaker and more
useful question of whether REGM at least ranks candidate laws like a complete
FEMU. P43 remains blocked until that ranking is measured.

## Reproducibility incident

The first noise implementation attached independent measured noise to all 338
adaptive solver substeps, although only eight macro endpoints are camera
observations. It made the constitutive history non-integrable and produced no
result artefact. The protocol was amended before a result existed: measured
noise is sampled at the eight observation times and interpolated causally over
the hidden adaptive steps. This amendment is recorded in the preregistration
and commit history.
