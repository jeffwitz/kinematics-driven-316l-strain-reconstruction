# SRIX-REGM versus FEMU ranking preregistration

Date: 2026-08-23  
Status: **frozen before the first population forward run**

## Question

Does the cheap SRIX-REGM objective rank off-truth constitutive parameter sets
in the same order as a complete nonlinear forward simulation on the exact M8
digital twin?

## Fixed experiment

- target: the already archived exact M8 SRIX twin;
- parameters: `(tau0, R, Q, b)`, with `C`, `d`, elasticity and FCC interaction
  coefficients fixed;
- REGM: replay the archived target strain history, weak interior equilibrium
  residual, EBSD-cubic `K0`, identity observation and whitening;
- FEMU control: run the complete adaptive nonlinear solver for every candidate
  and compare the eight macro-endpoint interior nodal displacements with the
  corresponding target fields;
- no global Newton or Krylov iteration is allowed inside a REGM evaluation;
- the true point is evaluated as a numerical-floor control but excluded from
  correlations, because a shared exact zero would artificially anchor them;
- 20 off-truth candidates are used: the frozen initial point followed by 19
  Latin-hypercube log offsets generated once with seed `20260824`;
- all four log offsets of the Latin hypercube lie in `[-0.30, +0.30]`.

The 19 offsets `(log tau0, log R, log Q, log b)` are:

```text
01 -0.106535 +0.195647 +0.282258 +0.166001
02 -0.268867 +0.266724 +0.229828 +0.207584
03 -0.003049 +0.146794 -0.241087 -0.138832
04 -0.048460 -0.099961 -0.045376 -0.018534
05 +0.183792 +0.116277 +0.160055 +0.030313
06 -0.178428 -0.023892 +0.266417 +0.100787
07 +0.252891 -0.049726 -0.122052 +0.198030
08 -0.030582 +0.290063 -0.050611 -0.095495
09 -0.224258 -0.112246 -0.083937 -0.164764
10 +0.079777 -0.277424 -0.224168 -0.212370
11 -0.250035 +0.033006 +0.093018 -0.264816
12 -0.169448 -0.207373 +0.036584 -0.291297
13 +0.128381 +0.219309 +0.128248 +0.123126
14 +0.226227 +0.066613 -0.168320 -0.007200
15 +0.271877 -0.001076 -0.007494 +0.061936
16 +0.058258 -0.253252 +0.196774 +0.297952
17 -0.130702 +0.089323 +0.077862 -0.060408
18 +0.028116 -0.172289 -0.199220 -0.191474
19 +0.144311 -0.180116 -0.295437 +0.264668
```

## Objectives and statistics

For candidate `i`:

- `J_REGM` is the RMS of the reconditioned pseudo-displacement vector;
- `J_FEMU` is the RMS, in mm, of the interior nodal displacement difference
  between candidate and target over the eight macro endpoints.

Correlations use only successfully converged off-truth candidates. Pearson is
computed after the natural logarithm of the strictly positive objectives.
Spearman uses the raw ranks. The best-five overlap is the intersection of the
five smallest objectives. At least 15 off-truth forward runs must converge for
the gate to be interpretable. Failures remain explicit rows and are never
silently replaced.

## Frozen gate

Proceed toward P43 only if all conditions hold:

1. Spearman correlation is at least `0.80`;
2. log-objective Pearson correlation is at least `0.70`;
3. the best-five overlap contains at least three candidates;
4. at least 15 off-truth candidates have a complete forward result.

The performance target is a median speedup of at least `5x`. Missing the
performance target triggers profiling but is not a scientific rejection.

If the ranking gate fails, no P43 optimization is authorized. The negative
result and the existing transfer/noise gate are documented without tuning the
population or the thresholds.
