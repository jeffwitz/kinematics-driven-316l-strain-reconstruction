# Why the temporal loading path matters

**Mode:** explanation  
**Domain:** reconstruction

For a material with memory, the final displacement is not a sufficient
description of the final material state. A local update can be written

$$
q_{n+1}=\Phi(q_n,\Delta\varepsilon_n),
$$

so, in general,

$$
\Phi_N\circ\cdots\circ\Phi_1\neq\Phi_{\mathrm{total}}.
$$

The practical message is simple:

```text
same final displacement
≠ same constitutive history
≠ same plastic state
```

## Three notions that must not be conflated

**Physical loading path** is the sequence of boundary strains or displacements
actually followed by the specimen. Two paths can share an endpoint while
activating different slips, reversals or hardening histories.

**Numerical temporal refinement** splits one prescribed physical path into more
substeps. Under a convergent integration scheme, refining the same path should
approach the same solution; it is not a new physical experiment.

**Physical rate dependence** changes the response when the same path is
traversed at a different physical speed. This is a material property, not a
convergence test. A run that fails with eight increments and converges with
sixteen may indicate an integration or local-Newton robustness problem; it does
not by itself demonstrate physical rate dependence.

## SRIX and Méric--Cailletaud

In the registered use of SRIX, the law is rate-independent: its physical
response depends on the strain path, not on the elapsed time used to parameterise
that path. The numerical increments are only a discretisation of the prescribed
path and should cease to be an independent influence under temporal convergence.
At a fixed physical path, changing the numerical partition should therefore
converge to the same response. A different response under a different
non-proportional path is nevertheless expected.

Méric--Cailletaud contains a viscous/rate-dependent evolution. Its response can
therefore depend on the physical loading rate, provided that a physical or
pseudo-time scale is part of the material contract. Its numerical integration
must still converge when the same time history is refined. The distinction
between these two effects is why the archived eight-versus-sixteen-increment
comparison is recorded as a robustness observation, not as proof of a physical
time-step effect. See the [Méric reference](../../reference/scientific/meric_cailletaud)
for the constitutive contract.

## What the registered P43 replay shows

The measured 40-state boundary history and a proportional ramp were driven to
the same final boundary displacement. Under the same J2 mechanics, their core
PEEQ fields differ by **15.82% relative L2**, with a band-structure ratio of
**13.11**. The 20-versus-40-increment proportional control differs by only
**0.2017%**, so the discretisation effect is about 78 times smaller than the
path effect. This is a direct demonstration that an endpoint does not determine
the accumulated plastic state.

The observable conclusion is more cautious. After both mechanical fields are
passed through the same symmetric DIC observation, the final total-strain
comparison is registered as indistinguishable: the relative-L2 difference
between measured-path and proportional-path predictions is `0.01545`, below
the pre-registered margin `0.0202`. The internal PEEQ difference is therefore
real in the model but largely invisible in this DIC observable at the tested
scale. Path-dependent state and path-insensitive observation can coexist.

These are computed-path comparisons, not a claim that the specimen's true load
fractions are known. The image states are ordered observations without a
synchronised force history, and PEEQ is not directly measured by DIC.

## Why all frames remain part of the data contract

The intermediate images serve two distinct purposes:

```text
boundary history
    → drives the path-dependent mechanical forward

interior DIC history
    → supplies repeated observables for comparison and identification
```

Replacing the sequence by its endpoint turns a history-dependent forward into
a different problem. It can be a useful control, such as the proportional ramp,
but it cannot silently replace the recorded path. A future information analysis
may select a subset of frames for a cheaper FEMU, but that selection must be
declared and justified rather than assumed here.

## Consequence for FEMU

The identification forward must replay the same boundary history before the
model prediction is compared with the observations:

```text
theta
  ↓
measured boundary history u_b(t)
  ↓
path-dependent mechanical forward
  ↓
u(theta,t)
  ↓
O_DIC
  ↓
compare selected experimental frames
```

This is not equivalent to applying one final boundary displacement and asking
for a one-shot constitutive state. A richer path can, in principle, separate
parameter combinations that are confounded under proportional loading, for
example through reversals or non-proportional increments. Whether it does so is
an observability question, not an assumption.

The registered conclusions and solver limitations are detailed in
`validation/dic_multistep_p0043_path_dependence_results.md` and
`validation/dic_multistep_p0043_observed_path_comparison_results.md`. The
measured-history replay itself also records the current numerical limitation:
the undamped solver fails on an early transition, so the replay is a diagnostic
and not yet a production multi-step FEM/DIC prediction.
