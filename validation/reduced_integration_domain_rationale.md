# The reduced integration domain: what it is, and when it will pay

This page exists so nobody has to rediscover, six months from now, why a
hyper-reduction was designed, why it was not adopted for J2, and why it is
expected to matter for crystal plasticity. The short version is at the bottom;
the arithmetic that justifies it is above.

## 1. The problem it addresses

A nonlinear mechanical increment is a Newton loop. Each iteration needs, at
every material point, the stress and the algorithmic tangent that the
constitutive law produces from the current strain and the committed internal
state:

```text
(eps, z_n)  ->  sigma, C_alg
```

At full field that is **22.3 million material points** per Newton iteration.
Measured on the J2-Ludwik bench, the constitutive integration and the residual
assembly account for **76 %** of a nonlinear run at 512 pixels square, and 74 %
at 256. The linear solver — Jacobian applications, preconditioner, GMRES
overhead — is the remaining quarter, of which the preconditioner alone is 6 %.

So the dominant cost is not the mechanics. It is calling the constitutive law
tens of millions of times.

## 2. What Ryckelynck's hyper-reduction does, and what we do differently

Ryckelynck's hyper-reduction (2009, and the 2016 calibration framework) reduces
the **displacement space** to a POD basis, and evaluates the constitutive law
only on a *reduced integration domain*, extrapolating the internal variables
elsewhere from reduced information.

**We deliberately keep the displacement full.** Everything this project has
established rests on the full-field equilibrium operator: an adjoint qualified
at 4.4e-17 over 22.3 million unknowns, a preconditioner independent of the mesh
from 24 to 3599 pixels square, and the finding that the number of local plastic
coefficients does not change the mechanical cost at all. Reducing the
displacement would give all of that away, and the earlier campaigns already
refuted global reduced representations of this field.

What we take from Ryckelynck is narrower and, for our purposes, the valuable
part: **evaluate the expensive constitutive law on a subset of points and
reconstruct it elsewhere**, while the residual is still assembled over every
point.

## 3. The construction

Split the stress around the last converged state and a homogeneous elastic
reference `C_0`:

```text
sigma(u) = sigma_n + C_0 : d eps(u) + h(u)
                     \___________/   \___/
                      cheap, exact,   expensive,
                      everywhere      sampled
```

`h` is the nonlinear correction. It vanishes at the converged state — `h(u_n) =
0` — which is why the split is taken there and not at the origin: during an
optimisation the solve is a small perturbation around exactly that point, so the
sampled quantity starts at zero and stays small.

Evaluate `h` exactly only on a reduced integration domain `S`, reconstruct it
over the domain with an operator `R_S`, and assemble the residual everywhere:

```text
R_HR(u) = B^T [ sigma_n + C_0 : d eps(u) + R_S(h_S(u)) ]
```

The Jacobian follows the same structure, with `Delta C = C_alg - C_0` sampled on
the same points and no constitutive law re-entered during a Jacobian-vector
product:

```text
J_HR v = K_0 v + B^T R_S[ Delta C_S : P_S B v ]
```

`R_S` is not symmetric, so its transpose is implemented separately and tested,
never assumed.

## 4. What it can gain, exactly

Let `r = N_RID / N` be the sampled fraction and `n` the Newton count. With one
full-field certification per solve, the constitutive speedup is

```text
S = n / (n r + 1)
```

At `r = 0.1`: **4.44** at eight Newton, 1.67 at two, and **0.91 at one** — a
loss. Certification costs a full integration, so it can exceed the entire solve
it certifies.

That matters because the regime training lives in is *one* Newton: a coefficient
perturbation re-solved from a nearly correct state was measured at 1 Newton and
19 Krylov, against 8 Newton cold. So certification must be **amortised** over
several optimisation steps. Certifying every `q` steps gives

```text
S = 1 / (r + 1/q)
```

3.33 at `q = 5`, 5.0 at 10, 6.67 at 20, tending to `1/r`.

## 5. Why it will not pay for J2 — the two ceilings

**Amdahl.** Those figures describe the constitutive term, which is 76 % of the
run. Over the whole calculation:

```text
r = 0.10, perfect amortisation :  1 / (0.24 + 0.076) = 3.16
r = 0.02, perfect amortisation :  1 / (0.24 + 0.015) = 3.92
```

The 24 % that is mechanics, residual assembly and vector algebra bounds the
whole method near **4**, however aggressive the sampling.

**The remedies are substitutes, and this is now measured end to end.** The same
J2 bench at 256 pixels square, eight increments, differing only in backend:

| backend | total | GMRES | constitutive + residual | share |
|---|---|---|---|---|
| Python batch | 160.6 s | 46.4 s | 114.2 s | 71.1 % |
| **MFront, 8 threads** | **73.6 s** | 40.7 s | **32.9 s** | **44.7 %** |

**2.18x on the whole run**, and 3.5x on the constitutive term itself -- better
than the 1.9 measured on an isolated plastic evaluation, since a real run mixes
branches. The constitutive share falls from 71 % to 45 %, and the
hyper-reduction ceiling falls with it:

```text
1 / (0.55 + 0.045) = 1.68
```

Switching backend has already consumed most of what a reduced integration
domain could return on J2. Both attack the same term; they do not compose.

So for J2 the arithmetic is unattractive. A threaded backend is a *parameter*
and delivers 2.18x today. The hyper-reduction delivers at most 1.68x on top of it,
and requires a reduced integration domain, a reconstruction operator, an exact
transpose, a certification policy, sentinel points, adaptive enrichment, and a
fallback path. The effort-to-gain ratio is poor and the honest thing is to say
so before spending days on it.

## 6. Why it will pay for crystal plasticity

Everything above is a statement about **one number**: the constitutive share of
the run. For J2 it is 76 %, and Amdahl caps the method near 4. That share is a
property of the constitutive law, not of the method.

J2 in plane stress solves a **scalar** local problem: one plastic multiplier,
one radial return. Forest-Rubin FCC crystal plasticity solves a local problem
with **twelve slip systems** and, in the generalised plane-stress closure, a
joint Newton on 21 unknowns. The registered figures for it are in a different
class: `2.81 ms` per material evaluation for the condensed reference, `0.42 ms`
for the specialised UMAT closure — against roughly `2.6 microseconds` per point
for J2 on the plastic branch.

That is three orders of magnitude of local work. When the local problem is that
expensive:

* the constitutive share rises from 76 % towards 95 % and beyond;
* the Amdahl ceiling rises with it, from about 4 towards `1/r` — a factor of
  ten at a tenth of the domain, fifty at a fiftieth;
* threading the backend no longer competes, because it is already threaded and
  the work is genuinely there rather than in marshalling;
* and the elastic branch, where sampling is worthless because `h = 0`, becomes
  a negligible part of the domain once the specimen is plastically active.

There is a second reason, structural rather than arithmetic. Crystal plasticity
carries far more internal state per point — twelve slip amplitudes, their
accumulated counterparts, hardening variables — and that state is what makes
each evaluation expensive. A reduced integration domain reduces exactly the
thing that scales with that state, while the mechanics it leaves untouched does
not depend on it at all.

## 7. What the RID must respect, whenever it is built

These are not optional, and each of them is a way the method fails silently.

**Sentinel points disjoint from the RID.** Monitoring the approximation at the
points used to build it is close to tautological. A separate, partly refreshed
set is what detects a plastic band moving into a region the domain does not
cover.

**The domain frozen during a Newton solve.** If `S` changes mid-solve, `R_HR`
changes definition and the Jacobian is no longer the derivative of the residual
being solved. Enrichment happens between solves, never inside one.

**The committed state never approximate.** After a hyper-reduced solve
converges, the increment is integrated exactly at full field and *that* state is
committed, so reconstruction error cannot accumulate from increment to
increment.

**The reported residual always the exact one.** `|R_full(u_HR)|`, never
`|R_HR|`. A hyper-reduced residual can be small for a solution that is not
equilibrated.

**The gradient checked, not just the field.** Training consumes the gradient
with respect to the local coefficients, so `cos(g_HR, g_FOM)` and its relative
error matter more than the displacement error. A biased gradient can point
somewhere plausible and converge to the wrong answer.

## 8. Where this stands

Implemented and tested: the exact split `sigma = sigma_n + C_0 : d eps + h`,
with `C_0` read from the behaviour's own tangent rather than assumed — these
batches are engineering while the spectral solver is Kelvin, and chaining the
two conventions is what left the P43 elastic lifting retaining 32 % of its
interior residual. Seven tests cover the reassembly, the vanishing correction at
the committed state, and the immutability of the committed state under trials.

Not implemented, deliberately: the reduced domain itself, the reconstruction
operator, certification and enrichment. The arithmetic above says they would
return at most 1.68x on J2 over the threaded backend, and the threaded backend
was one parameter for 2.18x.

**The trigger for resuming this work is the crystal-plasticity campaign.** When
the constitutive share passes about 90 %, the ceiling passes 6 and the effort is
justified. The split is already in place for that day, and the specification it
should follow is
`validation/constitutive_hyperreduction_preregistration.md`.

---

**In one paragraph.** A reduced integration domain evaluates an expensive
constitutive law on a fraction of the points and reconstructs it everywhere,
while equilibrium stays full-field. Its ceiling is set by how much of the run
the constitutive law occupies. For J2 that is 76 %, capping the method near 4 —
and a threaded MFront backend already returns 2.18 of that for the cost of one
parameter, dropping the share to 45 % and the remaining ceiling to 1.68, which
is too little to justify the machinery. For crystal plasticity,
where a single point costs three orders of magnitude more, the share approaches
95 % and the ceiling approaches `1/r`. The method is right; J2 is simply the
wrong problem to spend it on.
