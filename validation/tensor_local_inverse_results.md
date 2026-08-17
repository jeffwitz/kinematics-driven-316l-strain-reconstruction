# Milestone 3 — results: the tensor family fits better and is observable less

Against `validation/tensor_local_inverse_preregistration.md`, thresholds frozen
before the runs. One gate fails and the failure is the result.

## Verdict

| gate | registered | measured | |
|---|---|---|---|
| 1 — the transpose of `A` | `<= 1e-10` | **4.0e-15** | pass |
| 2 — the gradient | `<= 1e-5`, V-shaped | **3.4e-14**; V inapplicable, see below | pass |
| 3 — spectrum of `du/da` | `>= 90 %` above `1e-6` | **90.1 %** (173 of 192) | marginal pass |
| 4 — twin recovery, gauge | `<= 5 %` | **31 %** descending, **80 %** at the least-squares optimum | **fail** |
| 5 — admissibility | zero violations | `-1.9e-18`, active fraction 0.003 | pass |
| 6 — cost | reported | 18 s / 644 evaluations, no Newton loop | — |
| 7 — family comparison | `>= 1` decade | **5.19 decades** | pass |

## What the two decisive gates say together

**The free tensor family fits far better.** Against a twin whose truth departs
from the J2 cone by 0.50 in the gauge, the tensor family reaches **5.19 decades**
below the `Delta p + n_J2` family fitted through identical mechanics and an
identical objective. The restricted family stalls at `3.76e-10` with a 103 %
gauge error; the free one reaches `2.43e-15`. The extra tensor freedom is real
and it is what absorbs the kinematic defect, exactly as the branch-D result
originally suggested.

**And the free tensor family is not identifiable from displacement.** Gate 4
fails at 31 %, and the failure is not the optimiser. Exact least squares — the
best any method can produce from this observation — is **worse**, at 80 %:
L-BFGS-B was doing better only because early stopping regularises. A truncated
SVD sweep gives the floor directly:

| truth | truncation | directions kept | gauge error |
|---|---|---|---|
| registered | 1e-2 | 146 | 0.821 |
| registered | 1e-3 | 171 | **0.797** |
| registered | 1e-6 | 173 | 0.799 |
| zero-mean | 1e-2 | 146 | 0.559 |
| zero-mean | 1e-3 | 171 | **0.517** |
| zero-mean | 1e-6 | 173 | 5.095 |

No truncation does better than **52 %**, and past `1e-3` the reconstruction
blows up on the near-null directions. This is a property of the data, not of the
method.

## Why, in one measured fact

A spatially **uniform** eigenstrain produces exactly zero displacement:

```text
uniform xx        |du| = 0.0000e+00
uniform yy        |du| = 0.0000e+00
uniform xy_kelvin |du| = 0.0000e+00
```

Its eigenstress is uniform, so the interior divergence vanishes identically and
nothing moves. More generally any self-equilibrated eigenstress is invisible —
in two dimensions that is the whole Airy family — and the measured nullspace is
**19 directions of 192**, with condition number `3.5e16`.

Of the registered truth, **78 %** lies in that invisible subspace, which is why
its floor is 80 %. Part of that is my construction: the truth carries a large
spatial mean, and the mean is precisely what cannot be seen. Removing it drops
the invisible share to 32 % — and the floor only to 52 %. The effect is
attenuated by a better-chosen truth and not removed by one.

**The scalar family escaped this because it was restricted.** One amplitude per
point, with the tensor character pinned to the spatially varying `n_J2(sigma)`,
cannot produce a uniform eigenstrain from a uniform amplitude. Its
parameter-to-observable map had condition **200** and no nullspace at all, and
its twin was recovered to `2.1e-4`. The restriction was acting as a regulariser.

So the trade-off is sharp and now quantified:

```text
Delta p + n_J2 :  identifiable to 2e-4,  but 5.19 decades worse at fitting
free tensor    :  5.19 decades better,   but irreducibly 52-80 % wrong as a field
```

## What this means for the programme

Neither family is sufficient on its own, and this is the useful part.

The DIC displacement **cannot** determine a free tensor plastic increment. The
missing information is not a matter of more iterations, better preconditioning
or a larger network — it is a nullspace of the forward operator. Anything that
selects within it must come from outside the displacement data: a constitutive
prior, the stress or load data, temporal coherence across increments, or a
learned structure shared between points.

That last one is the architecture already proposed — rich local representation,
little shared learned structure, global mechanics — and this milestone changes
its status. The shared learned structure is not an efficiency device. **It is
what makes the local representation identifiable at all**, and it should be
designed against the measured nullspace rather than chosen for capacity.

## Three corrections recorded

**The first family comparison was meaningless and is kept.** The restricted arm
read its amplitude from the `xx` channel, which the tensile start point makes
negative; `max(., 0)` pinned it at `Delta p = 0` with a zero gradient, and it
never moved. It reported 0.00 decades and a 5.90-decade separation that measured
nothing but a dead ReLU. The arm now reads the `yy` channel and a `moved` flag
guards the comparison, without which gate 7 cannot pass.

**The rationale for the branch-D ordering was wrong.** I wrote that a mode-wise
blend could sum to an inadmissible field. It cannot: `H_sigma` is a convex cone,
so a partition of unity's non-negative blend of admissible contributions is
always admissible, and 200 random coefficient sets never produced a violation.
The ordering still matters — the two orders differ by up to **62 %** — but
because mode-wise projection clips each contribution in isolation, shrinking the
reachable family and tying it to an arbitrary decomposition into modes. The
preregistration carries the correction inline.

**The V-shape sub-criterion of gate 2 is inapplicable here, and that is proved
rather than asserted.** With the projection inactive the objective is *exactly
quadratic* in `a`, so central differences carry no truncation error and only
roundoff remains, giving a monotone `1/h` ramp instead of a V. The evidence is
the second difference divided by `h^2`, constant to ten significant digits
across four decades. The numeric threshold was met by nine orders, and the
gradient was separately verified with the projection **100 % active**, at
`2.6e-13` — without which the transpose of `P_H` would have gone untested, since
the registered base point never activates it.

## What was not done

The real DIC objective. It sits behind the elastic lifting repair, and the
comparison of gate 7 against real data is the one that decides the question for
316L. Everything here is a twin generated by the exact forward model, which
isolates the inverse from model error and says nothing about the material.

## Next

1. **Characterise the nullspace explicitly** and decide what supplies the
   missing information. The invisible subspace is computable at this size and it
   is where the design decision now lives.
2. Repair the elastic lifting with an asserted conversion, then gate 7 on the
   real DIC.
3. Only then the generator, designed against the nullspace rather than around
   it.

The reduced integration domain stays closed; its reopening condition is
registered elsewhere and nothing measured here touches it.
