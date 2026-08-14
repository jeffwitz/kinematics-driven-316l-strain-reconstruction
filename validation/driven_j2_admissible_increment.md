# The driven-J2 local solve, and the wall it kept hitting

The experimental oracle prescribes an equivalent plastic increment `Delta p`
per point and asks the material for the stress that carries it. The local
equation is

```text
sigma - sigma_trial + Delta p * C : n(sigma) = 0,    n = M sigma / q(sigma).
```

It was solved by a 3x3 Newton with a backtracking line search. The directional
replay failed on it at state 21, point 117, with `driven J2 local line search
failed`, and survived neither a local continuation on `Delta p` nor the
adaptive variant added afterwards.

## The equation is scalar, not 3x3

In plane stress the elasticity `C` and the von Mises metric `M` commute, so
they share an eigenbasis: the in-plane hydrostatic mode, the in-plane
deviatoric mode and the shear mode. `M` has the fixed eigenvalues `1/2`, `3/2`
and `3`; `C` has `E/(1-nu)`, `E/(1+nu)` and `E/(2(1+nu))`.

Writing `t` for the modal trial stress and `a_i = Delta p c_i m_i`, each
component decouples,

```text
s_i = t_i q / (q + a_i),
```

and substituting into `q^2 = sum_i m_i s_i^2` leaves **one scalar equation**:

```text
phi(q) = sum_i m_i t_i^2 / (q + a_i)^2 = 1.
```

`phi` is strictly decreasing and convex on `[0, inf)`. Three consequences,
none of which the 3x3 form exposed:

- the root is unique and bracketed by `[0, q_trial]`, since `phi(q_trial) < 1`;
- convexity keeps every Newton iterate on one side of the root, so there is
  nothing for a line search to do;
- a solution exists **iff** `phi(0) > 1`, which is closed form:

```text
Delta p < Delta p_max = sqrt( sum_i t_i^2 / (c_i^2 m_i) ).
```

That bound is the whole story of point 117. Associated J2 relaxes the
deviatoric stress towards the origin as `Delta p` grows and reaches it at a
finite value; past it there is no state with `q > 0`, because the flow would
have to pass through the point where its own direction is undefined. The old
solver met a wall of non-existence and reported it as a conditioning accident,
which is what sent the investigation towards branch following.

## What is implemented

`DrivenJ2PlaneStressBatch` now solves the scalar equation with a bracketed
Newton, and exposes
`maximum_admissible_equivalent_plastic_increment(in_plane_strain)`.

Newton is started from `q_trial - a_eff`, the weighted-mean version of the
classical radial return, which is *exact* when the two distinct relaxation
values coincide. This matters: Newton from `q = 0` is monotone and cannot
overshoot, but far below the wall it advances by roughly a factor of 1.5 per
iteration from a first step of order `a/2`, so it needs `log(q_trial / a)`
iterations. At a continuation scale of `2.4e-4` that exceeded the iteration
cap, and the first version of this solver failed at state 4, point 281 —
recorded here because the failure looked like the one it had just replaced.

Every iterate is confined to the bracket, and a Newton step that leaves it
through round-off is replaced by a bisection.

## Verification

| check | result |
|---|---|
| modal basis diagonalises both `C` and `M` | asserted in the unit tests |
| `\|R\|/scale`, 20 000 random states, `Delta p` swept to 0.999 `Delta p_max` | `1.2e-15` |
| pure shear against the closed-form radial return | agrees to `1e-12` relative |
| beyond the bound | reports both `Delta p` and `Delta p_max` |

The directional diagnostic reproduces its previous numbers **bit for bit** at
states 10 and 20 — `J0 = 0.00904244`, `dJ/J0 = 4.3831e-3`, `rho =
[-0.02158, 0.05598]` and `J0 = 0.04272129`, `dJ/J0 = 6.0452e-3`, `rho =
[-0.07328, 0.06314]`. The new solver is not a different answer, it is the same
answer obtained unconditionally.

## Where the overshoot actually comes from

`scripts/diagnose_admissible_delta_p_wall.py` walks the archived oracle history
against the bound. Artefact:
`validation/_generated/performance/experimental_oracle_p43_m20/admissible_delta_p_wall.json`.

**All 40 states stay strictly inside the wall.** The worst ratio
`Delta p / Delta p_max` over the whole history is `0.871` at state 27, and
state 21 — the one that failed — peaks at `0.831`. No point reaches the wall.

So the prescribed history is admissible, and the overshoot belongs to the
**perturbed** states the directional probe visits: it moves the flow direction
by construction, which moves the trial state, which moves the bound. A
`Delta p` that was admissible for the unperturbed state need not be admissible
for the perturbed one.

## The recommendation, and what it is worth

Project `Delta p` onto `[0, 0.999 Delta p_max)` of the state actually being
integrated, inside the probe rather than inside the material. With that
projection the directional replay completes all four requested states:

| state | `J0` | `dJ_gn / J0` | `max abs(rho)` |
|---:|---:|---:|---:|
| 10 | `0.00904244` | `4.3831e-3` | `0.05598` |
| 20 | `0.04272129` | `6.0452e-3` | `0.07328` |
| 30 | `0.23204890` | `4.8671e-3` | `0.05837` |
| 40 | `0.90470059` | `4.5543e-3` | `0.06745` |

States 10 and 20 are unchanged to the last digit, so the projection acts only
where the unprojected run had no answer at all. The directional gain stays
below `0.61 %` at every state: the negative result now rests on four states
instead of two.

Two things this table does not say. The projection engaged 3221 times with a
worst requested ratio of `6.64`, so it is doing real work at some probe points,
not rounding repair — those points are genuinely far outside the admissible
set, and reporting how far is more useful than silently clipping. And `J0` at
state 40 is `0.905`: with the DIC whitening, a model at the noise level scores
`0.5`, so by state 40 the misfit has risen to about `1.35` times the noise RMS
rather than sitting below it as it does at state 10.

The projection is therefore **not** wired into `DrivenJ2PlaneStressBatch`. The
material reports the bound and refuses the inadmissible request; deciding what
to do about it belongs to the caller that chose the increment.
