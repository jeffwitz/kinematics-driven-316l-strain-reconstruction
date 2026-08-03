# SRIX at the material point: sensitivity to R, to the elasticity, and to orientation

Date: 2026-08-03
Section 11 of the 2026-08-03 specification.
Generator: `scripts/qualify_srix_material_point.py`
Archive: `validation/_generated/srix/`

Twenty-eight single-point runs: four crystal directions against seven parameter
sets. Uniaxial tension to 2 percent axial strain, 100 increments, room
temperature. No experimental data, no cluster, a few seconds in total.

**Nothing here is an identification.** `(Q, b, C, d)` and the interaction matrix
are held fixed at their historical values in every run, so a difference between
two rows is attributable to what was varied and to nothing else. The three
things varied — the overstress modulus `R`, the cubic elasticity, and the
orientation — are the three the specification asks about at this stage.

## What moves the answer

### Orientation moves it most

With the historical set, at 2 percent axial strain:

| axis | axial stress (MPa) | cumulated slip | active systems |
|---|---:|---:|---:|
| `[001]` | 185.7 | 0.0448 | 8 |
| `[011]` | 298.1 | 0.0695 | 8 |
| `[111]` | 297.9 | 0.0655 | 6 |
| `[123]` | 278.4 | 0.0616 | 9 |

A factor **1.6** between the softest and the stiffest direction, and a different
number of active systems in each. `[011]` and `[111]` land within 0.1 percent of
each other by coincidence of this parameter set, not by symmetry: they activate
8 and 6 systems respectively and their overstress distributions differ (below).

### R moves it more off-axis than on-axis

Sweeping `R` with everything else fixed:

| `R` (MPa) | `O_R` | `[001]` | `[123]` |
|---:|---:|---:|---:|
| 1 | 0.0077 | 172.4 | 234.7 |
| 2 | 0.0153 | 173.2 | 237.6 |
| 4 | 0.0306 | 174.7 | 243.0 |
| 8 | 0.0612 | 177.6 | 252.9 |
| 18.78 | 0.1438 | 185.7 | 278.4 |

**7.7 percent** of spread on `[001]`, **18.6 percent** on `[123]`, over a sweep
that leaves the cumulated slip essentially unchanged (0.0451 to 0.0448 on
`[001]`). `R` is therefore not a small correction on a general orientation, and
it cannot be calibrated on a `[001]` curve and assumed transferable.

`R` also changes the **active set**, not only the stress: `[123]` runs on 8
systems for `R <= 8` and on 9 at `R = 18.78`. A parameter that changes which
systems carry the deformation is not a scaling factor.

### The updated elasticity moves it by about two percent

`316l_srix_updated_elasticity_prior` gives 181.6 MPa on `[001]` against 185.7,
and 273.0 on `[123]` against 278.4 — 2.2 and 1.9 percent. That set also changes
`tau0` from 40 to 38.33, so the two effects are not separated here; the row is a
combined sensitivity, not an elastic one.

## Numerical quality of these runs

**Step convergence** is clean and first-order on every axis. Relative error
against a 400-increment reference:

| axis | N=10 | N=20 | N=40 | N=80 | N=160 |
|---|---:|---:|---:|---:|---:|
| `[001]` | 7.8e-4 | 2.6e-4 | 9.7e-5 | 3.7e-5 | 1.3e-5 |
| `[011]` | 2.4e-4 | 7.7e-5 | 2.8e-5 | 1.1e-5 | 3.8e-6 |
| `[111]` | 2.1e-4 | 8.8e-5 | 3.9e-5 | 1.7e-5 | 6.2e-6 |
| `[123]` | 5.5e-4 | 2.6e-4 | 1.2e-4 | 5.2e-5 | 1.9e-5 |

Monotone throughout, and already below `1e-3` at ten increments. **This is the
monotonic case only.** A reversal has a step size below which it is
qualitatively wrong, not merely inaccurate; see
`validation/srix_canonical_qualification_report.md`.

**Dissipation** is non-negative on every system at every increment, on every run.
The balance at 2 percent, historical set, in MPa (energy per unit volume):

| axis | elastic | stored isotropic | stored kinematic | dissipated |
|---|---:|---:|---:|---:|
| `[001]` | 0.160 | 0.004 | 0.071 | 2.116 |
| `[011]` | 0.160 | 0.010 | 0.071 | 3.609 |
| `[111]` | 0.132 | 0.014 | 0.053 | 3.727 |
| `[123]` | 0.151 | 0.013 | 0.063 | 3.480 |

Dissipation dominates by more than an order of magnitude, as it should for a
monotonic path at this strain, and the stored terms are small but not zero.

**Overstress** at the last increment, historical set:

| axis | max | q99 | q95 | mean over active | fraction above 10% | active |
|---|---:|---:|---:|---:|---:|---:|
| `[001]` | 0.132 | 0.132 | 0.132 | 0.132 | 0.67 | 8 |
| `[011]` | 0.260 | 0.260 | 0.191 | 0.191 | 0.67 | 8 |
| `[111]` | 0.371 | 0.370 | 0.367 | 0.230 | 0.42 | 6 |
| `[123]` | 0.470 | 0.453 | 0.382 | 0.170 | 0.33 | 8 |

`[001]` is uniform by symmetry — all eight active systems carry the same
overstress, so max, q99, q95 and mean coincide. The general orientation `[123]`
has the widest spread. These are descriptive: the flow rule is linear in the
overstress, so a large value means the increment demanded a lot of slip, not
that anything went wrong.

## What this does not establish

- **No parameter is identified.** Every set used here carries at best
  `literature_prior` on its hardening and `analytical_transposition` or
  `exploratory` on `R`. The archive records that per group.
- **The sweep is one-dimensional.** `R` was varied alone. Its interaction with
  `tau0` — the two together set where and how sharply yield begins — is not
  explored, and the `O_R` column exists precisely because that combination, not
  `R`, is the dimensionless quantity that matters.
- **One strain level, one path.** Monotonic tension to 2 percent. Nothing here
  says how `R` behaves on a reversal, where the back stress is what dominates.
- **A single point.** No gradient, no neighbour, no finite element.

## Reading for the calibration

Three things follow for `validation/srix_316l_calibration_preregistration.md`.

`R` must be identified on a **general** orientation or on several, not on `[001]`
where its influence is smallest and where it happens to be analytically tied to
the Méric-Cailletaud correspondence.

`R` and `tau0` must be identified **together or in a stated order**, because
`O_R` couples them and only their combination is observable in the width of the
transition.

The elasticity must come **first**, since a 2 percent change in it moves the
stress by as much as a factor two in `R` does at `[001]`.

## Reproduction

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
python scripts/qualify_srix_material_point.py --output validation/_generated/srix
```
