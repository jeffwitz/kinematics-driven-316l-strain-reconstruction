# CPS4R-AS — qualification preregistration

Date: 2026-08-04
Written before any CPS4R-AS result exists. The element algebra is implemented
and verified; no case has been solved with it.
Sections 16, 17, 19 and 25 of the 2026-08-04 specification.

## What is being qualified, and against what

Four formulations, all on identical meshes, increments and boundary data:

| name | constitutive points | stabilisation |
|---|---:|---|
| `CPS4` | 4 | none — **the scientific reference** |
| `CPS4R-elastic` | 1 | `beta (K_full - K_reduced)` on a frozen elastic tangent |
| `CPS4R-AS-current` | 1 | assumed strain on the **current** tangent |
| `CPS4R-AS-energy` | 1 | the same, on a spectrally floored symmetric tangent |

`CPS4R-elastic` is carried as a **falsified baseline**, not a candidate: its
qualification failed on 2026-08-03 and its `beta` is exactly the free coefficient
this work removes.

## The thresholds, frozen

All are relative `L2` norms over the element grid, against the `CPS4` solution
of the same case.

| criterion | symbol | bound |
|---|---|---|
| equivalent plastic strain, J2 | `E_p` | **< 1 %** |
| cumulated slip, SRIX (`Gamma = sum_s p_s`) | `E_Gamma` | **< 1 %** |
| stress | `E_sigma` | **< 1 %** |
| reactions | `E_R` | **< 1 %** |
| constitutive speed-up | `S_const` | **> 3.5** |
| total speed-up | `S_total` | **> 1.8** |

A 2 % level may be *reported* as exploratory but never as qualification.
Displacement error is reported three ways — relative, absolute RMS, and as a
multiple of the DIC noise `9.40e-5 mm` — and **a small displacement error may
not be used to offset a constitutive one**; the constitutive criterion governs.

Localisation is compared on centre of mass, orientation, width, amplitude,
connectivity and position of the maxima, **separately**. No single score.

## Registered cases

Homogeneous: isotropic elasticity, oriented cubic elasticity, J2 monotonic, J2
unload, J2 tension–compression, SRIX on `[001]`, `[011]`, `[111]`, `[123]`, SRIX
reversal, SRIX shear, SRIX biaxial.

Heterogeneous: the pixelwise J2 case already used for the CPS4R qualification
(same mesh, same increments, same reference), a hard–soft checkerboard, an
oblique interface **aligned on element boundaries**, a synthetic polycrystal with
one orientation per element group, and a controlled hourglass perturbation added
to a physical field.

## What may not be asked of a one-point element

> A formulation with one constitutive point cannot represent two materials or
> two orientations inside one element.

No case will place an interface, a grain boundary or a sub-element plastic front
inside an element and then score the stabilisation on reconstructing it. Those
belong to mesh refinement, multiple quadrature or an enriched method, and their
failure is **not** a failure of hourglass control.

## Falsifiers

**F1 — the constitutive-call guarantee.** If any variant records more than one
constitutive evaluation per element per Newton iteration, measured by an
instrumented material that counts calls, it is disqualified outright. This is
not a threshold to negotiate; it is the premise of the whole formulation.

**F2 — spurious modes.** If the stabilised element shows any null mode beyond
the three rigid-body ones, or produces cutbacks CPS4 does not, the verdict is
**case D** and no accuracy number is reported.

**F3 — accuracy bought with increments.** If a variant reaches the accuracy
bounds only with a substantially larger increment count than CPS4, it is not
qualified. Converged pseudo-times, cutbacks, increment sizes and Newton history
are archived for every run so this can be checked rather than asserted.

**F4 — the tangent floor.** If the verdict changes when `relative_floor` moves
over `1e-8` to `1e-4`, the floor is doing physical work rather than conditioning
work and must be re-derived. It will **not** be tuned to recover a passing
result.

**F5 — frame dependence.** `asoi` and `asoi_half` are measured to change their
stabilisation energy by 38 % under a mesh rotation. They may be compared on the
axis-aligned pixel meshes this project uses, where every element shares one
frame, but a verdict obtained with them carries that restriction explicitly and
does not transfer to a general mesh. `asmd` is the default precisely because it
has no such restriction.

## What is forbidden, restated because it is easy to drift into

No search for a better `beta`. No spectral bound fitted to CPS4. No threshold
moved after seeing a result. Hourglass energy is a diagnostic and **never** a
certificate of accuracy. CPS4 remains the reference until a verdict exists, and
CPS4R-AS is not used for any scientific conclusion before one.

## Performance protocol

Constitutive, assembly, linear-solve, Newton and total times measured
separately, as **medians over at least five repetitions after a warm-up**, with
memory, MFront call count, Newton iterations and cutbacks recorded alongside. A
single elapsed time is not a measurement; that lesson cost a nearly-published
finding on 2026-08-03.

## The verdict

One of the four forms of section 26 — A qualified, B fast but imprecise, C
precise but slow, D unstable — written into
`validation/cps4r_assumed_strain_report.md` together with every number behind
it, including the ones that fail.
