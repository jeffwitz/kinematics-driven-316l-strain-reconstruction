# Choose an MFront backend

For the qualified 316L SRIX + EBSD workflow, use
`mfront-native-generalised-plane-stress` with
`gps_composite_fd_tangent: true`. Use
`mfront-3d-condensed-plane-stress` as the independent numerical reference and
for 3D behaviours without a GPS implementation.

| Backend | Recommended use | Advantage | Limitation |
|---|---|---|---|
| `mfront-native-plane-stress` | simple native 2D laws | direct | not general for 3D crystal plasticity |
| `mfront-3d-condensed-plane-stress` | independent reference / new 3D law | works with any 3D law | Python condensation |
| `mfront-native-generalised-plane-stress` | qualified SRIX GPS workflow | monolithic and performant | requires a GPS law variant |
| `mfront-structural-plane-stress` | generic qualified SRIX workflow | reusable structural closure | limited to the demonstrated V1 MFront contract |
| `python` | historical J2 regression | independent of MFront | not production crystal plasticity |

GPS (*generalised plane stress*) retains all six three-dimensional components
and locally solves the three transverse strains required to impose
`sigma_zz = sigma_xz = sigma_yz = 0` in the structural frame. The specialised
SRIX GPS variant carries this closure in its constitutive Newton. The generic structural
backend applies the same closure through the reusable `StructuralPlaneStress3D`
transformation, while the condensed route performs it in the Python bridge.

## Why there are two broad ways to do plane stress on a 3D law

The repository contains three implementations of two broad mechanisms. All
three compute the same physical closure -- they agree to `1e-11` at a material
point -- so the choice is not about accuracy. It is about what each one lets
you do.

**Condensed 3D works with any law.** The bridge iterates the transverse strain
and hands the behaviour a complete six-component gradient. The law is never
modified and does not know plane stress exists, so a new constitutive model
becomes usable the day it compiles. That is why it stays the reference. It
costs roughly `6.6` full integrations per material point per global Newton
iteration, and the closure lives in Python, so it cannot leave this repository.

**Generalised plane stress travels.** The closure is part of the law, so any
code able to call an MFront/MGIS behaviour obtains plane stress without writing
a closure loop, and the logic cannot drift between hosts. It costs one law
written for it per behaviour.

## Why `gps_composite_fd_tangent: true`

When a GPS point must sub-step an increment, the last sub-step does not by
itself provide the derivative of the composed trajectory. The composite FD
tangent reconstructs that derivative for the affected points only.

It is not a refinement, it is what makes the route worth taking. Switching the
option off and changing nothing else, on P43 M100 EBSD at eight increments:

| | time | Newton |
|---|---:|---:|
| condensed reference | `62.38 s` | 57 |
| GPS, tangent as the DSL returns it | `74.05 s` | 85 |
| **GPS + composite FD** | **`58.38 s`** | **58** |

The finite difference touched `192` points and `1152` trajectories for `2.15 s`
of its own cost, with a converged residual of `5.3e-09`. Without it the GPS
route is *slower* than the reference; with it, it is faster.

That experiment is the only archived one that measures the option against
itself, and it was run under `2defce9`: `srix_p43_m100_condensed_runtime_blas1`,
`gps_composite_fd_m100_runtime_blas1` and `gps_fd_m100_runtime_blas1` under
`validation/_generated/performance/`. Read its three times as one ratio, not
alongside the absolute times below, which come from a later commit on the same
case — comparing across the two sets measures the intervening work, not the
backends.

Sub-stepping is local. In the comparison campaign it affected `192` points on
the M100 crop (about `1.92 %` of its `10,000` constitutive points); on the
smaller M20 crop it affected `19` points. The repair remains cheap because it
is applied only to those points.

The condensed backend accepts the key and ignores it — it has no local Newton
to sub-step, so there is nothing to repair. Switching a configuration from GPS
to condensed therefore silently drops the option rather than failing, so read
`constitutive_backend` before crediting it with anything.

## What each route costs, all three together

The current comparison, all three backends on the same case and the same
commit — P43 M100 EBSD, eight increments, four MFront threads, BLAS/FFTW/OpenMP
pinned to one thread each, executed under `c8af766` and archived as
`p43_m100_backend_comparison_latest.json`:

| Backend | time | Newton |
|---|---:|---:|
| `mfront-3d-condensed-plane-stress` | `56.72 s` | 57 |
| `mfront-native-generalised-plane-stress` | `51.65 s` | 58 |
| `mfront-structural-plane-stress` | `54.56 s` | 58 |

The spread is about `10 %`; the routes are not separated by cost, so choose on
what each one lets you do.

They are also not three numerical answers. Against the hand-written GPS run,
the generic structural closure agrees to `1.2e-16` on displacement and
`4.8e-12` on in-plane stress, both backends sub-stepping the same `192` points.
Prefer the generic route when you want the GPS closure for a 3D law you have
not hand-written a GPS variant for, and it fits the demonstrated V1 contract.

The same three-way comparison exists at M20 as
`p43_m20_backend_comparison_latest.json`, if you want a case that runs in
seconds.

## Qualified production route

```yaml
solver:
  constitutive_backend: mfront-native-generalised-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4

  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6

    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3

    crystal_orientation:
      mode: ebsd
      # orientation source defined by the case
```

## Independent reference

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
```

The first configuration is the qualified production route for SRIX + EBSD. The
second is the independent reference to qualify a new law or check a GPS result.

The generic structural route uses the same qualified host substepping and
composite-tangent policy:

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_library: build/mfront/src/libBehaviour.so
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    gps_composite_fd_step: 1.0e-6
```

It is currently qualified for the small-strain, `Implicit`,
`StandardElasticity`-compatible SRIX workflow. See
{doc}`../reference/numerics/mfront_structural_plane_stress` for the
formulation and its demonstrated scope.

## Checking what you actually ran

The solver diagnostics record the backend and, for GPS, the sub-stepping and
composite-FD counters. A run reporting `native_substep_points = 0` never needed
sub-stepping and would behave identically with the option off. A run reporting
`composite_fd_points` greater than zero used the repair on that many points.

If you have never used MFront and want a working configuration before reading
any of this, start at {doc}`run_316l_crystal_plasticity`.

For the details of `R`, the 316L parameters, orientations and per-system
outputs, see {doc}`use_srix_crystal_law`. The reference formulation is detailed
in
{doc}`../reference/numerics/three_dimensional_condensation`.
