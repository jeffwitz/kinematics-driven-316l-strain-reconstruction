# How to choose an MFront backend

**Category: How-to.** Which `constitutive_backend` to put in your
configuration, and why. If you only want the answer for 316L crystal
plasticity on EBSD orientations, it is the first block below.

## The short answer

```yaml
solver:
  constitutive_backend: mfront-native-generalised-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  mfront_threads: 4
  constitutive_options:
    gps_composite_fd_tangent: true
    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3
    crystal_orientation:
      mode: ebsd
      euler_bunge_deg: [...]        # (nx, ny, 3), degrees
```

**Recommended for the qualified P43 SRIX/EBSD workflow.** Use
`mfront-3d-condensed-plane-stress` as the independent numerical reference, or
for a 3D behaviour that has no GPS variant.

## The four backends

| Value | What it is | Use it when |
|---|---|---|
| `mfront-native-plane-stress` (alias `mfront`) | the law itself is written under `@ModellingHypothesis PlaneStress` | your behaviour has a native plane-stress hypothesis — the J2 pixel laws do |
| `mfront-3d-condensed-plane-stress` | a 3D law, with the plane-stress condition closed by an outer Newton in Python | **any** 3D behaviour, including one written by someone else and never modified; also the independent reference |
| `mfront-native-generalised-plane-stress` | a 3D law whose own local Newton carries the closure (a "GPS" law) | the behaviour has a GPS variant, and you want speed or portability to another FEM code |
| `python` | the historical analytical/tabulated J2 implementation | regression only |

## Why there are two ways to do plane stress on a 3D law

A crystal law is written in six components; a plane-stress solver needs three.
The three transverse stresses must vanish and the three transverse strains are
unknowns. Closing that gap can happen **outside** the law or **inside** it, and
the trade is not about accuracy — the two agree to `1e-11` at a material point
— but about what each one lets you do. {doc}`../reference/numerics/three_dimensional_condensation`
derives both and gives the exact conventions.

**Condensed 3D — works with any law.** The bridge iterates the transverse
strain and hands the behaviour a complete six-component gradient. The law is
never modified and does not know plane stress exists, so a new constitutive
model becomes usable the day it compiles. It costs about `6.6` full
integrations per material point per global Newton iteration, and the closure
lives in Python, so it cannot leave this repository.

**Generalised plane stress — travels, and is faster here.** The closure is part
of the law, so any code able to call an MFront/MGIS behaviour gets plane stress
without writing a closure loop. It costs one law written for it per behaviour.
Today only `fcc_forest_rubin_srix` has such a variant
(`Fcc316LForestRubinSrixGps`, selected automatically).

## Why `gps_composite_fd_tangent: true`

The GPS local Newton refuses the full increment at a few deeply plastic points
— two out of four hundred on the small P43 window — so those points, and only
those, are integrated in sub-steps. A sub-stepped point then returns the
tangent of its **last sub-step**, which is not the derivative of the composite
path the point actually followed, and the global Newton pays for it.

`gps_composite_fd_tangent` rebuilds the tangent of exactly those points by
finite differences along the composite trajectory. On P43 M100 EBSD it concerns
`192` points and `1152` trajectories for `2.15 s` of its own cost, and it is
what brings the GPS Newton count back in line:

| P43 M100 EBSD, 8 increments, 4 threads | time | Newton |
|---|---:|---:|
| condensed reference | `62.38 s` | 57 |
| GPS, tangent as the DSL returns it | `74.05 s` | 85 |
| **GPS + composite FD** | **`58.38 s`** | **58** |

Leaving it off is not wrong, only slower. Turning it on for the condensed
backend is refused, so a configuration cannot carry the option without effect.

## Checking what you actually ran

The solver diagnostics record the backend and, for GPS, the sub-stepping and
composite-FD counters. A run that reports `native_substep_points = 0` never
needed sub-stepping and would behave identically with the option off.
