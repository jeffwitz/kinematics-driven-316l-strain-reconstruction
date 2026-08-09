# Plane stress from a three-dimensional law

**Category: Reference.**

A plane-stress solver needs a three-component response, and a crystal law is
written in six. The missing three components are not a modelling choice: the
sheet has free surfaces normal to `z`, so the transverse **stresses** vanish and
the transverse **strains** are unknowns. Closing that gap is what this page is
about, and there are two ways to do it in this repository. They compute the same
thing and they are interchangeable — the qualification measures their agreement
at `1e-11` — but they are not interchangeable in what they let you build on.

## The condition, once

Partition the engineering strains and stresses:

$$
\varepsilon_a=[\varepsilon_{11},\varepsilon_{22},\gamma_{12}],\qquad
\varepsilon_b=[\varepsilon_{33},\gamma_{13},\gamma_{23}],\qquad
\sigma_b=[\sigma_{33},\sigma_{13},\sigma_{23}].
$$

Plane stress is $\sigma_b(\varepsilon_a,\varepsilon_b)=0$ — three equations for
the three unknowns $\varepsilon_b$, at every material point, at every step. It
is a **structural** condition: the free surface is normal to `z` in the
**global** frame, not in the crystal frame, so for an oriented grain the three
equations mix all six components of the material-frame stress.

Once it is satisfied, the tangent the two-dimensional solver needs is the Schur
complement

$$
C^{PS}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba},
$$

which is the derivative at **constrained** transverse stress, not the in-plane
block of the 3D tangent. Linear systems are solved; matrices are never
explicitly inverted.

## Route A — the closure outside, in Python

`constitutive_backend="mfront-3d-condensed-plane-stress"`

The bridge owns $\varepsilon_b$ and iterates on it. At each trial it hands the
law a **complete** six-component strain, integrates it to convergence, reads
$\sigma_b$ and corrects:

$$
C_{bb}\,\Delta\varepsilon_b=-\sigma_b .
$$

Every trial restarts from the same committed material state, so each outer
iterate is a genuine converged constitutive state. The bridge rotates the strain
into the crystal frame and the stress back out, monitors the conditioning of
$C_{bb}$, limits the correction, freezes points that have converged, and stops
on a tolerance scaled by the local stress.

**Its advantage is that it works with any three-dimensional law.** The law is
called through the standard MGIS interface with a full gradient and is never
modified — no extra state variables, no extra residual rows, no knowledge that
plane stress exists. Swap `mfront_behaviour_id` and the closure is unchanged.
That is why this route is the reference: a new constitutive model becomes
usable in plane stress the day it compiles, with nothing to write and nothing
to re-qualify on the closure side.

**What it costs.** The closure multiplies the constitutive calls — measured at
about `6.6` full integrations per material point per global Newton iteration on
P43 — and the orchestration lives in Python, about 21 % of the material time on
that case. And the closure cannot travel: it is Python, so it only exists inside
this repository.

```python
SolverConfig(
    constitutive_backend="mfront-3d-condensed-plane-stress",
    mfront_behaviour_id="fcc_forest_rubin_srix",   # any registered 3D law
    mfront_library="build/mfront/src/libBehaviour.so",
    constitutive_options={
        "crystal_orientation": {"mode": "ebsd", "euler_bunge_deg": angles},
    },
)
```

## Route B — the closure inside the UMAT

`constitutive_backend="mfront-native-generalised-plane-stress"`

The law carries the closure itself. `Fcc316LForestRubinSrixGps` has the same
constitutive body as `Fcc316LForestRubinSrix`, under the same standard
`@ModellingHypothesis Tridimensional`, and differs in how its residual is
written: the elastic residual is assembled in the **global** frame, its three
in-plane rows hold the kinematics, and its three transverse rows are free to
hold $\sigma_{g,b}/G_{\text{ref}}=0$. The rotation reaches the law as nine
per-point material properties `Q11..Q33`. There is no outer loop at all — one
local Newton returns a converged plane-stress state.

**Its advantage is that the closure travels with the behaviour.** Any
finite-element code able to call the MFront/MGIS GPS behaviour can impose the
generalised plane-stress closure without reimplementing that local closure.
However, the presently qualified robust integration policy is still host-side:
selective sub-stepping and the composite FD tangent live in this repository's
GPS adapter. A different host calling only the behaviour does not automatically
obtain that policy and may therefore have a different global-Newton cost or
robustness envelope.

**Sub-stepping, and why it needs a tangent repair.** The joint local Newton
refuses the full increment at a few deeply plastic points -- two out of four
hundred on the small P43 window, `0.071 %` of the batch at M100 -- and only
those points are integrated in sub-steps. A sub-stepped point then returns the
tangent of its **last** sub-step, which is not the derivative of the composite
path it actually followed, and the global Newton pays for it: 85 iterations
against the reference's 57 at M100.

`gps_composite_fd_tangent` rebuilds the tangent of exactly those points by
finite differences along the composite trajectory. It is what makes this route
usable, and it is not optional in practice:

| P43 M100 EBSD, 8 increments, 4 threads | time | Newton |
|---|---:|---:|
| condensed reference | `62.38 s` | 57 |
| GPS, tangent as the DSL returns it | `74.05 s` | 85 |
| **GPS + composite FD** | **`58.38 s`** | **58** |

The finite difference touched `192` points and `1152` trajectories for `2.15 s`
of its own cost, and the converged residual is `5.3e-09`.

The benchmark below is from one qualified campaign and is not a universal
speed guarantee. Hardware, thread placement, BLAS settings, compiler versions
and the selected loading path can change the wall time:

| P43 M100 EBSD, 8 increments | Newton | elapsed |
|---|---:|---:|
| 3D + Python condensation | 57 | approximately `62.4 s` |
| GPS + composite FD | 58 | approximately `58.4 s` |

The diagnostic comparison also explains why the composite tangent is part of
the qualified GPS route. Without it, the same GPS run takes 85 Newton
iterations; with it, it takes 58. A wrapper that replaces one constitutive
increment by a composition of sub-steps computes

$$
D\Phi_{\mathrm{last}}
\neq
D(\Phi_n\circ\cdots\circ\Phi_1),
$$

in general. The tangent returned by the last sub-step is therefore not the
tangent of the composed path seen by the global Newton. The sparse composite
FD correction supplies that missing derivative only where sub-stepping was
used.

**What it costs.** The law has to be written for it, once per behaviour: a
crystal model that has not been given the closure rows cannot use this route.
And a host code owes three conventions, which the bridge here implements and
another host would have to reproduce:

1. **supply `Q11..Q33`** as per-point material properties and pass the gradient
   **unrotated** — the law rotates it internally, so MGIS's own rotation must
   not be used as well;
2. **rotate the returned stress** and elastic strain back to the global frame,
   the law returning material-frame quantities;
3. **post-multiply the consistent tangent by the in-plane projector**
   $P=\mathrm{diag}(1,1,0,1,0,0)$ before taking its in-plane block, because
   $\partial f_{el}/\partial\varepsilon^{\text{imposed}}=-P$ and the DSL
   assumes $-I$.

The transverse total strain comes back as the auxiliary state variables `ezz`,
`eyz`, `exz`.

```python
SolverConfig(
    constitutive_backend="mfront-native-generalised-plane-stress",
    mfront_behaviour_id="fcc_forest_rubin_srix",   # routed to ..._gps
    mfront_library="build/mfront/src/libBehaviour.so",
    constitutive_options={
        "crystal_orientation": {"mode": "ebsd", "euler_bunge_deg": angles},
    },
)
```

`mfront_behaviour_id` names the **parent** law; the factory selects the
`_gps` variant. Naming `fcc_forest_rubin_srix_gps` directly works too.

## Choosing

| | Route A, Python | Route B, UMAT |
|---|---|---|
| works with any 3D law | **yes** | no, one law per behaviour |
| usable from another FEM code | no | **yes** |
| constitutive calls per point and global iteration | `6.6` | `2.0` |
| needs `gps_composite_fd_tangent` | — | yes, in practice |
| P43 20x20 material time against route A | — | `1.4 – 1.5x` faster |
| P43 100x100, total time | `62.38 s` | **`58.38 s`** |

The two agree to `1e-11` at a material point and to `1e-8` on P43
displacements, so the choice is about what you need to build, not about which
answer you get.

Route A remains the **numerical reference**: it accepts any 3D behaviour, and
every route-B result is measured against it. Route B with the composite tangent
is now the **faster** of the two on the qualified SRIX/EBSD case and is what
{doc}`../../how-to/choose_mfront_backend` recommends there — closure to
`2e-14` MPa, finite-difference tangent to `1.2e-07`, agreement with route A to
`1e-11`. Without the composite tangent route B is slower than route A at M100,
so the two are not interchangeable on performance alone.

Both build and run on **unmodified TFEL/MFront 5.1.0**. An earlier attempt to add
a `GeneralisedPlaneStress` modelling hypothesis to the generator was abandoned:
route B needs no patched toolchain, which is the reason it was preferred. See
`docs/explanation/spectral_mechanics/srix_monolithic_plane_stress_architecture.md`
for that decision and
`docs/explanation/spectral_mechanics/umat_gps_handoff_2026-08-07.md` for the
qualification and the open performance questions.
