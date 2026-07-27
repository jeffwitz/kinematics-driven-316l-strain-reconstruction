# P43 generalized section-equilibrium baseline preregistration

Date: 2026-07-27

## Question

Do the archived local and micromorphic P43 stress fields satisfy the
cross-section-integrated form of vertical equilibrium at the resolution of
their saved element-centred stresses?

This is a numerical self-consistency baseline. It is not a comparison with a
measured load and has no acceptance threshold in this first run.

## Frozen cases

| Label | Campaign |
|---|---|
| `local` | `results/constitutive-local-p0043-pad150` |
| `alpha_1` | `results/constitutive-nonlocal-p0043-pad150-a100` |
| `alpha_2` | `results/constitutive-nonlocal-p0043-pad150-a200` |
| `alpha_4` | `results/constitutive-nonlocal-p0043-pad150-a400` |

The analysis uses partition 43, the reported specimen thickness of 2 mm, and
the 1.84 µm element spacing declared by each immutable campaign manifest.
Both the complete padded solve domain and the retained core are reported.

## Frozen equation and discretisation

P43 is an interior partition with artificial displacement boundaries. The
naive condition

```text
N_y(y) = t integral sigma_yy(x,y) dx = constant
```

does not apply because shear traction can cross its lateral cuts. The
diagnostic therefore evaluates, between adjacent cell-centred sections,

```text
Delta N_y
+ Delta y * t * mean[sigma_12(x_R) - sigma_12(x_L)]
= residual.
```

`N_y` is integrated with the midpoint rule. Lateral boundary stresses are
approximated by the first and last saved cell-centred values. The residual is
therefore a post-processing diagnostic and is not expected to equal the
quadrature-level finite-element residual.

## Reported quantities

For each case and region:

- mean section force in N;
- relative standard deviation of section force;
- RMS interval balance residual in N;
- relative L2 interval balance residual;
- full profiles of section force, lateral shear flux and interval residual.

The simple force dispersion is descriptive only. Scientific interpretation
must use the generalized balance including lateral shear. No pass/fail
threshold will be introduced after observing this baseline.

## Claim boundary

The physical force cannot be validated because no synchronized load-cell
series and no confirmed full gauge width are currently available. The
reported thickness comes from the article rather than a traceable specimen
measurement. Absolute resultants must therefore not be presented as
experimental force validation.
