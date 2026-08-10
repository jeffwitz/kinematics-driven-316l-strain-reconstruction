# SRIX parameter sets

Registered, immutable parameter sets for `Fcc316LForestRubinSrix`, selected by
identifier. The authority is `fem_inhouse.core.srix_parameters`; this page is
the reader's copy.

**No registered set identifies 316L for the SRIX law.** Every one of them mixes
adopted, transposed or exploratory values, a test asserts that none claims
otherwise, and the reason is spelled out under [Statuses](#statuses) below.

## Selecting one

Every parameter is an MFront `@Parameter`, so a set is applied at run time
through MGIS and nothing is recompiled:

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  constitutive_options:
    parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3
```

Individual values may be overridden on top of a set:

```yaml
constitutive_options:
  parameter_set: 316l_srix_transposed_from_nasri2018_rate_1e-3
  parameters:
    R_mpa: 2.0
    tau0_mpa: 38.33
```

Accepted names are `R_mpa`, `tau0_mpa`, `Q_mpa`, `b`, `C_mpa`, `d`, and the
cubic stiffnesses `C11_mpa`, `C12_mpa`, `C44_mpa` — the last three as a group or
not at all, since two of the three describe no material. Anything else is
refused before the first solve, as is an unknown `parameter_set`.

Selecting nothing applies the registered historical parameter set. Reproducing
a documented result also requires its backend, numerical options and software
version; the parameter default alone is not a complete campaign specification.

## The registered sets

Stresses in MPa; `b` and `d` are dimensionless. `O_R` is the overstress ratio
$\frac{\sqrt 6}{8}\frac{R}{\tau_0}$, a dimensionless reading of how rounded the
elastic-plastic transition is at first yield.

| identifier | `C11 / C12 / C44` | `R` | `τ0` | `Q` | `b` | `C` | `d` | `O_R` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `316l_srix_transposed_from_nasri2018_rate_1e-3` | 197000 / 125000 / 122000 | 18.7819 | 40 | 10 | 3 | 40000 | 1500 | 0.1438 |
| `316l_srix_updated_elasticity_prior` | 218300 / 144800 / 125400 | 18.7819 | 38.33 | 10 | 3 | 40000 | 1500 | 0.1500 |
| `316l_srix_exploratory_r1` | 197000 / 125000 / 122000 | 1 | 40 | 10 | 3 | 40000 | 1500 | 0.0077 |
| `316l_srix_exploratory_r2` | 197000 / 125000 / 122000 | 2 | 40 | 10 | 3 | 40000 | 1500 | 0.0153 |
| `316l_srix_exploratory_r4` | 197000 / 125000 / 122000 | 4 | 40 | 10 | 3 | 40000 | 1500 | 0.0306 |
| `316l_srix_exploratory_r8` | 197000 / 125000 / 122000 | 8 | 40 | 10 | 3 | 40000 | 1500 | 0.0612 |
| `316l_srix_exploratory_r18p7819` | 197000 / 125000 / 122000 | 18.7819 | 40 | 10 | 3 | 40000 | 1500 | 0.1438 |

All seven share the interaction matrix
$(1.0,\ 1.0,\ 0.6,\ 1.8,\ 1.6,\ 12.3,\ 1.6)$; see
{doc}`fcc_interaction_matrix_mapping` for what each slot means. All are stated
at `293.15 K`.

**The historical set.** The one every archived SRIX result was computed with.
Its name says what it is: transposed from the Méric-Cailletaud set of Nasri and
others (2018) at a reference rate of `1e-3 s⁻¹`. It is not "the 316L SRIX
model".

**The updated-elasticity set.** New single-crystal stiffnesses and a `38.33 MPa`
threshold, with `(Q, b, C, d)` and the interaction matrix explicitly inherited
from the historical set and provisional. Moving `τ0` without re-identifying the
hardening moves the whole curve, so this set is for sensitivity work.

**The exploratory `R` series.** Everything except `R` matches the historical
set, so a difference between two of them is attributable to `R` alone. The
historical value is included as its own point rather than compared from outside
the sweep.

## Statuses

Provenance is recorded **per group of parameters**, not per set, because a set
routinely mixes an adopted elasticity, a prior threshold and a modulus
transposed from a different flow rule. Presenting that mixture as "the 316L
parameters" is the error the scheme exists to prevent.

| status | meaning |
|---|---|
| `identified` | fitted to measurements of this material with this law |
| `literature_measurement` | measured and published for this material |
| `literature_prior` | a published value for a comparable material, adopted as a starting point |
| `analytical_transposition` | computed from a parameter of a *different* law through a closed-form correspondence |
| `exploratory` | chosen to span a range; carries no claim about any material |

Only the first two support a statement about 316L. Today:

| identifier | overstress modulus | everything else |
|---|---|---|
| `316l_srix_transposed_from_nasri2018_rate_1e-3` | `analytical_transposition` | `literature_prior` |
| `316l_srix_updated_elasticity_prior` | `analytical_transposition` | `literature_prior` |
| the five `316l_srix_exploratory_r*` | `exploratory` | `literature_prior` |

An inline `parameters` override demotes its own group to `exploratory`: nothing
in the software knows where an inline number came from, so it cannot support a
claim either.

## What a run records

Each SRIX solve writes the parameter half of its provenance from
`SrixParameterSet.provenance_record()`: the identifier, every value with its
unit, the origin and status of each group, the interaction matrix and its
convention, the temperature, the reference strain rate, how `R` was obtained,
and the list of MFront parameter names actually set. The run half — MFront file
hash, TFEL and MGIS versions, git commit — is added at solve time, since none of
it is a property of the set.

## The route to an identified set

Registered in `validation/srix_316l_calibration_preregistration.md`. In short:
elasticity first, then `τ0` from the slip threshold, then `R` from the width of
the elastic-plastic transition — directly, with no Méric-Cailletaud law in the
chain — then `(Q, b)` and the interaction matrix from several monotonic
orientations, then `(C, d)` from reversed loading, and finally validation on
orientations and paths not used during identification. Identifying everything at
once on one macroscopic curve is forbidden there.
