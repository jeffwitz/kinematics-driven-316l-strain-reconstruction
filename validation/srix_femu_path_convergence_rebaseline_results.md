# E-SRIX-FEMU-PATH-002R — results

Status: **negative primary gate** (2026-08-24)

The experiment starts from the corrected v17 common path and uses nested
mandatory midpoint insertion with strict local repairs. The implementation and
artefact are in
`validation/reference_data/srix_femu_path_convergence_v3/` (`dirty=false`,
run commit `0784b2b`).

## Levels

| level | mandatory path | actual steps | local repairs |
|---|---:|---:|---:|
| L0 | v17 | 94 | 0 |
| L1 | midpoint of L0 | 188 | 0 |
| L2 | midpoint of L1 | 392 | 16 |

The L2 repairs were local strict-convergence repairs; no mandatory midpoint was
removed. The observed forward changes are small:

| comparison | observed forward relative L2 |
|---|---:|
| L0 → L1 | `2.176e-4` |
| L1 → L2 | `9.514e-5` |

## Primary L1 → L2 gate

| quantity | result | criterion | status |
|---|---:|---:|---|
| `log(tau0)` column change | `3.67 %` | `< 2 %` | fail |
| `log(R)` column change | `4.20 %` | `< 2 %` | fail |
| `log(Q)` column change | `1.36 %` | `< 2 %` | pass |
| rank-3 maximum principal angle | `2.606°` | `< 2°` | fail |
| first-three singular-value changes | `0 %, 0.95 %, 3.11 %` | `< 5 %` | pass |

The primary path-convergence claim is therefore **false**. The forward field is
already very stable, but the sensitivity geometry has not reached the
pre-registered tolerance. No threshold was moved after computation.

The normalized spectra are:

```text
L0: (1, 0.180201, 0.040286, 6.318e-5)
L1: (1, 0.180726, 0.042223, 6.192e-5)
L2: (1, 0.182455, 0.043576, 6.234e-5)
```

The fourth mode remains extremely weak and is almost exactly the `Q-b`
contrast: its alignment with `(0, 0, 1, -1)/sqrt(2)` is `0.999987` at all
three levels. This is diagnostic evidence for a practically weak `Q/b`
direction, not a license to declare it non-identifiable before a further
refinement study.

Identification and P43 remain unauthorized. The next permissible step is a
targeted refinement/continuation study for the remaining sensitivity drift,
not parameter optimization.
