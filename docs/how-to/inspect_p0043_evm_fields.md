# Inspect the P43 observed-EVM fields visually

**Category: How-to.** Produces one image per computation so the `(ell, alpha)`
matrix can be swept by eye in a file browser, and states what is and is not
legitimate to read off them.

Nothing here recomputes anything: the script loads archived `.npy` fields and
draws them.

## Producing the images

```bash
.venv/bin/python scripts/plot_p0043_matrix_evm.py --profile legacy_script_2021
.venv/bin/python scripts/plot_p0043_matrix_evm.py --profile declared_medium_v4
```

Output, one directory per DISFlow profile:

```
validation/reference_data/p0043_small_parameter_matrix_v1/
  evm_fields_legacy_script_2021/
  evm_fields_declared_medium_v4/
```

Seventeen fields per profile, each written as **PNG and SVG**: the DIC, the
fourteen converged matrix points, and the two negative controls. The two
non-converged points, `(alpha=4, ell=20)` and `(alpha=4, ell=40)`, produce no
file and the script says so.

## What makes them comparable

Two properties are deliberate, and both matter for sweeping.

**One colour scale for every image**, taken from the **1st and 99th percentiles
of the DIC**: `1.596e-03` to `8.934e-03`. Percentiles rather than min-max
because the EVM has a long upper tail — the DIC reaches `1.31e-02`, half again
the upper limit, on a handful of pixels. Scaling to that maximum would push
almost every pixel into the bottom of the colour map and flatten exactly the
band structure one is trying to see. The consequence to keep in mind: **the
brightest pixels are clipped**, in the DIC and in the candidates alike, so
saturation means "at or above `8.9e-03`", not "equal".

**A fixed layout.** The field occupies the same figure rectangle in every file,
independent of the title length. Flipping through a directory therefore moves
nothing under the eye except the data. This is why the layout is set by explicit
rectangles rather than an automatic one.

## File naming

Names are built so that a plain alphabetical listing is already a sweep:

```
evm_DIC.png                     the measurement, first
evm_a0p5_ell020p00.png          then the matrix, by alpha then ell
evm_a0p5_ell040p00.png
...
evm_a4p0_ell090p00.png
evm_zz_control_homogeneous.png  then the controls, last
evm_zz_control_translated.png
```

`ell` is zero-padded to two integer digits so `020` sorts before `090`. To sweep
along `ell` at fixed `alpha`, read consecutive files; to sweep along `alpha` at
fixed `ell`, sort by the second field.

Each image carries its `q95` in the subtitle, which is the quantity
`D_amplitude` is built on, so the numeric trend is visible without leaving the
picture.

## What the eye can legitimately read

**The texture difference is real but is not model error.** The DIC is finely
speckled at every scale; every FEM field is smooth. The symmetric replay applies
DISFlow's spatial transfer to the FEM displacement but adds **no
speckle-decorrelation noise**, so the observed FEM is smoother than the DIC by
construction. Fine-grain roughness must not be counted against a candidate.

**The band-scale geometry is comparable**, and this is where the eye is useful:
where the high-strain regions run, whether they are one connected object or
several, and how wide they are.

**Saturation is informative in one direction only.** A candidate showing more
saturated area than the DIC is genuinely over-predicting the upper tail, since
both are clipped at the same value. A candidate showing less is
under-predicting. But two saturated regions cannot be compared with each other.

## What the images confirm, quantitatively

The visual impression matches the measured indicators, and the numbers are
worth having in hand while looking:

| observation in the images | measured counterpart |
|---|---|
| every candidate shows one merged bright region, the DIC shows two | object count 1 against 2 for 13 of 14 points |
| candidate bands look about twice too wide | minor-axis ratio `1.88` to `2.08` |
| brightness falls steadily as `alpha` rises | high-pass energy ratio `R` from `0.97` to `0.18` |
| the homogeneous control is uniformly dim | `R = 0.02`, zero objects above the Otsu threshold |
| the translated control is bright but in the wrong place | shape defect `0.794`, the worst of any field |

The single visual anomaly worth knowing about: `(alpha=2, ell=20)` is the only
candidate that segments into two objects, but its second object is a `620 px`
fragment against the DIC's `8 340 px` band. It looks like a match to an
automatic object count and is not one by eye.

## What no image licenses

A picture cannot establish that one parameterisation is better than another
here. The registered campaign concluded **case B**, a zone of ten
indistinguishable points, and the images are consistent with that: across
`alpha <= 1` the fields are visibly similar to one another. Use them to
understand *how* the candidates differ and to check that a number means what it
seems to, not to pick a point.
