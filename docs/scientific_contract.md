# Scientific contract

The exact manuscript used to define this contract is identified by size and
SHA-256 in `ArticleSource/manifest.json`. The manifest deliberately records a
null DOI because no DOI is present in the supplied manuscript; publication
metadata must not be invented.

Status: implementation baseline, 2026-07-24.

This document makes the conventions used by the code explicit. It is based on
`ArticleSource/ArticleAdil.pdf`; remaining ambiguities must be resolved against
the original Abaqus input generator and ODB extraction scripts.

## Scientific objective

The software reconstructs a mechanically admissible kinematic field from DIC
measurements. It aims to reproduce the spatial organization of microscale
strain localization. Pixel-wise elastoplastic parameters are effective
reconstruction descriptors conditional on the loading path, DIC resolution,
filtering, and constitutive assumptions. They are not intrinsic grain
properties.

Direct FE stress and strain-based reconstructed stress answer different
questions and must never be substituted for one another.

## Array and coordinate convention

All structured nodal and element arrays use:

- axis 0: physical `x`, transverse direction;
- axis 1: physical `y`, tensile/loading direction;
- `u_x`: displacement along `x`;
- `u_y`: displacement along `y`.

The historical DIC names map as follows in the cropped mesh frame:

- `V` is `u_x`;
- `U` is `u_y`.

For an array `a`, `numpy.gradient(a, dx, dy)` therefore returns derivatives
along `(x, y)` in that order.

## Units

The internal unit system is:

- length and displacement: millimetres;
- stress and elastic moduli: megapascals;
- strain: dimensionless;
- reaction force: newtons only after the plane-stress section thickness has
  been specified consistently with Abaqus.

The nominal DIC-to-mesh spacing is `0.001 mm × 1.84 = 0.00184 mm`.

The physical specimen thickness reported by the article is `2 mm`; this fact
justifies the plane-stress assumption but does not, by itself, establish the
section thickness used by the Abaqus CPS4 model. That value remains to be
verified from the original input file.

## Small-strain components

The tensorial in-plane strain components are:

```text
epsilon_xx = d(u_x) / dx
epsilon_yy = d(u_y) / dy
epsilon_xy = 0.5 * (d(u_x) / dy + d(u_y) / dx)
```

Engineering shear is:

```text
gamma_xy = 2 * epsilon_xy
```

FE element vectors use `[epsilon_xx, epsilon_yy, gamma_xy]`. DIC invariant
calculations use tensorial `epsilon_xy`; conversion must therefore be explicit.

## Plane-stress equivalent strain

Under plane stress:

```text
epsilon_zz = -nu / (1 - nu) * (epsilon_xx + epsilon_yy)
```

The equivalent strain is evaluated from the three-dimensional deviatoric
invariant:

```text
epsilon_vM = sqrt(2/3 * epsilon_dev : epsilon_dev)
```

with `epsilon_xz = epsilon_yz = 0`. The implementation computes this invariant
directly rather than relying on a separately simplified formula.

## Constitutive convention

The local hardening law is:

```text
sigma_eq = sigma_y + K * epsilon_p**n
```

Nominal article values are `E = 205000 MPa`, `nu = 0.30`, `n = 0.245`,
`sigma_y = 124 MPa`, and `K = 380 MPa`. Local maps may replace `sigma_y` and
`K`; `E`, `nu`, and `n` remain homogeneous for the supported case.

The article states that Abaqus used a table over `0 <= epsilon_p <= 0.2`, with
1000 points and a minimum positive increment of `1e-6`. This is retained as a
historical reproduction mode, not as the production constitutive contract.

The default MFront law regularises only the first interval
`0 <= epsilon_p <= 1e-6`, then evaluates the analytical power law without an
upper PEEQ cap. This avoids introducing a non-physical plateau outside the
tabulated Abaqus range.

## Four macroscopic curves

The workflow must keep these four curves distinct:

1. measured macroscopic stress-strain curve;
2. DIC strain-based reconstructed equivalent stress;
3. FE strain-based reconstructed equivalent stress;
4. direct FE equivalent stress.

Curves 2 and 3 reapply the scalar constitutive law to spatially averaged
equivalent strain. They are consistency checks, not independent stress
predictions.

Curve 4 is computed by first spatially averaging `S11`, `S22`, and `S12`, then
applying the plane-stress von Mises formula. The direct curve may remain below
the measured response; this is a scientific result and must not be hidden by
switching to curve 3 after yielding.

## Shared DIC/FE comparison

For scientific field comparison, both DIC and FE strains are derived from
their nodal displacement fields with the same NumPy gradient convention. This
avoids mixing DIC differentiation with Abaqus visualization extrapolation.
