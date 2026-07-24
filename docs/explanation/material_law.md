# J2 plasticity and Ludwik hardening

## Supported constitutive model

The model is small-strain, isotropic, associative J2 plasticity under plane
stress. Its elastic constants are homogeneous:

| Parameter | Value |
|---|---:|
| Young’s modulus $E$ | `205000 MPa` |
| Poisson’s ratio $\nu$ | `0.30` |

Its hardening parameters are:

| Parameter | Spatial character | Nominal value |
|---|---|---:|
| initial yield stress $\sigma_y$ | element-wise map | data-driven |
| coefficient $K$ | element-wise map | scaled from DIC identification |
| exponent $n$ | homogeneous | `0.245` |

## Elastic response

With engineering shear $\gamma_{12}=2\epsilon_{12}$, the in-plane
plane-stress elastic operator is

$$
\begin{bmatrix}
\sigma_{11}\\ \sigma_{22}\\ \sigma_{12}
\end{bmatrix}
=
\frac{E}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0\\
\nu & 1 & 0\\
0 & 0 & (1-\nu)/2
\end{bmatrix}
\begin{bmatrix}
\epsilon^e_{11}\\ \epsilon^e_{22}\\ \gamma^e_{12}
\end{bmatrix}.
$$

MFront enforces $\sigma_{33}=0$ through its `PlaneStress` modelling
hypothesis and maintains the internal axial strain required by that constraint.

Plane stress does not imply $\epsilon_{33}=0$. After convergence, total,
elastic, and plastic transverse strains are retained in the public complete
tensors. Their derivation and the native MFront state variables are documented
in {doc}`plane_stress_tensors`.

## Yield criterion and flow

Let

$$
q = \sqrt{\frac{3}{2}\,\mathbf{s}:\mathbf{s}}
$$

be the von Mises equivalent stress computed from deviatoric stress
$\mathbf{s}$. The yield function is

$$
f(\boldsymbol{\sigma},p)=q-R(p)\leq 0,
$$

where $p$ is PEEQ and $R(p)$ is the current yield-surface radius. Associated
flow uses the normal to the von Mises surface, so plastic flow is isochoric in
the three-dimensional constitutive model.

Consequently,

$$
\epsilon^p_{11}+\epsilon^p_{22}+\epsilon^p_{33}=0.
$$

This relation completes the plastic strain tensor in the Python backend and
provides an independent check of the native MFront result. It does not change
the return mapping or the accumulated PEEQ update.

## Analytical Ludwik law

Away from the origin:

$$
R(p)=\sigma_y+K\,p^n.
$$

Because $0<n<1$, the analytical derivative $Kn\,p^{n-1}$ diverges as
$p\rightarrow0^+$. The implementation uses a finite first segment with
$p_0=10^{-6}$:

$$
R(p)=\sigma_y+K
\begin{cases}
p\,p_0^{n-1}, & 0\le p\le p_0,\\
p^n, & p>p_0.
\end{cases}
$$

The radius is continuous at $p_0$. MFront uses the linear-segment derivative
below $p_0$ and the analytical derivative above it. This regularization
removes the singular origin while preserving the power law over every
scientifically relevant strain.

```{image} ../_static/ludwik_hardening.*
:alt: Analytical unbounded Ludwik hardening compared with the historical capped table.
:width: 92%
:align: center
```

## Why the 1000-point law is historical only

The previous Abaqus-oriented path sampled the law at 1000 plastic-strain values
up to $p=0.2$, then retained the final tabulated stress beyond that point.
It therefore introduced:

- interpolation error inside the calibrated interval;
- an artificial perfectly plastic plateau after $p=0.2$;
- array construction and storage with no added scientific content.

The nominal MFront law evaluates the formula directly and has no upper PEEQ
cap. The Python table remains available only to reproduce or compare with the
historical Abaqus representation.

On the validated article corner partition, maximum PEEQ is `0.06496`; the old
cap was not reached in that particular calculation. Removing it is still
important because the constitutive definition should not change silently when
a later loading path exceeds the historical table.

## Local heterogeneity

Each integration point receives the local element properties:

```text
InitialYieldStress = yield_stress_mpa[i, j]
HardeningCoefficient = hardening_coefficient_mpa[i, j]
HardeningExponent = 0.245
```

The same element values are used at all four Gauss points of that pixel-sized
CPS4 element. Heterogeneity therefore enters through element maps, not through
multiple phases or crystal orientations.

## Consistent tangent

Newton’s method needs

$$
\mathbf{C}_\mathrm{alg}
=\frac{\partial\boldsymbol{\sigma}_{n+1}}
       {\partial\boldsymbol{\epsilon}_{n+1}},
$$

the derivative of the integrated constitutive update, not merely the elastic
matrix. MFront returns this consistent operator through MGIS. The FE code
converts it from Kelvin notation to its engineering-shear convention and
assembles it directly.

The tangent affects convergence speed; it does not define a different final
equilibrium solution when both implementations converge to the same tolerance.
