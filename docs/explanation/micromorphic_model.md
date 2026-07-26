# A constitutive internal-length model

**Category: Explanation.** How can spatial interaction enter the constitutive
model without turning the reconstruction into post-processing?

## Energy and fields

The coupled micromorphic model augments the local small-strain J2 energy:

$$
\psi=
\psi_e+\psi_L(p)
+\frac{H_\chi}{2}(p-\chi)^2
+\frac{H_\chi\ell^2}{2}\lvert\nabla\chi\rvert^2.
$$

Here:

- $p$ is the local accumulated equivalent plastic strain;
- $\chi$ is its spatially regularized micromorphic representation;
- $H_\chi$ is the coupling modulus in MPa;
- $\ell$ is the spatial length in millimetres.

Stationarity with respect to $\chi$ gives

$$
\chi-\ell^2\Delta\chi=p,
\qquad \nabla\chi\cdot n=0.
$$

The yield-surface radius becomes

$$
R(p,\chi)=R_{\mathrm{local}}(p)+H_\chi(p-\chi).
$$

Where a local peak has $p>\chi$, coupling raises resistance. In the
neighbourhood, where $p<\chi$, it lowers resistance. The mechanism therefore
redistributes plastic activity rather than merely averaging an output.

```{graphviz}
digraph micromorphic {
  rankdir=LR;
  node [shape=box, style="rounded,filled"];
  p [label="local p\nGauss points", fillcolor="#fff0d4", color="#d97706"];
  chi [label="nonlocal chi\nspatial field", fillcolor="#d9edf7", color="#2980b9"];
  response [label="yield correction\nHchi (p - chi)", fillcolor="#dff0d8", color="#2e8b57"];
  p -> chi [label="Helmholtz over ell"];
  chi -> response;
  p -> response;
  response -> p [label="constitutive feedback"];
}
```

## Strength and length have different roles

Campaigns use the normalized coupling

$$
\alpha=\frac{H_\chi}{H_{\mathrm{ref}}},
$$

where $H_{\mathrm{ref}}$ is derived from the local hardening response. The
combination

$$
A_\chi=H_\chi\ell^2
$$

is important for identifiability.

For a spatial Fourier mode of wavenumber $k$, the micromorphic correction
scales as

$$
H_\chi\frac{\ell^2k^2}{1+\ell^2k^2}.
$$

$H_\chi$ controls how strongly the local/nonlocal mismatch affects plasticity.
$\ell$ controls which spatial wavelengths are affected. When $\ell k\ll1$,
the response depends mainly on $A_\chi k^2$, making the two parameters
difficult to separate.

## Discrete interpretation and boundaries

The current research model averages Gauss-point $p$ to each structured element,
solves the Helmholtz equation on the padded element grid, and returns the
element value of $\chi$ to its Gauss points. Zero micromorphic flux is imposed
on the artificial solved boundary. Padding keeps that boundary away from the
retained scientific core.

This page describes the physics. The staggered fixed point, transactional
MFront state, DCT implementation and sparse mechanical solver are documented
in {doc}`../reference/numerics/nonlocal_fixed_point`,
{doc}`../reference/numerics/mfront_transaction` and
{doc}`../reference/numerics/sparse_solver`.

## Conclusion

> The coupling modulus controls the intensity of spatial feedback, while the
> spatial length controls the affected scales. Their product can dominate in
> an asymptotic regime, so both cannot be identified from a single global
> error metric.

The discriminating strategy is the subject of
{doc}`parameter_identification`.
