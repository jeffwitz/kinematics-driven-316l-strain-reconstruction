# Coupled micromorphic J2 plasticity

## Scientific role

The output-only Helmholtz diagnostic showed that broadening the FEM fields can
improve agreement with DIC. It does not change the mechanical solution. The
coupled model asks the next question:

> Can a spatial interaction in the evolution of plasticity broaden the raw
> mechanically equilibrated fields, without filtering the final EVM map?

The model is deliberately limited to the existing small-strain J2/Ludwik case
study. It adds one scalar micromorphic field, not a new mechanical displacement
and not a general nonlocal constitutive framework.

## Energy and yield radius

The augmented free-energy contribution is

$$
\psi =
\psi_e(\boldsymbol\varepsilon-\boldsymbol\varepsilon^p)
+\psi_L(p)
+\frac{H_\chi}{2}(p-\chi)^2
+\frac{H_\chi\ell^2}{2}\lVert\nabla\chi\rVert^2.
$$

Here \(p\) is the local accumulated equivalent plastic strain (`PEEQ`),
\(\chi\) is its micromorphic counterpart, \(H_\chi\) is a coupling modulus in
MPa, and \(\ell\) is a length in mm. Stationarity with respect to \(\chi\)
gives

$$
\chi-\ell^2\Delta\chi=p,
\qquad
\nabla\chi\cdot\mathbf n=0.
$$

The current regularized Ludwik radius becomes

$$
R(p,\chi)=\sigma_{y0}+K\,h(p)+H_\chi(p-\chi),
$$

and its local derivative at fixed \(\chi\) is

$$
\left.\frac{\partial R}{\partial p}\right|_\chi
=K\,h'(p)+H_\chi.
$$

This is not the empirical substitution \(K\chi^n\). At the centre of a sharp
peak, \(p>\chi\), so the coupling raises the local resistance. Around the
peak, \(p<\chi\), so it lowers it. This redistribution mechanism follows from
the declared energy.

## Discrete field

Plasticity remains stored at the four Gauss points of each CPS4 element. The
Helmholtz source is the element average

$$
p_e=\frac14\sum_{g=1}^{4}p_{e,g}.
$$

The existing element-centred DCT solver computes

$$
(I+\ell^2L_h)\boldsymbol\chi=\mathbf p_e
$$

on the complete padded structured grid. One \(\chi_e\) value is then supplied
to all four Gauss points of element \(e\). This avoids an additional Q4
projection and preserves exactly the same discrete Neumann operator that was
verified during the diagnostic stage.

The artificial micromorphic boundary is separated from the retained core by
padding. The P154 validation profile uses 128 pixels for
\(\ell=32\) pixels, hence a padding-to-length ratio of exactly four.

## Staggered constitutive solve inside Newton

For every global Newton trial strain, the code solves a local/nonlocal fixed
point:

1. set the MFront external state variable
   `NonlocalEquivalentPlasticStrain` to \(\chi^{(k)}\);
2. integrate every material point from the last committed state;
3. extract local `PEEQ` and average it per element;
4. solve the Helmholtz problem for \(\chi^\star\);
5. relax
   \(\chi^{(k+1)}=(1-\omega)\chi^{(k)}+\omega\chi^\star\);
6. repeat until the mixed relative residual is below tolerance;
7. reevaluate MFront once with the converged \(\chi\).

The convergence norm is

$$
\frac{\lVert\chi^{(k+1)}-\chi^{(k)}\rVert_2}
{\max(1,\lVert\chi^{(k+1)}\rVert_2,\lVert\chi^\star\rVert_2)}.
$$

The unit floor avoids a meaningless relative singularity when plasticity first
appears and both fields are close to zero. It is the same mixed scaling used
by the project's other constitutive invariants.

MFront is never committed inside this fixed point. Every evaluation starts
from the same accepted material state. The final trial is committed exactly
once, and only after the global mechanical Newton iteration converges. A
cutback reverts the MFront state, displacement trial, and micromorphic field
together.

## Tangent scope

The MFront consistent tangent contains the derivative of the local J2 update
at fixed \(\chi\), including \(H_\chi\) in
\(\partial R/\partial p\). The native plane-stress behaviour supplies its
condensed tangent directly. The 3D behaviour is reduced through the existing
local transverse-stress Newton solve and Schur complement for every current
\(\chi\).

This is a staggered tangent, not the full monolithic derivative of the
Helmholtz-coupled system. Convergence and cutback diagnostics must therefore
remain visible; the implementation does not claim monolithic quadratic
convergence.

The fixed point uses the mesh-independent mixed maximum norm

$$
\eta_\chi =
\frac{\lVert\chi^{k+1}-\chi^k\rVert_\infty}
{\max\left(1,\lVert\chi^{k+1}\rVert_\infty,
\lVert\chi^\star\rVert_\infty\right)}.
$$

The unit scale is the absolute branch for the dimensionless PEEQ field; the
state magnitude is the relative branch. A raw global \(L_2\) norm is not used
because it makes the same pointwise error harder to accept when the ROI gains
elements. The saved diagnostics identify this choice as
`mixed_relative_linf`.

## Two MFront behaviours

`PixelMicromorphicLudwikJ2Plasticity` uses the native `PlaneStress`
hypothesis. It is the production candidate for the current isotropic law.

`PixelMicromorphicLudwikJ2Plasticity3D` uses `Tridimensional` and is condensed
by the Python/MGIS adapter. It exists to verify that the architecture can later
accept a genuinely three-dimensional law without changing the global 2D FEM
kernel.

Both expose:

- material property `MicromorphicCouplingModulus`;
- external state variable `NonlocalEquivalentPlasticStrain`;
- internal `EquivalentPlasticStrain`, `ElasticStrain`, and
  `YieldSurfaceRadius`.

## What zero coupling means

When \(H_\chi=0\), the mechanical response is independent of \(\chi\). The
solver still computes the nonlocal output field, but the historical `U`, `S`,
`E`, `PE`, `PEEQ`, and `RF` solution follows the reference material law.
Complete-Newton regression tests compare this path with the local native
MFront path at \(10^{-10}\) relative tolerance.

## Interpretation limits

The coupled raw EVM field is the primary DIC comparison. Filtering that final
field is allowed only as a secondary diagnostic.

The initial \(\ell=58.88\,\mu\mathrm m\) comes from the P48 selection and P42
confirmation campaign. It remains a diagnostic candidate until the coupled
model passes held-out spatial tests. Similarly, \(H_\chi\) is selected only
within the pre-registered P154 sweep and must be frozen before transfer to P42
or P48.
