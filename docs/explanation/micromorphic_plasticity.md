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

MFront is never committed inside this fixed point. Every evaluation starts
from the same accepted material state. The final trial is committed exactly
once, and only after the global mechanical Newton iteration converges. A
cutback reverts the MFront state, displacement trial, and micromorphic field
together.

## Lightweight constitutive hot path

The fixed point needs only the updated local PEEQ until \(\chi\) converges.
The production native-plane-stress adapter therefore separates three
operations:

1. intermediate fixed-point integrations use MFront without a tangent and
   expose only PEEQ;
2. one integration with the consistent tangent produces the in-plane stress
   and tangent needed by the mechanical Newton assembly;
3. full 3D stress and strain tensors are reconstructed once, after final FEM
   convergence, immediately before the accepted state is committed.

This split does not alter the constitutive equations, fixed-point tolerance,
Newton algorithm, increment sequence, sparse assembly, or PyPardiso solve. The
committed MGIS state remains transactional: intermediate calls never
accumulate plastic strain. Reusable Kelvin-strain, Gauss-point PEEQ, and
nonlocal-field workspaces remove repeated allocations from the same loops. For
the proportional DIC loading used here, the elastic predictor direction is
also assembled once and scaled at each requested load factor.

The detailed timers distinguish MFront calls with and without a tangent,
Kelvin conversion, final tensor completion, internal forces, element matrices,
sparse assembly, free-system extraction, and PyPardiso. The condensed 3D
adapter retains its existing local transverse Newton and tangent requirement:
the lightweight no-tangent path primarily benefits the native `PlaneStress`
backend used for the micromorphic campaign.

The benchmark evidence and numerical equivalence checks are preserved in
`validation/performance/nonlocal_hot_path_optimization.json` and summarised in
{doc}`../reference/results`.

## Fixed CSR and explicit PARDISO phases

The subsequent linear-solver optimization keeps the free--free stiffness
graph fixed for the whole calculation. Element contributions are mapped once
to CSR `data` positions; every Newton assembly updates only the values of the
same matrix object. The former COO-to-CSR reconstruction and repeated
free-system extraction are no longer present in the hot loop.

PyPardiso 0.4.7 normally combines symbolic analysis and numerical
factorization when matrix values change. The adapter now drives MKL PARDISO
explicitly:

1. phase 11 analyses the fixed graph once;
2. phase 22 factorizes every new tangent numerically;
3. phase 33 solves the current right-hand side.

The first validated version deliberately retained `mtype=11`. Matrix type is
now selected from the material capability. The current J2 behaviours use an
upper-triangular fixed CSR graph and symmetric positive-definite `mtype=2`.
Unclassified behaviours keep the complete graph and `mtype=11`.

No Newton modification, fixed-point acceleration, or coupled micromorphic
tangent is introduced. Phase counts, matrix type, and timings are saved in
`SolverDiagnostics`.

The P187 complete-solver gate records one phase 11 and 139 phase 22/33 pairs.
It reduces sparse assembly by 73.4%, free-system extraction by 99.4%, and
PARDISO time by 48.3%. See {doc}`../reference/results`.

The subsequent `mtype=2` gate reduces PARDISO time from `33.825 s` to
`20.964 s`, including a `46.2%` reduction in numerical factorization. Total
wall time falls from `244.67 s` to `227.34 s`, and peak RSS falls by `8.7%`.
The maximum observed relative tangent asymmetry is `6.46e-16`.

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
model passes held-out spatial tests. Similarly, \(H_\chi\) was explored only
within the pre-registered P154 sweep. Because that ROI proved too homogeneous
for identification, no value was frozen. The next calibration target is P43,
selected after a morphology scan and explicit visual confirmation of two
diagonal deformation bands.

## P43 validation outcome

The fixed-length P43 sweep tested \(\alpha=0,1,2,4\) on a `660 x 610`
padded domain. All 20 increments converged without cutback for every
candidate. From local to alpha 4, Pearson correlation increases from 0.3791
to 0.5036 and relative L2 decreases from 0.9516 to 0.4341. The PEEQ maximum
decreases from 0.0642 to 0.0116 while its mean changes only from 0.00308 to
0.00279: coupling primarily redistributes plasticity and suppresses narrow
peaks.

The response is not sufficient to identify a unique \(H_\chi\). Alpha 2 has
the highest top-10% IoU, whereas alpha 4 has the best correlation, L2 and
absolute DIC-q90 IoU. Both are non-dominated and alpha 4 remains the largest
tested value. No larger alpha is added after inspecting the result. Moreover,
the length was held fixed, so this campaign validates the mechanism but does
not constitute a joint identification of \(H_\chi\) and \(\ell\). See
`validation/nonlocal_p0043_validation_results.md` for the complete numerical
record and {doc}`p43_coupled_results` for the detailed figure-by-figure
interpretation and temporary conclusions.

## P154 validation outcome

The 20-increment, 128-pixel-padding sweep tested
\(\alpha=H_\chi/H_\mathrm{ref}\) equal to 0.5, 1, and 2. All candidates
converged without cutback and progressively reduced the raw FEM--DIC field
error. The best tested point, \(\alpha=2\), increased Pearson correlation by
0.164, reduced relative L2 by 42.17%, and increased top-10% IoU by 0.033.
It passed seven of eight pre-registered criteria.

The failed criterion is physically informative: the area above the absolute
DIC-q90 threshold remained 21.85%, above the registered 20% maximum. The
model therefore demonstrably diffuses the local plastic zone, but the current
parameter sweep does not fully reproduce its measured width. The conclusion
is **partially supported**, and no \(H_\chi\) is frozen for transfer. See
`validation/nonlocal_p154_validation_results.md` for the complete numerical
record.
