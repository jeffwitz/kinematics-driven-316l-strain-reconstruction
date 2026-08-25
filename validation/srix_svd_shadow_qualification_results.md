# SVD-shadow qualification — P43 M20 smoke

## Scope

This is a local qualification against the archived two-point global-FD smoke,
not the completed Sobol campaign. The global Sobol run remains incomplete
because one admissible sample caused an MFront integration failure.

The test uses the archived aggregate SVD basis, rank 7, and 14 persistent
constitutive shadow histories. Each projected shadow column is compared with
the exact archived oracle

```text
J_z^FD = J_eta^FD V_7
```

No identification is performed.

## Direct-shadow versus FD

For the prior and one off-nominal smoke point, the relative column errors and
cosines are:

```text
prior errors:  0.00299, 0.00243, 0.00282, 0.00790,
               0.00330, 0.00202, 0.00432
prior cosines: 0.9999955, 0.9999970, 0.9999963, 0.9999689,
               0.9999946, 0.9999980, 0.9999907

Sobol errors:  0.00588, 0.00459, 0.00330, 0.01191,
               0.00267, 0.00327, 0.00720
Sobol cosines: 0.9999829, 0.9999896, 0.9999946, 0.9999293,
               0.9999964, 0.9999947, 0.9999743
```

The fourth retained mode is the weakest of this rank-7 set and is below the
3% relaxed target at both points. The other modes are at or close to the 1%
target; the off-nominal point is the more demanding case.

## Step-size check

Repeating both points with half the adaptive shadow steps gives a maximum
column variation of 0.90% and a minimum column cosine of 0.999959. This is
consistent with a stable finite-difference shadow derivative rather than a
noise-dominated result.

The nominal adaptive steps were:

```text
0.00100, 0.002373, 0.002987, 0.00500, 0.00500, 0.00500, 0.00500
```

## Cost

Each point used one nonlinear base forward, 14 local shadow histories, and 224
linear tangent solves. The shadow phase took approximately 80–84 seconds per
point in this M20 smoke.

## Decision

```text
projected_svd_shadow_smoke_passed = true
svd_shadow_qualified_global = false
experimental_identification_authorized = false
```

The next required gate is to complete or repair the Sobol global campaign,
then recompute the rank-7 subspace angles and repeat this comparison at one
nominal and one off-nominal point from that qualified basis.
