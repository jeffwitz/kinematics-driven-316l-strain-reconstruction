# Information geometry of SRIX-REGM and FEMU — M8 twin

The three Jacobians were computed at the registered SRIX preset in the same
log-parameter coordinates and with the same affine-preserving DIC transfer:

1. `REGM_exact`: exact mechanical history replayed by REGM;
2. `REGM_observed`: transferred history replayed by REGM;
3. `FEMU_observed`: complete forward FEMU perturbations scored after the DIC
   transfer at the eight macro endpoints.

The FEMU Jacobian required eight perturbed forward solves plus the truth solve.
No P43 calculation was launched.

## Singular spectra

| Jacobian | normalized singular values | numerical rank | condition number |
| --- | --- | ---: | ---: |
| REGM exact | `1.000, 0.422, 0.0324, 4.65e-5` | 4 | `2.15e4` |
| REGM observed | `1.000, 0.337, 0.0178, 1.27e-5` | 4 | `7.90e4` |
| FEMU observed | `1.000, 0.542, 0.407, 0.0679` | 4 | `14.7` |

The noise-free observed FEMU twin therefore carries four numerically visible
directions, although the last one is weaker. The problem is not intrinsically
only two-dimensional at this resolution. REGM, even with the exact latent
history, compresses the third and fourth directions much more strongly.

## Parameter correlations

The FEMU Gauss–Newton correlation has strong positive correlations among the
hardening parameters: `rho(R,Q)=0.887`, `rho(R,b)=0.908` and
`rho(Q,b)=0.933`. This is a real local identifiability warning, but not a proof
that `Q` and `b` are unidentifiable: the fourth FEMU singular value remains
`6.8 %` of the leading one.

REGM exact has an almost perfectly correlated `Q/b` pair (`rho=1.000`) and a
condition number four orders of magnitude larger than FEMU. The transferred
REGM geometry is even more ill-conditioned.

## Sensitivity subspaces

Full-rank four-dimensional subspaces have trivial principal angles, so the
report compares the leading one-, two- and three-dimensional subspaces. The
leading one-dimensional angles are:

| pair | angle, rank 1 |
| --- | ---: |
| REGM exact / REGM observed | `33.0 deg` |
| REGM exact / FEMU observed | `68.4 deg` |
| REGM observed / FEMU observed | `42.7 deg` |

The leading two-dimensional REGM-exact/FEMU angle is `67.2 deg`, whereas the
leading two-dimensional REGM-exact/REGM-observed angle is only `0.81 deg`.
Thus the observation transfer changes the REGM geometry, but a more important
finding is that REGM exact is already not aligned with the true observed FEMU
information geometry.

## Loading history

The smallest cumulative normalized singular value of FEMU reaches about
`0.038` after two macro endpoints and `0.068` at the final endpoint. REGM exact
remains below `5e-5` at the final endpoint, and REGM observed below `2e-5`.
The direct FEMU experiment therefore reveals parameter directions much earlier
and more strongly than the equilibrium-gap surrogate.

## Decision

This is a decisive refinement of the previous diagnosis. The observation
operator is not the only limitation: even exact-space REGM does not reproduce
the local FEMU Fisher geometry. REGM remains useful for a coarse exact-twin
ranking, but it cannot currently be treated as an information-preserving
surrogate for four-parameter SRIX identification.

The correct next step is not to invert the DIC transfer. It is to either
reformulate REGM so that its residual has the same local sensitivity geometry as
FEMU, or restrict identification to directions demonstrably shared by FEMU and
REGM. Both options require another twin gate before P43.

Primary machine-readable artefacts and figures are in
`validation/reference_data/srix_regm_information_geometry_v1/`.
