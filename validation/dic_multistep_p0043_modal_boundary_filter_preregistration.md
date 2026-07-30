# P43 modal filtering of the measured boundary history — preregistration

Date: 2026-07-30
Written before the filtered history is produced.

## Rationale

`dic_boundary_loading_subspace_p0043_results.md` established that the P43
boundary motion is essentially low-dimensional: mode 1 carries `99.9118 %` of
the displacement energy at temporal roughness `0.0023`, modes 1 to 3 carry
`99.999 %` with roughness at most `0.109`, and from mode 13 onward the
roughness sits at `1`, the value of a temporally white series. Per-state
measurement noise is `0.0511 px`.

Imposing the raw field node by node at `1.84 µm` spacing asserts spatial
information the instrument does not provide: the chain's MTF-50 is near
`49 px`. Truncating the boundary data to its resolved modes removes content
that is at or below the noise, without touching the solver: the constraint
stays exact Dirichlet, the conditioning is unchanged, reactions keep their
meaning, and no archived result is invalidated.

## Filter definition

Applied to the boundary ring only. Interior values of the history array are
untouched; the solver reads only the boundary.

1. form the deviation from the straight endpoint ramp,
   `d_k = u_k - (k / N) * u_N`, so that `d_0 = 0` and `d_N = 0` **exactly**;
2. SVD of the `(N + 1) x n_boundary_dof` deviation matrix;
3. truncate to `r = 3` modes;
4. **pin the ends**: subtract from the reconstruction the field linear in the
   state index that makes its first and last rows exactly zero. A rank
   truncation does not preserve a zero row, so the two pinned states must be
   restored explicitly. A linear-in-time correction injects no new temporal
   structure and the two end rows cancel exactly in IEEE arithmetic;
5. add the ramp back.

Origin and endpoint are therefore preserved bit-for-bit, with
no discontinuity introduced at the last increment. The truncation acts on the
departure from proportional loading, which is where the temporal noise lives.

`r = 3` is fixed in advance. The registered mode-selection rule of the stage-0
campaign, temporal roughness below `0.5`, is evaluated on the deviation modes
and **reported as a check**, not used to re-tune `r` after the fact.

## Registered acceptance criteria

1. **The filter must remove noise, not signal.** The RMS amplitude of the
   removed content, over the boundary, must not exceed the measured per-state
   noise `0.0511 px`. If it exceeds it, the filter is removing resolved signal
   and `r = 3` is withdrawn.
2. **Origin and endpoint bit-identical** to the input history.
3. **Interior untouched**, bit-identical outside the boundary ring.
4. The retained energy fraction of the deviation is reported.

## Registered mechanical comparison

The filtered history is then run through the same workflow and compared with
the unfiltered measured run, reusing the tooling already registered:

- core PEEQ, via `compare-path-dependence`, against the unfiltered measured run;
- DIC agreement, via the symmetric observation replay on both profiles, with
  the same significance margins as
  `dic_multistep_p0043_observed_path_comparison_preregistration.md`.

### Registered expectations

- **Solver**: the filtered history should converge at least as easily. Fewer
  Newton iterations or cutbacks would be consistent with less noise being
  injected; more would be surprising and must be reported.
- **PEEQ**: a difference is expected but should be **much smaller** than the
  `15.82 %` measured between the measured and proportional paths, because the
  filter removes content at the noise level whereas the two paths differ
  structurally. A filtered-versus-unfiltered PEEQ difference **larger** than
  `15.82 %` would mean the filter is not a small perturbation and would put
  criterion 1 in doubt regardless of its amplitude test.
- **DIC agreement**: expected indistinguishable at the registered margins,
  since the previous campaign showed this observable is insensitive to changes
  of this size. Recorded in advance so that a null result is not read as
  failure.

## Claim boundary

The filter is justified by the measured noise floor and the measured spatial
resolution of the chain. It is **not** justified by whether it improves
convergence or agreement. If it were tuned on those, the result would be
fabricated; `r` is fixed here and is not revisited on the basis of the outcome.

## Deliverable

`validation/dic_multistep_p0043_modal_boundary_filter_results.md` and
`reference_data/dic_multistep_history_p0043_modal3_v1/`.
