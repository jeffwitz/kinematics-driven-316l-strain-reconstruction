# Run the micromorphic identification campaign

**Category: How-to.** This campaign is **specified but not yet run**. It is
written so it can be handed over and executed on a cluster.

Read the rationale below before launching. Two of the choices look like
inefficiencies and are not; changing them silently invalidates the result.

The registered protocol is
`validation/micromorphic_symmetric_identification_preregistration.md`. This page
is the operating manual for it.

## Why this campaign exists

The local reconstruction localises over too much area: against the DIC q90
threshold, which selects 10 % of the field by construction, the local model is
**over-active by 61 %**. Micromorphic coupling is the only lever tested that
measurably fixes this — at $\alpha = 4$ the over-activity falls to **+2.6 %**,
and relative $L_2$ improves by 9.6 times the noise margin of that metric.

But the coupling modulus $H_\chi$ and the spatial length $\ell$ have never been
identified **separately**. Every archived campaign varies $\alpha$, which scales
$H_\chi$, at one fixed $\ell = 58.88\ \mu$m.

The question this campaign answers is therefore not *which $\alpha$ is best*.
It is:

> Are $H_\chi$ and $\ell$ separately identifiable from this observation, or is
> the objective degenerate along $A_\chi = H_\chi \ell^2$?

If only the product is identifiable, no separate spatial length may be claimed,
and that is a publishable negative result.

## Two traps, both easy to fall into

:::{danger}
**Do not reuse `configs/joint_nonlocal_identification_p0043.yaml` as it
stands.** It already contains a 21 by 21 grid and looks ready to run. Its
objective applies **no observation operator to the FEM side**:
`identification/observation.py` declares `spatial_filter: Literal["none"]` and
offers only grid reduction and core masking.

That is the raw FEM against image-observed DIC comparison. Lot V3 showed it
changes amplitude, morphology **and the ranking of coupling candidates**, which
is why identification was suspended in the first place. Reusing it would repeat
the exact error the suspension existed to prevent.
:::

:::{warning}
**Do not enable the low-fidelity `spatial_reduction: 2` tier.** It is the
obvious way to make the campaign cheaper and it is incompatible with the
objective.

The symmetric operator warps a displacement field onto the reference image and
re-observes it through DISFlow. That is only defined when **one element is one
pixel**. At reduction 2 the field is at half resolution and the operator is a
different operator, so the scores are not comparable with anything archived.
:::

Temporal reduction is the only reduction allowed, and it is held at **20
increments**. The measured discretisation sensitivity is `0.20 %` on core PEEQ
between 20 and 40 increments, which bounds the error of using 20. It says
nothing about 10, so do not use 10.

## The grid

$\ell \in \{20, 30, 40, 50, 58.88\}\ \mu$m and $\alpha \in \{1, 2, 3, 4, 6\}$,
25 points.

The coupling modulus passed to the solver is $\alpha H_\mathrm{ref}$ with
$H_\mathrm{ref} = 5168.147582748343$ MPa, read from
`results/constitutive-local-p0043-pad150/HREF.json`. Use the full precision:

| $\alpha$ | `--nonlocal-coupling-modulus-mpa` |
|---:|---|
| 1 | `5168.147582748343` |
| 2 | `10336.295165496686` |
| 3 | `15504.442748245029` |
| 4 | `20672.590330993372` |
| 6 | `31008.885496490058` |

**Three points already exist** and must not be recomputed:
$(\alpha, \ell) = (1, 58.88)$, $(2, 58.88)$ and $(4, 58.88)$, in
`results/constitutive-nonlocal-p0043-pad150-a100`, `-a200` and `-a400`. They are
already observed symmetrically in
`validation/reference_data/dic_symmetric_observation_p0043_v1/`.

That leaves **22 runs**. They are tabulated ready for a job array in
`campaigns/mm_id_points.tsv`, one point per line as
`alpha_tag`, `ell_tag`, `ell_um`, `hchi_mpa`, tab separated:

```text
a001	ell20	20	5168.147582748343
a001	ell30	30	5168.147582748343
...
a006	ell58p88	58.88	31008.885496490058
```

## One grid point

```bash
fem-inhouse --verbose partition \
  --input data/processed/case_study \
  --output "results/mm-id-p0043-a${ALPHA_TAG}-ell${ELL_TAG}" \
  --parts-x 10 --parts-y 10 --padding 150 \
  --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --nonlocal-plasticity \
  --nonlocal-length-um "${ELL_UM}" \
  --nonlocal-coupling-modulus-mpa "${HCHI_MPA}" \
  --nonlocal-relaxation 0.5 \
  --nonlocal-tolerance 1e-6 \
  --nonlocal-max-iterations 15 \
  --partition-id 43 \
  --mfront-threads 8
```

Only `--nonlocal-length-um` and `--nonlocal-coupling-modulus-mpa` change between
points. Everything else is fixed, and fixed deliberately: the archived campaigns
used these settings, so the three reused points stay comparable.

## Resources per job

Measured on an archived coupled run of comparable type, scaled to the P43
solve support of `660 x 610` elements:

| Resource | Value |
|---|---|
| wall clock | `27` to `36` min, allow **1 h** |
| cores | 8 requested, about `3.2` effective |
| peak RSS | about `10` GB, request **16 GB** |
| disk per point | about `150` MB |

Total for 22 points: about **11 to 12 h serial**, or one array pass if the
cluster runs them concurrently.

## Cluster job array

The 22 points are independent, so this is embarrassingly parallel. A SLURM
skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=mm-id-p0043
#SBATCH --array=0-21
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mm-id-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# TFEL/MFront environment. env.sh is not compatible with `set -u` when these
# are unset, so initialise them first.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${TFEL_PREFIX}/share/tfel/env/env.sh"
export PYTHONPATH="${TFEL_PREFIX}/lib/python3.12/site-packages:${PYTHONPATH}"
export MFRONT_BEHAVIOUR_LIBRARY="${PWD}/build/mfront/src/libBehaviour.so"
export PATH="${PWD}/.venv/bin:${PATH}"

readarray -t POINTS < campaigns/mm_id_points.tsv
IFS=$'\t' read -r ALPHA_TAG ELL_TAG ELL_UM HCHI_MPA <<< "${POINTS[$SLURM_ARRAY_TASK_ID]}"

fem-inhouse --verbose partition \
  --input data/processed/case_study \
  --output "results/mm-id-p0043-a${ALPHA_TAG}-ell${ELL_TAG}" \
  --parts-x 10 --parts-y 10 --padding 150 \
  --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --nonlocal-plasticity \
  --nonlocal-length-um "${ELL_UM}" \
  --nonlocal-coupling-modulus-mpa "${HCHI_MPA}" \
  --nonlocal-relaxation 0.5 --nonlocal-tolerance 1e-6 --nonlocal-max-iterations 15 \
  --partition-id 43 --mfront-threads 8
```

Before submitting, on the cluster:

```bash
bash scripts/build_mfront_behaviour.sh
test -f build/mfront/src/libBehaviour.so
.venv/bin/python -c 'import tfel, mgis.behaviour; print(tfel.getTFELVersion())'
fem-inhouse backend    # must report PyPardiso/MKL, not SuperLU
```

If `backend` reports SuperLU the run will be far slower and single-threaded.
Fix the MKL installation before launching the array.

## Then the observation step, which can run anywhere

The symmetric observation is **cheap**, seconds per point on the `661 x 611`
support, and it needs the raw DIC images. Those need not be on the cluster:
copy back the `U.npy` of each point and run this locally.

```bash
for RUN in results/mm-id-p0043-*; do
  fem-inhouse replay-dic-observation \
    --campaign "${RUN}" \
    --prepared-case data/processed/case_study \
    --reference-image "${DIC_IMAGES}/000294.tif" \
    --partition-id 43 \
    --profile legacy_script_2021 \
    --output "validation/reference_data/mm_id_observed/$(basename "${RUN}")_legacy"
done
```

Repeat with `--profile declared_medium_v4` for the registered sensitivity.
`legacy_script_2021` is primary **by provenance**, not because it scores better;
see {doc}`../explanation/current_evidence`.

## Checking each run before using it

A point is usable only if all of these hold:

- `status.json` reports `complete: true`;
- the run recorded **zero cutbacks**, matching the archived coupled runs;
- the nonlocal fixed point converged at every increment;
- fields are finite.

A point that needed cutbacks is not automatically wrong, but it is not
comparable with the others and must be flagged in the report rather than
silently averaged in.

## Reading the result

:::{important}
**Do not rank the points by a single score.** The archived symmetric result
already shows the ranking depends on the objective: $\alpha = 4$ wins on
relative $L_2$ and Pearson, $\alpha = 1$ on q90 IoU, and the local model on
top-10 % IoU. Collapsing four disagreeing metrics into one number would
manufacture an identification that the data does not support.
:::

Produce instead:

1. the four metric surfaces over the grid;
2. the Pareto-non-dominated set across the four;
3. the per-objective optimum, reported separately.

Then answer the actual question by fitting the orientation of the objective
valley in $(\log H_\chi, \log \ell)$ and comparing it with the constant-$A_\chi$
direction:

| Outcome | Conclusion |
|---|---|
| valley not aligned with constant $A_\chi$, on most metrics | the two parameters are separately identifiable |
| valley aligned within the metric margins | **only the product is identifiable**; claim no separate $\ell$ |
| Pareto set spans more than half the grid | identification fails at this resolution; report and stop |

The significance margins are fixed in advance and are **not** to be rechosen:
relative $L_2$ `0.0202`, Pearson `0.0185`, top-10 % IoU `0.0189`, absolute-q90
IoU `0.0217`. They come from the measured DIC-noise sensitivity intervals.

The third outcome is considered likely. It is a real result about the
observable, not a failed campaign, and it must be reported as such.

## What this campaign cannot deliver

- **no transferable material internal length.** That needs unchanged-parameter
  transfer to another ROI, another observation resolution and ideally another
  test;
- **no prediction claim**: the boundary conditions remain measured throughout;
- all runs use the proportional loading path, so the `16 %` path systematic
  measured on core PEEQ applies. It is uniform across the grid and so should not
  move the ranking, but that is an assumption to state, not a verified fact.

Cite these limits with any number taken from this campaign.
