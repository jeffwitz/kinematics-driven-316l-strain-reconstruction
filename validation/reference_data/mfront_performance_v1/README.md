# Constitutive backend performance benchmark v1

This directory preserves the first timing comparison between the current
NumPy constitutive implementation and the MGIS/MFront bridge.

## Workload

- 200,000 heterogeneous material points;
- 20 proportional strain increments;
- stress update, state update, and consistent tangent at every increment;
- 4,000,000 material-point updates per backend and repetition;
- two repetitions with reversed execution order;
- MFront measured both serially and with an explicit eight-thread MGIS pool.

The complete command took `1 min 03.24 s` and reached a peak resident set of
`402,892 KiB` (`393.45 MiB`).

## Results

| Backend | Median wall time | Range | Throughput |
|---|---:|---:|---:|
| Python/NumPy | `12.347 s` | `11.543–13.150 s` | `0.324 M updates/s` |
| MFront serial | `13.333 s` | `13.236–13.430 s` | `0.300 M updates/s` |
| MFront, 8 threads | `3.527 s` | `3.342–3.712 s` | `1.134 M updates/s` |

MFront serial is `1.080×` slower than Python. The eight-thread MFront backend
is `3.780×` faster than MFront serial and `3.500×` faster than Python for this
kernel workload.

Serial and parallel MFront results are identical to the last saved bit for
stress and PEEQ. The final Python/MFront stress difference is
`4.114e-6` in relative L2 norm.

## Interpretation limits

This is a constitutive-kernel benchmark, not an end-to-end finite-element
benchmark. It excludes element assembly, PyPardiso, Newton communication,
partition I/O, and DIC preparation.

The Python process used more CPU time than wall time (`27.55 s` versus
`12.35 s` median), so it must not be presented as a strictly single-core
baseline even though the standard thread environment variables were unset.
MFront thread counts are explicit.

Two repetitions are sufficient for an engineering orientation but not for a
publication-grade performance claim. The observed ranges show that frequency,
thermal state, and system activity still contribute visible variation.

Python uses the 1000-point tabular hardening law, whereas MFront uses the
regularised analytical form. Both compute the same J2 plane-stress workload,
and the final stress parity above confirms that the measured paths remain
numerically close.

## Artifacts

- `report.json`: raw timings, environment, hashes, summaries, ratios, and
  numerical checksums;
- `final_states.npz`: complete inputs and final stress/plastic states for both
  laws, plus 4096 tangent samples;
- `timings.png`: median time and throughput plot;
- `resource-usage.txt`: `/usr/bin/time -v` output for the complete command.

Reproduction:

```bash
source /home/jeff/.local/share/tfel/env/env.sh
bash scripts/build_mfront_behaviour.sh
.venv/bin/python scripts/benchmark_constitutive_backends.py \
  --output results/mfront-performance-$(date -u +%Y%m%dT%H%M%SZ) \
  --points 200000 \
  --increments 20 \
  --repeats 2 \
  --threads 8
```
