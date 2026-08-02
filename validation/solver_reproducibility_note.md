# The archived `U.npy` hashes are not reproducibility guarantees

Date: 2026-08-02
Found while checking the non-regression of the constitutive-extension port on
branch `crystal-plasticity-boundaries`.

## What was measured

`results/constitutive-local-p0043-pad150` was recomputed twice from its own
archived manifest, with the same source, the same inputs and the same
configuration. The two runs **do not agree bit for bit**, and neither
reproduces the archived file:

| comparison | max absolute | relative to `\|U\|max` | bit-identical points |
|---|---:|---:|---:|
| run 1 against run 2, **identical source** | `5.55e-17` | `7.80e-16` | `45.9 %` |
| run 1 against the archive | `6.94e-17` | `9.75e-16` | `45.1 %` |
| run 2 against the archive | `6.94e-17` | `9.75e-16` | `46.0 %` |

`\|U\|max` is `7.12e-02`, so every deviation is a few units in the last place.

**The solver is not bit-reproducible.** The archived run uses
`mfront_threads: 8` and PyPardiso; both reduce in a thread-dependent order, and
that order is not fixed between processes.

## Why this matters beyond one port

`partitions/<id>/status.json` records a SHA-256 of `U.npy`, and
`replay_dic_observation` verifies it before observing a field. That check is
worth keeping, but it establishes **less than it appears to**:

- it proves the stored artefact has not been altered since it was written;
- it does **not** prove the computation reproduces, and it cannot, because it
  re-reads the same file rather than recomputing it.

Anyone reading `outputs.U` as a reproducibility guarantee is over-reading it.
It is an integrity hash.

## What does reproduce

The observation chain does, exactly. Re-observing the three archived coupled
runs at `ell = 58.88 um` through warp, DISFlow and the EVM operator on
2026-08-01 reproduced their archived `fem_observed_evm.npy` **byte for byte**.
So the non-determinism is in the mechanics, not downstream.

## Consequence for non-regression testing

A bit-for-bit criterion on `U.npy` is not achievable and must not be
registered as an acceptance condition; it would fail on an unmodified source.
The criterion used for the extension port is instead a **relative deviation
bounded at `1e-13`**, roughly a hundred times the measured run-to-run noise and
some ten orders of magnitude below any physical effect, together with the
explicit check that the deviation is not larger than that noise.

For the port itself the ratio was `1.25`: its deviation was indistinguishable
from re-running the same code.

## What would make it reproducible, if that is ever wanted

Not attempted here, and none of it is free:

- `mfront_threads: 1`, at a large cost in wall-clock time;
- a PARDISO configuration with a fixed reduction order, if the binding exposes
  one;
- pinning BLAS threading.

Whether bit-reproducibility is worth that cost is a separate decision. Nothing
in the current results depends on it: every campaign conclusion in this
repository rests on quantities whose measurement floors are between `1e-4` and
`1e-2`, twelve to fourteen orders of magnitude above this noise.
