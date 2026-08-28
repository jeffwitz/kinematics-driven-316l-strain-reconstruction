# Repository storage policy

This repository has a fast, effective MFront reference backend, but MFront is
not a practical route to a future GPU implementation.  The native Python SRIX
backend is therefore being built as a portable CPU/GPU architecture: NumPy is
the qualified CPU implementation today, with Numba kernels for small local
systems and a future CuPy backend as the GPU target.  That work produces many
intermediate fields and benchmark snapshots.  They are evidence during
development, not automatically durable repository artefacts.

## Why this policy exists

The repository contains a large amount of numerical evidence generated while
qualifying the FEM, MFront, and native SRIX backends.  The scientific result
is normally the report, metrics, configuration, and provenance—not a second
copy of a full field produced by a benchmark that was already shown to be
identical.

The repository therefore separates three kinds of material:

1. **Versioned evidence**: source code, tests, scripts, configurations,
   manifests, JSON/CSV metrics, and Markdown reports.
2. **Golden fields**: a small number of explicitly selected fields used for
   regression tests and backend equivalence checks.
3. **Generated or historical fields**: reproducible outputs kept outside the
   normal Git history and referenced by their report/provenance when needed.

The governing rule is **golden only** for LFS:

> Git stores the scientific proof and the recipe to reproduce it.  It does not
> store every field produced while developing or benchmarking the solver.

## Cleanup performed on 2026-08-28

The canonical `agent/plastic-observability` history was rewritten from the
backup bundle below.  The following generated payloads were removed from Git
history while their reports and scripts were retained:

- binary outputs below `validation/_generated/` (`*.npz`, `*.npy`, model
  checkpoints and weights);
- redundant P43 native SRIX M20/M100 and scaling field snapshots;
- large intermediate `dic_multistep_history_p0043_*` NumPy histories.

The full pre-cleanup Git history is recoverable from:

```text
/home/jeff/CNRS/Theses/Adil/repo-backups/p0043-cleanup-20260828/
fem_inhouse-before-cleanup.bundle
```

Bundle SHA-256:

```text
4b7501d953db7aa1e3121269340239d16ab8c116c04e4312b8d543f7fa75a833
```

The cleanup does not remove raw experimental inputs, source code, tests,
MFront references, article data, or quantitative reports.

## Rules for future campaigns

- Keep reports, metrics, manifests, command lines, and provenance in Git.
- Do not commit a full field for every timing or implementation variant.
- Add a full field only when it is explicitly designated as a golden reference
  or is required by a non-reproducible external deliverable.
- Put large generated fields and checkpoints in external archival storage,
  record a SHA-256 and source commit in a manifest, and keep the report in Git.
- The `.gitignore` rules for `validation/_generated/` are preventive; an
  explicit exception is required before adding a generated binary reference.
- Git LFS is reserved for intentionally retained golden arrays, not as a way
  to archive every intermediate output.

### Where a new artefact belongs

| Artefact | Location | Versioning |
|---|---|---|
| source, tests, scripts, parameters | repository | Git |
| report, metrics, manifest, command line | `validation/reports/`, `validation/metrics/`, or `validation/manifests/` | Git |
| small deterministic test fixture | tests or `validation/golden/` | Git; LFS only when explicitly justified |
| full simulation field, DIC history, checkpoint, benchmark snapshot | external archive | Git stores a manifest and SHA-256 |
| generated plots or `_generated` output | local `validation/_generated/` | ignored by Git |

Before committing a binary or a large result, ask whether it is a stable
golden reference needed by a regression test.  By default, calculation output
is generated locally and ignored.  Do not add a new LFS rule under
`validation/reference_data/`; use an explicit path under `validation/golden/`
only after documenting why the object is durable and necessary.

### Archiving a large campaign

Keep a compressed archive outside Git/Git LFS, for example
`p0043_srix_m100_20260828.tar.zst`, and commit a small manifest containing:

```json
{
  "archive": "p0043_srix_m100_20260828.tar.zst",
  "sha256": "...",
  "source_commit": "...",
  "command": "...",
  "description": "...",
  "status": "archived"
}
```

The archive may currently be institutional or local; the policy does not
prescribe a particular external service.

### Automated guard

Run `python scripts/check_repository_storage.py` before committing.  It fails
for a new Git-normal file larger than 20 MiB unless the path is explicitly
whitelisted (currently `validation/golden/**`).  LFS pointers themselves are
small and are not counted as payloads.  For a branch/PR comparison, pass its
base commit with `--base origin/main`.

## Important limitation

Rewriting Git history does not automatically delete orphaned objects already
stored in GitHub LFS.  LFS quota cleanup requires the archived inventory and,
if necessary, a GitHub support request or repository migration.
