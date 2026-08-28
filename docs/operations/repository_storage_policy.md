# Repository storage policy

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

## Important limitation

Rewriting Git history does not automatically delete orphaned objects already
stored in GitHub LFS.  LFS quota cleanup requires the archived inventory and,
if necessary, a GitHub support request or repository migration.
