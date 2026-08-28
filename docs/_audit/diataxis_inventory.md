# Diátaxis inventory (baseline and phase 2)

This baseline records the routing decision for every documentation page at the
start of the refactor.  The machine-readable source is
[`../diataxis_manifest.yml`](../diataxis_manifest.yml); the structure checker
fails if a later page is not covered exactly once.

| Current path family | Apparent content | Target mode | Target navigation | Action |
|---|---|---|---|---|
| `tutorials/*.md` | guided first runs | tutorial | Tutorials | KEEP |
| `how-to/*.md` | one operational task | how-to | How-to | KEEP / SPLIT when hybrid |
| `reference/*.md` | contracts and definitions | reference | Reference | KEEP / SPLIT when hybrid |
| `reference/numerics/*.md` | numerical contracts | reference | Reference / Numerics | KEEP |
| `explanation/*.md` | scientific argument | explanation | Explanation | KEEP / SPLIT when hybrid |
| `explanation/spectral_mechanics/*.md` | spectral science | explanation | Explanation / Spectral | KEEP |
| `evidence/index.md` | evidence routing | portal | Scientific evidence | ADD |
| `maintainers/index.md` | developer routing | portal | Maintainers | ADD |
| `docs/*.md` | legacy monolithic pages | explanation/reference | legacy only | REDIRECT / SPLIT |
| `adr/*.md`, `agent/*.md` | internal decisions and handover | reference | Maintainers | KEEP / INTERNAL |

The legacy root pages are intentionally not promoted as primary navigation.
Their useful equations, results and limitations are being routed to the
quadrant pages; they remain available while individual splits are completed.

Phase 2 added canonical, domain-specific pages for reconstruction, DIC/EBSD,
constitutive laws, structural plane stress, native SRIX, spectral mechanics,
identification and evidence. The checker currently audits 211 documentation
pages. The remaining legacy families are explicitly historical and are the
next migration candidates; they are not hidden under a broad current-domain
glob.
