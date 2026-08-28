# Phase 4 navigation and scientific backlog

## Legacy pages removed from current menus

| Former root page | Canonical destination |
|---|---|
| `reference/model_contract.md` | `reference/scientific/model_contract.md` |
| `reference/tensor_conventions.md` | `reference/scientific/tensor_conventions.md` |
| `reference/dic_axis_conventions.md` | `reference/data/dic_axis_conventions.md` |
| `reference/srix_parameter_sets.md` | `reference/scientific/srix_parameter_sets.md` |
| `reference/fcc_interaction_matrix_mapping.md` | `reference/scientific/fcc_interaction_matrix_mapping.md` |
| `reference/experimental_data_inventory.md` | `reference/data/experimental_data_inventory.md` |
| `reference/input_contract.md` | `reference/data/input_contract.md` |
| `reference/output_contract.md` | `reference/data/output_contract.md` |
| `reference/api.md`, `cli.md`, `configuration.md`, `extension_interfaces.md` | `reference/software/` equivalents |
| root data/extension how-to pages | `how-to/data/` and `how-to/extend/` equivalents |

The old pages remain historical for reproducibility, but no current `toctree`
points to them.  The structure checker now rejects any current `toctree`
target whose manifest status is not `current`, or whose navigation is
`legacy`.

## Routed subjects and blockers

The machine-readable backlog is in
`scientific_coverage.yml`; the corresponding paths and human-readable summary
are in `scientific_coverage_matrix.md`.  Current blockers are deliberately
conservative:

- DIC, J2, plane stress and REGM need a content review before completion.
- EBSD needs a dedicated explanatory page for orientation/registration.
- SRIX and native SRIX need an evidence-route review.
- Méric, spectral/FFTW, FEMU/SVD and reduced integration need more specific
  reference contracts.

No subject was promoted to `complete` by declaration alone.  The checker still
requires current manifest entries, mode agreement and reachability from the
canonical portal for that status.
