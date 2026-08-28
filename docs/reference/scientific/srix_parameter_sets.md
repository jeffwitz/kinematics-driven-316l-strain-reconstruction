# SRIX parameter-set contract

**Mode:** reference  
**Domain:** crystal-plasticity

SRIX parameters are selected from the immutable registered sets in
`fem_inhouse.core.srix_parameters`.  MFront and the native backend consume the
same object; no backend-specific copy of `C11`, `C12`, `C44`, `R`, `tau0`, `Q`,
`b`, `C`, `d` or the FCC interaction matrix is allowed.

All runs must record the set identifier, units, provenance/status of each
parameter group, interaction-matrix convention, temperature, reference rate,
backend and git/software versions.  Inline overrides are exploratory unless
their provenance is separately declared.  A registered set is not an
identified material merely because its name contains `316l`.

The authoritative values and status table are maintained by the Python
parameter module; this page defines the selection and provenance contract.
See {doc}`fcc_interaction_matrix_mapping` for the seven MFront interaction
slots.
