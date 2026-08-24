# E-SRIX-FEMU-PATH-002S — preregistration

Date: 2026-08-24

## Objective

Run one final nested loading-path level after the negative `PATH-002R` gate.
The purpose is to decide whether the remaining sensitivity drift is a short
convergence tail or requires a targeted numerical diagnosis.

## Fixed protocol

- source L0/L1/L2: `srix_femu_path_convergence_v3`;
- construct L3 by mandatory midpoint insertion of every L2 interval;
- retain all mandatory nodes and add only strict local repairs when required;
- one base forward and one direct sensitivity Jacobian per level;
- same corrected Dirichlet contract, oracle, MFront, plane-stress and scoring
  support as `PATH-002R`;
- no new finite-difference forwards, identification, or P43.

## Primary comparison

Judge only L2→L3 with the unchanged `PATH-002R` thresholds:

- observed forward relative change `< 5e-3`;
- columns 1--3 relative change `< 2e-2`;
- columns 1--3 cosine `> 0.999`;
- rank-3 principal angle `< 2 degrees`;
- first three normalized singular values change by `< 5%`.

The fourth mode remains diagnostic only. Record its magnitude and its
alignment with the `Q-b` contrast. This extension is the final refinement
attempt: regardless of its outcome, do not launch L4 or further blind
refinements. A failed L2→L3 gate triggers a targeted sensitivity/continuation
diagnosis rather than another refinement level.

## Artefact

Write `validation/reference_data/srix_femu_path_convergence_v4/` and a result
note recording the decision. Identification and P43 remain unauthorized until
the result is interpreted.
