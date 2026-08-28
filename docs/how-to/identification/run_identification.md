# FEMU identification: current boundary-only status

**Mode:** how-to  
**Domain:** identification

There is currently no registered production boundary-only FEMU driver in this
checkout. Do not present the smoke script as one. A valid FEMU workflow must
declare the observable field, parameter perturbations, forward backend and
stopping criteria, impose measured kinematics only on the boundary, and
compare the predicted interior through the observation operator.

The available `scripts/srix_femu_smoke.py` is instead a full-field-Dirichlet
negative control. It is useful to reproduce the fact that prescribing the
interior displacement annihilates constitutive sensitivity, but it cannot
qualify parameter identification. Reproduce that control with
{doc}`../reproduce/reproduce_femu_full_field_dirichlet_negative_control`.

Definitions and required provenance are in
{doc}`../../reference/numerics/femu_sensitivity_and_svd`; interpretation is in
{doc}`../../explanation/identification/femu_identification`.
