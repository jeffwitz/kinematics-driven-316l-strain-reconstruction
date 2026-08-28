# Temporal loading paths and reconstructed strain

## Two distinct paths

The reconstruction uses measured displacement information through a sequence
of load states. Two quantities must be distinguished:

1. the path used to drive the constitutive and FEM solve;
2. the path used to evaluate an observable such as an accumulated equivalent
   strain or a comparison with DIC.

The first path is a sequence of finite increments. The second may be a
post-processing functional of the accepted displacement and internal-state
history. They coincide only when the observable is defined directly from the
same incremental state variables.

## Why the distinction matters

Crystal plasticity is path dependent. The state after a sequence

$$
\Phi_N\circ\cdots\circ\Phi_2\circ\Phi_1
$$

is not generally equal to the state obtained by replacing the sequence with a
single increment of the same total boundary displacement. This applies to
slip, hardening, and any derived accumulated strain.

Consequently, comparisons with DIC must specify the frame, reference state,
load-state pair, and temporal accumulation rule. A post-processing filter or
temporal interpolation is a modelling choice and must not be confused with a
change in the constitutive loading path.

## Observable construction

For each accepted increment, store the displacement, stress, signed slip
increments, accumulated absolute slip, and any case-specific strain observable.
Construct the reported temporal observable from those accepted states using a
single documented convention. Do not mix a total displacement difference with
an accumulated internal variable unless that relation is part of the model.

The observation operator and its frame conventions are specified in
{doc}`../reference/scientific/observation_operator`. Current evidence and its limits are
summarised in {doc}`current_evidence`.

## Interpretation

Agreement at the boundary tests the imposed kinematics. Agreement in the
interior tests the constitutive law, orientation field, discretisation, and
solver together. A temporal observable can therefore be useful for comparing
model and experiment, but it should not be used to hide a discrepancy in the
underlying displacement or internal-state fields.
