# Add a nonlocal criterion

**Category: How-to.**

## Scope of the first extension interface

The current fixed-point workspace accepts a scalar element-centred nonlocal
field. The historical PEEQ-Helmholtz law is registered as
`peeq_helmholtz`. New scalar quantities, including accumulated slip or a
crystal-plasticity activity indicator, can be added without modifying Newton.

Tensor-valued fields require a later generalisation of the workspace and the
result schema. Do not flatten a tensor into an undocumented scalar merely to
fit the first interface.

## Implement the criterion

Implement `ScalarNonlocalCriterion`. The object owns four constitutive and
spatial decisions:

1. whether the selected material batch exposes the required fields;
2. how the nonlocal trial field is sent to the material batch;
3. how the local source and the constitutive safety observable are read;
4. how the spatial regularisation is evaluated.

The fixed-point driver retains responsibility for transactions, relaxation,
convergence, failure classification and the final consistent-tangent trial.

The spatial method returns `NonlocalRegularisationResult`. It is therefore
possible to replace Helmholtz with another verified operator without changing
the mechanical algorithm.

## Register and select it

Register a factory in `NONLOCAL_CRITERIA`:

```python
from fem_inhouse.core.nonlocal_criteria import NONLOCAL_CRITERIA

NONLOCAL_CRITERIA.register(
    "accumulated_slip_helmholtz",
    lambda options: AccumulatedSlipCriterion(**options),
)
```

An installed package can expose the same factory through the entry-point group
`fem_inhouse.nonlocal_criteria`. Factories are discovered automatically on
first use.

Select it in the case configuration:

```yaml
nonlocal_plasticity:
  enabled: true
  criterion: accumulated_slip_helmholtz
  criterion_options:
    slip_norm: l1
```

Criterion factories must validate every option. Unknown options must fail
before the first assembly.

## Validate

At minimum, add tests for:

- a uniform field;
- a signed field if negativity is allowed;
- zero coupling as an exact local non-regression;
- repeated trial evaluations followed by `revert`;
- final tangent evaluation from the same committed state;
- a deliberately incompatible material plugin;
- preservation of the relevant integral or mean, when claimed by the spatial
  operator.
