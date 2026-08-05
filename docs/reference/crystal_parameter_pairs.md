# Paired 316L crystal-law parameter set

`316l_guilhem2013_nasri2018_meric_srix_rate_1e-3` is the only registered
configuration for a controlled Meric--SRIX comparison.

It is built from the existing immutable preset registries; the common values
are not copied into a second source of truth. The pair locks:

- cubic elasticity `(C11, C12, C44) = (197000, 125000, 122000) MPa`;
- the twelve FCC `<0,1,-1>{1,1,1}` slip systems;
- the seven-coefficient MFront interaction matrix
  `(1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6)`;
- `tau0 = 40 MPa`, `Q = 10 MPa`, `b = 3`, `C = 40000 MPa` and `d = 1500`;
- `293.15 K` and the same crystal orientation supplied by the run.

The only constitutive difference is the flow rule:

| law | flow parameters |
|---|---|
| Méric--Cailletaud | `K = 12 MPa`, `n = 11`, physical or pseudo-time increment |
| Forest--Rubin SRIX | `R = 18.781910070526294 MPa`, analytically transposed at `1e-3 s^-1` |

The SRIX value is a transposition, not a measurement or an identification on
P43. The shared backbone is a literature prior. The pair therefore authorizes
a controlled constitutive-formulation comparison, not a claim that the two
laws are equivalent material identifications.

## Selecting the pair

```yaml
constitutive_options:
  paired_parameter_set: 316l_guilhem2013_nasri2018_meric_srix_rate_1e-3
```

The option is accepted by both `fcc_meric_cailletaud` and
`fcc_forest_rubin_srix`. It injects the common MFront parameters explicitly and
then injects only the flow-rule-specific parameters. Combining it with the
legacy `parameter_set` or `parameters` options is refused.

The legacy SRIX `parameter_set` API remains available for compatibility. A
qualification script must use `--paired-parameter-set`; an Méric run cannot
use the SRIX-only `parameter_set` option.

## Structural contract

The two `.mfront` sources are fingerprinted before a paired qualification.
`CrystalStructure`, `SlidingSystem` and `InteractionMatrix` must occur once in
each source and must match exactly. A missing, duplicated or malformed
declaration fails before the solver starts.

The generated reports include the canonical backbone digest, the structure
digest, the orientation digest, the imposed-boundary digest and the complete
flow-rule manifest. `compare_paired_crystal_reports.py` refuses a comparison
when any common digest, mesh, crop, units or imposed field differs. Different
increment counts are explicitly not field- or performance-comparable.

The locked contract is archived in
`validation/_generated/performance/crystal_316l_meric_srix_pair_contract.json`.
