# Pre-registration: constitutively coupled nonlocal plasticity on P154

Date: 2026-07-25

## Scientific question

Can a micromorphic interaction in the evolution of J2 plasticity broaden the
raw plastic zones and improve FEM--DIC agreement without filtering the final
FEM observable?

The tested energetic coupling is

```text
0.5 Hchi (p - chi)^2 + 0.5 Hchi ell^2 |grad chi|^2
```

with the local yield radius

```text
R(p, chi) = sy0 + K h(p) + Hchi (p - chi).
```

Replacing `p` by `chi` in the Ludwik law is outside the registered model.

## Selection region

The sole development and coupling-selection region is partition P154 of a
`20 x 20` layout over the `3600 x 3100` element ROI.

| Property | Frozen value |
|---|---:|
| partition id | `154` |
| partition index | `(7, 14)` |
| retained core | `x=[1260,1440)`, `y=[2170,2325)` |
| retained elements | `180 x 155 = 27,900` |
| solved region | `x=[1132,1568)`, `y=[2042,2453)` |
| solved elements | `436 x 411 = 179,196` |
| validation padding | `128 pixels = 0.23552 mm` |
| nominal length | `58.88 um = 0.05888 mm = 32 pixels` |
| padding/length | `4.0` |

The `padding=64` profile is restricted to compilation, transaction and
convergence smoke tests. It cannot support a scientific conclusion at the
registered length.

## Frozen numerical controls

- 20 mechanical increments for validation;
- native MFront plane stress for the primary campaign;
- eight MGIS threads;
- fixed-point relaxation `omega=0.5`;
- mesh-independent mixed relative maximum-norm fixed-point tolerance `1e-6`;
- at most 15 micromorphic iterations per global Newton evaluation;
- Helmholtz residual at most `1e-10`;
- existing global Newton and cutback settings;
- homogeneous Neumann flux for `chi` at the padded-domain boundary.

No constitutive state may be committed during a micromorphic fixed point.

## Numerical amendment before candidate inspection

The first positive-coupling smoke execution exposed a missing norm definition:
the initial implementation accumulated pointwise changes in one global
\(L_2\) norm. During the five-increment `alpha=0.5` run this produced repeated
cutbacks for changes only slightly above `1e-6`; the acceptance difficulty
therefore depended on the number of elements in the ROI.

Before inspecting any positive-\(H_\chi\) DIC metric or selecting a candidate,
the stopping criterion was frozen as the mesh-independent mixed maximum norm

```text
max(abs(chi_next - chi)) /
max(1, max(abs(chi_next)), max(abs(chi_star)))
```

The relaxed change must be at most `omega * 1e-6`, which guarantees that the
final unrelaxed fixed-point residual is at most `1e-6`. Diagnostics record
`nonlocal_convergence_norm = mixed_relative_linf`. The original long-running
smoke is retained as a numerical diagnostic; all candidate comparisons use
fresh campaigns produced after this amendment.

## Coupling sweep

First compute the local P154 reference and define

```text
Href = median(K * n * p**(n - 1))
```

over retained-core elements satisfying `p > p0=1e-6`, using the final local
element-average PEEQ and the element hardening map.

The initial smoke sweep uses five increments and
`alpha in {0, 0.5, 1}` with `Hchi = alpha * Href`.

The primary 20-increment sweep uses the same three alpha values. `alpha=0.25`
or `alpha=2` may be added only if all registered metrics indicate that the
best candidate lies outside the covered interval. The selected alpha must be
frozen before any transfer to P42 or P48.

## Primary comparison

Metrics are evaluated only on the retained core. The primary FEM field is the
raw `EVM_HISTORICAL` reconstructed from the coupled displacement field with
the same operator as DIC. It must not be Helmholtz-filtered before primary
metrics are calculated.

Relative to the local P154 calculation, a coupled candidate must satisfy all
of:

- Pearson correlation gain at least `0.05`;
- relative-L2 reduction at least `5%`;
- top-10% localization IoU gain at least `0.02`;
- absolute DIC-q90 IoU gain at least `0.02`;
- predicted active fraction at the DIC-q90 threshold in `[5%, 20%]`;
- interior-displacement error degradation no greater than `5%`;
- no loss of the validated plane-stress residual level;
- no non-finite field;
- no abnormal cutback accumulation.

Filtered coupled fields are secondary diagnostics only.

## Allowed conclusions

1. **Coupled spatial interaction supported:** at least one positive `Hchi`
   passes every registered criterion.
2. **Partially supported:** the coupled model improves localization width but
   fails at least one amplitude, displacement, or robustness criterion.
3. **Insufficient:** no tested positive `Hchi` improves the registered spatial
   and field metrics together.

The selected length remains a diagnostic candidate. This campaign cannot by
itself identify a material internal length.
