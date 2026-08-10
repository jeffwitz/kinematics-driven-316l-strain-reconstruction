# Configuration reference

The typed configuration is composed of `MeshConfig`, `MaterialConfig`, and
`SolverConfig` inside `CaseStudyConfig`.

## MeshConfig

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `nx` | `int` | required | element count along array axis 0 / physical $x$ |
| `ny` | `int` | required | element count along array axis 1 / physical $y$ |
| `base_pixel_size_mm` | `float` | `0.001` | base coordinate unit |
| `scale_factor` | `float` | `1.84` | microscopy pixel scale |

`element_size_mm = base_pixel_size_mm * scale_factor`.

## MaterialConfig

| Field | Default | Unit | Use |
|---|---:|---|---|
| `young_modulus_mpa` | `205000` | MPa | homogeneous elasticity |
| `poisson_ratio` | `0.30` | — | homogeneous elasticity and plane stress |
| `hardening_exponent` | `0.245` | — | homogeneous Ludwik exponent |
| `plastic_strain_max` | `0.2` | — | historical Python table only |
| `plastic_table_points` | `1000` | — | historical Python table only |
| `first_positive_plastic_strain` | `1e-6` | — | origin regularization |

The MFront behaviour fixes $E$, $\nu$, and the regularization point to
these supported values. The solver rejects incompatible values rather than
mixing definitions.

Initial yield stress and hardening coefficient are not scalar configuration
fields. They are element-wise input maps.

## SolverConfig

| Field | Default | Meaning |
|---|---:|---|
| `increments` | `20` | nominal pseudo-time increments |
| `max_newton_iterations` | `15` | iteration limit per attempted increment |
| `residual_tolerance` | `1e-6` | relative residual convergence threshold |
| `minimum_step_divisor` | `1024` | smallest accepted fraction of nominal step |
| `element_formulation` | `"cps4"` | `cps4` reference, one-point `cps4r`, or assumed-strain `cps4r_as` |
| `stabilisation_strategy` | `"assumed_strain_energy"` | `cps4r_as` only; see {doc}`quas4_assumed_strain_derivation` |
| `stabilisation_projection` | `"asmd"` | `cps4r_as` only; `(e1, e2, e3)` of R3.06.10 |
| `stabilisation_tangent_floor` | `1e-6` | `cps4r_as` only; spectral floor of the energy variant |
| `jacobian_correction` | `"none"` | `cps4r_as` only; `broyden` is implemented and **rejected** |
| `jacobian_correction_memory` | `5` | secant pairs kept per element, `1..5` |
| `hourglass_scale` | `1.0` | CPS4R stiffness scale, with `0 < beta <= 1` |
| `hourglass_energy_warning_ratio` | `0.01` | warn above this energy ratio; `None` disables |
| `hourglass_energy_failure_ratio` | `None` | optional hard failure threshold |
| `require_pypardiso` | `true` | require MKL-backed sparse direct solve |
| `hardening_mode` | `"ludwik"` | analytical nominal mode |
| `constitutive_backend` | `"mfront"` | constitutive implementation |
| `mfront_library` | `build/mfront/src/libBehaviour.so` | generic-interface library |
| `mfront_threads` | `1` | explicit MGIS thread-pool size |
| `mfront_behaviour_id` | `null` | optional declarative MFront catalogue entry |
| `constitutive_options` | `{}` | options forwarded unchanged to a registered plugin |
| `local_plane_stress_tolerance_mpa` | `1e-8` | absolute transverse-stress tolerance |
| `local_plane_stress_relative_tolerance` | `1e-10` | relative transverse-stress tolerance |
| `maximum_local_plane_stress_iterations` | `15` | local Newton iteration limit |
| `maximum_cbb_condition_number` | `1e12` | reject an ill-conditioned transverse tangent |

`hardening_mode="tabular"` is meaningful only with the historical Python
backend. The nominal MFront path never allocates the 1000-point table.

Supported backend values are:

| Value | Meaning |
|---|---|
| `mfront` | compatibility alias for native MFront plane stress |
| `mfront-native-plane-stress` | explicit native MFront plane stress |
| `mfront-3d-condensed-plane-stress` | 3D law with the plane-stress closure solved outside it, in Python; works with any 3D behaviour and is the numerical reference |
| `mfront-native-generalised-plane-stress` | 3D law whose own local Newton carries the closure; needs a GPS variant of the behaviour |
| `mfront-structural-plane-stress` | generated structural plane-stress closure for the V1 `Implicit`/`StandardElasticity` 3D contract |
| `python` | historical analytical/tabulated J2 regression implementation |
| any registered identifier | process-local constitutive plugin |

{doc}`../how-to/choose_mfront_backend` says which one to use, and
{doc}`numerics/three_dimensional_condensation` derives the plane-stress
routes. The local plane-stress controls below are used only by the condensed
3D backend.

### Selecting the law and its orientations

`constitutive_options` is forwarded unchanged to the selected backend, which
rejects any key it does not know. The crystal backends accept the following,
whichever plane-stress route is chosen.

| Key | Default | Meaning |
|---|---:|---|
| `crystal_orientation` | identity | mapping with a `mode` of `homogeneous` (one Bunge Euler triple, degrees) or `ebsd` (one triple per point, shaped `(nx, ny, 3)`). Absent, every point keeps the identity rotation, which is a single-crystal aligned with the global frame and almost never what a real case wants |
| `parameter_set` | `null` | named entry of the SRIX parameter registry, e.g. `316l_srix_transposed_from_nasri2018_rate_1e-3`. Only behaviours declaring a registry accept it; Méric-Cailletaud rejects it and asks for `paired_parameter_set` |
| `parameters` | `null` | explicit parameter overrides, for a set that is not registered. Mutually exclusive with `paired_parameter_set` |
| `paired_parameter_set` | `null` | a matched SRIX / Méric-Cailletaud pair, for comparing the two flow rules on parameters that mean the same thing. Cannot be combined with `parameter_set` or `parameters` |
| `condensation_block_size` | `2500` | condensed backend only; number of points condensed per block. A memory-versus-overhead knob, not a numerical one |

The 316L parameter sets are **transposed from published work, not identified on
this material**. See {doc}`../how-to/use_srix_crystal_law` before presenting a
number obtained with them.

### Production options for the generalised plane-stress backend

Read by `mfront-native-generalised-plane-stress` and
`mfront-structural-plane-stress`. **The condensed backend accepts these keys
and then ignores them**: it has no local Newton to sub-step, so there is no
tangent to repair. A `gps_*` key left in a configuration switched to
`mfront-3d-condensed-plane-stress` is silently inert — the one exception is
`gps_shadow_tangent`, which is rejected with an error. Check
`constitutive_backend` before reading a result as evidence that one of these
options did something.

| Key | Default | Meaning |
|---|---:|---|
| `gps_composite_fd_tangent` | `false` | rebuild, by finite differences along the composite trajectory, the tangent of points the local Newton had to sub-step. Recommended `true` for the qualified SRIX + EBSD workflow; otherwise a sub-stepped point returns its last sub-step's tangent. On P43 M100 it takes GPS from 85 Newton iterations to 58, against 57 for the reference |
| `gps_composite_fd_step` | `1.0e-6` | absolute engineering-strain perturbation used by the central finite difference. A numerical tolerance, not a relative increment scale |

### Optional smoothing of the SRIX flow rule

Accepted by every crystal backend, condensed included, because they set MFront
behaviour parameters rather than steering the host bridge. Both are refused for
any law other than Forest-Rubin SRIX.

The SRIX flow rule is built on Macaulay brackets, which are not differentiable
where they switch. These two keys optionally replace them by a generalized
Charbonnier norm. This is an experimental constitutive modification: it does
not preserve the sharp law and has not been shown to improve the qualified
M200 workflow.

| Key | Default | Meaning |
|---|---:|---|
| `srix_smoothing_epsilon` | `0.0` | stress scale of the regularisation, MPa. **`0.0` selects the historical non-smooth constitutive branch**, not a small-regularisation approximation. The inactive-system Jacobian was corrected in commit `51ace9e`, so the Newton path need not be bit-for-bit identical to older archived binaries. Any positive value changes the constitutive response and invalidates comparison with archived campaigns |
| `srix_smoothing_exponent` | `11.0` | exponent of the generalized Charbonnier norm. Higher is closer to the sharp bracket. Without a positive `srix_smoothing_epsilon` it has no effect |
| `srix_slip_smoothing_delta` | `0.0` | experimental compact C2 width for `abs(dg)` and its derivative, in strain units. It does not smooth the Macaulay bracket. `0.0` selects the historical `abs/sign` branch; positive values change the constitutive response and must be reported with the results |
| `srix_slip_zero_derivative` | `-1.0` | diagnostic only: subgradient used at exactly `dg=0` when `srix_slip_smoothing_delta=0`. `-1` preserves the historical branch; `0` is the semismooth control. Do not use it as a production calibration parameter |

### Diagnostic options

These must remain disabled in production unless a validation experiment
explicitly requests them. They exist to explain a run, not to produce one.

| Key | Default | Meaning |
|---|---:|---|
| `gps_failure_diagnostics` | `false` | record, per already-isolated GPS integration failure, the state that produced it. It adds diagnostic state copies to the existing bisection probes; only the GPS and structural backends act on it |
| `gps_shadow_tangent` | `false` | replace the Newton matrix by the reference Schur evaluated at the GPS state; useful for diagnosis, not production. Rejected by the condensed backend |
| `gps_shadow_tangent_scope` | `"all"` | diagnostic scope for the shadow tangent |

## NonlocalPlasticityConfig

This optional configuration activates the staggered micromorphic J2
extension. It does not alter a campaign when `enabled=false`.

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `enabled` | `false` | — | select the micromorphic MFront behaviours and fixed point |
| `criterion` | `peeq_helmholtz` | — | registered scalar source and spatial operator |
| `criterion_options` | `{}` | — | options validated by the criterion factory |
| `length_scale_mm` | `0.05888` | mm | Helmholtz interaction length |
| `coupling_modulus_mpa` | `0.0` | MPa | energetic coupling modulus \(H_\chi\) |
| `relaxation` | `0.5` | — | fixed-point relaxation \(\omega\) |
| `relaxation_strategy` | `fixed` | — | historical fixed Picard or optional bounded `aitken` |
| `minimum_relaxation` | `0.05` | — | lower Aitken bound |
| `maximum_relaxation` | `0.8` | — | upper Aitken bound |
| `aitken_residual_growth_factor` | `1.25` | — | reject acceleration above this residual-growth ratio |
| `relative_tolerance` | `1e-6` | — | mesh-independent mixed relative maximum-norm tolerance on \(\chi\) |
| `maximum_iterations` | `15` | — | fixed-point iteration limit per mechanical Newton trial |
| `maximum_helmholtz_residual` | `1e-10` | — | accepted relative DCT equation residual |
| `record_iteration_history` | `false` | — | persist per-fixed-point and enclosing Newton diagnostics |

The current implementation requires an MFront backend. The Python J2 backend
is retained as an independent local reference and rejects nonlocal activation.
The MFront tangent is consistent at fixed \(\chi\); it is not a monolithic
coupled tangent.

## PreparationConfig

| Field | Default | Meaning |
|---|---:|---|
| `pixel_size_um` | `1.84` | displacement conversion |
| `hardening_scale_mpa` | `380.0` | multiplier-to-MPa conversion |
| `nonfinite_policy` | `"error"` | fail until repair is selected explicitly |
| `nodal_completion` | `"edge-pad-upper"` | supported nodal completion |
| `crop_nx`, `crop_ny` | `None` | optional deterministic central crop |

Both crop dimensions must be provided together.

## Partition layout

The production CLI supports:

| Parameter | Values | Nominal article value |
|---|---|---:|
| `count` | `25` or `100` | `100` |
| `parts_x`, `parts_y` | positive integers supplied together | unset |
| `padding` | non-negative integer | `150` |
| `partition_id` | `0 <= id < count` | task-specific |

For 100 partitions:

```text
partition_id = index_x * 10 + index_y
```

The complete `(3600, 3100)` domain is divided into `(10, 10)` balanced cores.
`--parts-x 20 --parts-y 20` selects the P154 development layout while the
legacy `--count` interface remains unchanged.
