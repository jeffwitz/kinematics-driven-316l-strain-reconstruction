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
| `require_pypardiso` | `true` | require MKL-backed sparse direct solve |
| `hardening_mode` | `"ludwik"` | analytical nominal mode |
| `constitutive_backend` | `"mfront"` | constitutive implementation |
| `mfront_library` | `build/mfront/src/libBehaviour.so` | generic-interface library |
| `mfront_threads` | `1` | explicit MGIS thread-pool size |
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
| `mfront-3d-condensed-plane-stress` | experimental 3D law with local condensation |
| `python` | historical analytical/tabulated J2 regression implementation |

The local plane-stress controls are used only by the condensed 3D backend.

## NonlocalPlasticityConfig

This optional configuration activates the staggered micromorphic J2
extension. It does not alter a campaign when `enabled=false`.

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `enabled` | `false` | — | select the micromorphic MFront behaviours and fixed point |
| `length_scale_mm` | `0.05888` | mm | Helmholtz interaction length |
| `coupling_modulus_mpa` | `0.0` | MPa | energetic coupling modulus \(H_\chi\) |
| `relaxation` | `0.5` | — | fixed-point relaxation \(\omega\) |
| `relative_tolerance` | `1e-6` | — | mesh-independent mixed relative maximum-norm tolerance on \(\chi\) |
| `maximum_iterations` | `15` | — | fixed-point iteration limit per mechanical Newton trial |
| `maximum_helmholtz_residual` | `1e-10` | — | accepted relative DCT equation residual |

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
