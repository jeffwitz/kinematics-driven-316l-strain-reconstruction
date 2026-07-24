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

`hardening_mode="tabular"` is meaningful only with the historical Python
backend. The nominal MFront path never allocates the 1000-point table.

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
| `padding` | non-negative integer | `150` |
| `partition_id` | `0 <= id < count` | task-specific |

For 100 partitions:

```text
partition_id = index_x * 10 + index_y
```

The complete `(3600, 3100)` domain is divided into `(10, 10)` balanced cores.

