# DIC axis and displacement conventions

**Category: Reference.**

This page fixes the conversions used by the maintained image-level
observation operator. Plotting code is not allowed to introduce an additional
transpose, flip or component exchange.

## Coordinate systems

| Representation | Array axes | Components |
|---|---|---|
| OpenCV grayscale image | `(row, column)` | intensity |
| OpenCV flow | `(row, column, 2)` | `[column displacement, row displacement]` in pixels |
| canonical reconstruction field | `(x, y, 2)` | `[u_x, u_y]` in millimetres |
| FEM nodal `U` | `(x node, y node, 2)` | `[u_x, u_y]` in millimetres |
| element EVM | `(x element, y element)` | dimensionless |

For this experiment, image columns map to canonical `x` and image rows map to
canonical `y`. Conversion between image and canonical array support therefore
transposes the first two axes once:

```text
canonical[x, y, ux] = image_flow[row=y, column=x, dcolumn] * pixel_size
canonical[x, y, uy] = image_flow[row=y, column=x, drow]    * pixel_size
```

The pure functions `image_flow_to_canonical(...)` and
`canonical_to_image_flow(...)` implement this contract and are exact inverses
up to floating-point rounding.

## Historical U and V names

The received final fields use an experiment-specific convention:

```text
V_40 -> u_x, transverse
U_40 -> u_y, tensile
```

This is consistent across the raw-data README, prepared-data manifest,
preparation code and its regression tests. It is not inferred from generic
letter names.

The historical optical-flow script itself labels OpenCV component zero `u`
and component one `v`. Those local variable names do not override the
experiment-specific mapping of the received `U_40.npy` and `V_40.npy`
artefacts.

## Mask semantics

The unavailable historical `mask.png` was multiplied directly into `uint8`
images. Two explicit maintained operations exist:

- `legacy_uint8_multiply`: exact NumPy `uint8` multiplication, retained only
  for reproduction when the original mask is available;
- `binary_mask`: accepts boolean, `0/1` or `0/255` masks and applies clear
  zero/valid semantics.

Because the original mask is unavailable, new V3 campaigns use a generated
boolean all-valid mask identified as `declared_all_valid`. Its shape, values
and hash are recorded. This is a declared common support, not a claim about
the historical production mask.

## Spatial orientation in figures

All scientific maps are first converted to canonical `(x,y)` support. The
shared plotting helper is then responsible for displaying physical `x` and
`y` coordinates. A caller must not add `field.T`, `origin="lower"` or a flip
to make two maps appear aligned.
