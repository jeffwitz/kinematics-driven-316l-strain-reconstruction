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

For this experiment, the tensile direction is the image-column direction.
Image rows map to canonical transverse `x`; image columns map to canonical
tensile `y`. The array support is retained and the flow components are
exchanged:

```text
canonical[x=row, y=column, ux] = image_flow[row, column, drow]    * pixel_size
canonical[x=row, y=column, uy] = image_flow[row, column, dcolumn] * pixel_size
```

The pure functions implement this contract and are exact inverses up to
floating-point rounding. They do not transpose the spatial support.

## Historical U and V names

Direct reference-to-final correlation identifies the received convention
unambiguously:

```text
V_40 -> u_x, transverse
U_40 -> u_y, tensile
```

This is consistent across the raw-data README, prepared-data manifest,
preparation code and its regression tests. It is not inferred from generic
letter names. Among the eight component/sign candidates, `U=+flow_column` and
`V=+flow_row` gives correlations 0.9995 and 0.9969 with the archived fields;
the next candidates have errors tens of pixels larger. The residual biases
are consistent with an unavailable historical OpenCV binary and frame
provenance, not with a component ambiguity.

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
