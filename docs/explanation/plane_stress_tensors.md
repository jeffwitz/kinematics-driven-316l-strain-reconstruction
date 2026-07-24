# Reconstructing complete tensors from a 2D plane-stress solve

## This is post-processing, not a 3D finite-element model

The mechanical calculation remains strictly two-dimensional:

- the unknowns are \(u_x\) and \(u_y\);
- the CPS4 mesh, strain-displacement matrix, global Newton system, and
  plane-stress tangent are unchanged;
- there is no \(u_z\), thickness discretisation, or second constitutive solve.

After an increment has converged, the code completes the accepted 2D state
with the transverse components implied by small-strain J2 plasticity. The
reconstructed tensors are outputs only and are never fed back into Newton.

## Plane stress is not plane strain

Plane stress imposes

$$
\sigma_{33}=\sigma_{13}=\sigma_{23}=0.
$$

The membrane kinematics additionally set
\(\epsilon_{13}=\epsilon_{23}=0\). They do **not** set
\(\epsilon_{33}=0\). The thickness strain must adapt so that the transverse
stress vanishes.

The public stress tensor is therefore

$$
\boldsymbol{\sigma}=
\begin{bmatrix}
\sigma_{11}&\sigma_{12}&0\\
\sigma_{12}&\sigma_{22}&0\\
0&0&\sigma_{33}
\end{bmatrix}.
$$

For the Python backend, \(\sigma_{33}\) is zero by construction. For MFront,
the native numerical value is preserved as
`plane_stress_residual_mpa`; it is not overwritten with zero.

## Elastoplastic reconstruction

Small strain is additively decomposed:

$$
\boldsymbol{\epsilon}
=\boldsymbol{\epsilon}^{e}+\boldsymbol{\epsilon}^{p}.
$$

Associated J2 flow is isochoric, hence

$$
\mathrm{tr}(\boldsymbol{\epsilon}^{p})=0,\qquad
\epsilon^p_{33}=-(\epsilon^p_{11}+\epsilon^p_{22}).
$$

The accepted in-plane elastic strains are

$$
\epsilon^e_{11}=\epsilon_{11}-\epsilon^p_{11},\qquad
\epsilon^e_{22}=\epsilon_{22}-\epsilon^p_{22}.
$$

Isotropic elasticity and \(\sigma_{33}=0\) give

$$
\epsilon^e_{33}
=-\frac{\nu}{1-\nu}
(\epsilon^e_{11}+\epsilon^e_{22})
=-\frac{\nu}{E}(\sigma_{11}+\sigma_{22}).
$$

The total transverse strain is finally

$$
\boxed{\epsilon_{33}=\epsilon^e_{33}+\epsilon^p_{33}}.
$$

Every shear strain vector exposed by the historical API uses engineering
shear, \(\gamma_{12}=2\epsilon_{12}\). Consequently, the off-diagonal tensor
entry is half the third vector component.

## Why the historical closure fails after yielding

Take \(\nu=0.3\), a converged total in-plane state
\((\epsilon_{11},\epsilon_{22})=(0.012,-0.003)\), and plastic strains
\((\epsilon^p_{11},\epsilon^p_{22})=(0.008,-0.002)\).

The mechanically consistent reconstruction gives

$$
\epsilon^p_{33}=-0.006,\qquad
\epsilon^e_{33}=-\frac{0.3}{0.7}(0.004-0.001)=-0.001285714,
$$

and therefore \(\epsilon_{33}=-0.007285714\).

Applying the purely elastic closure directly to total strain would instead
give

$$
-\frac{0.3}{0.7}(0.012-0.003)=-0.003857143.
$$

The error is \(0.003428571\), because the latter expression ignores plastic
incompressibility. That old operation remains available only as the explicitly
named `EVM_HISTORICAL` article-comparison metric. The complete-tensor measure
is named `EVM_RECONSTRUCTED_3D`.

## Engineering, tensor, and Kelvin conventions

The conversions are centralised in
`fem_inhouse.core.tensor_reconstruction`:

| Representation | Strain components | Stress components |
|---|---|---|
| FE engineering 2D | \([e_{11},e_{22},\gamma_{12}]\) | \([s_{11},s_{22},s_{12}]\) |
| 3D tensor | \(e_{12}=\gamma_{12}/2\) | \(s_{12}\) |
| MFront Kelvin plane stress | \([e_{11},e_{22},e_{33},\gamma_{12}/\sqrt{2}]\) | \([s_{11},s_{22},s_{33},\sqrt{2}s_{12}]\) |
| MFront Kelvin 3D | \([e_{11},e_{22},e_{33},\sqrt{2}e_{12},\sqrt{2}e_{13},\sqrt{2}e_{23}]\) | \([s_{11},s_{22},s_{33},\sqrt{2}s_{12},\sqrt{2}s_{13},\sqrt{2}s_{23}]\) |

The \(\sqrt{2}\) scaling preserves tensor double contractions in Kelvin
coordinates. Conversion factors are tested and are not duplicated in the
solver.

## Native MFront state

The installed MGIS metadata for `PixelLudwikJ2Plasticity` exposes:

| MGIS variable | Size | Use |
|---|---:|---|
| gradient `Strain` | 4 | in-plane total strain and Kelvin shear |
| force `Stress` | 4 | complete Kelvin stress, including residual \(S_{33}\) |
| internal `ElasticStrain` | 4 | complete elastic strain |
| internal `AxialStrain` | 1 | native total \(\epsilon_{33}\) |
| internal `EquivalentPlasticStrain` | 1 | accumulated PEEQ |

For this behaviour, the third entry of the gradient array remains zero and is
**not** the converged axial strain. The adapter validates names, types, sizes,
and offsets, then reads `AxialStrain` after global convergence and before
commit. Plastic strain is the difference between native total and elastic
strain. The public MFront tensors retain those native values.

If a native MFront behaviour does not expose `AxialStrain`, this backend fails
with a capability error. It does not silently apply the J2 analytical
completion to an anisotropic or otherwise incompatible law. Analytical legacy
completion is available only when the caller explicitly declares
`completion_strategy="j2_isotropic_analytical"`.

The six-component path uses the verified MGIS order
`[11,22,33,12,13,23]` and retains all three transverse components. Its local
condensation and Schur-complement tangent are explained in
{doc}`mfront_3d_condensation`.

## Equivalent measures

The complete tensors support three unambiguous invariants:

$$
\epsilon_\mathrm{eq}
=\sqrt{\frac{2}{3}\boldsymbol{\epsilon}_\mathrm{dev}:
\boldsymbol{\epsilon}_\mathrm{dev}},
$$

$$
\epsilon^p_{\mathrm{eq,tensor}}
=\sqrt{\frac{2}{3}\boldsymbol{\epsilon}^{p}:
\boldsymbol{\epsilon}^{p}},
$$

and

$$
\sigma_\mathrm{VM}
=\sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}.
$$

The instantaneous plastic tensor norm is not generally equal to PEEQ. PEEQ is
accumulated along the loading path; equality is expected only for suitable
proportional monotonic histories.

## FEM fields versus a single DIC image

The FEM state can be completed because the solver knows the accepted loading
history and plastic internal variables. A single final DIC image provides only
\(\epsilon_{11}\), \(\epsilon_{22}\), and \(\gamma_{12}\). Once plastic flow
has occurred, these three values do not identify \(\epsilon_{33}\).

A complete DIC reconstruction would require the time history, initial state,
material parameters, and local constitutive integration at every pixel. The
software therefore does not apply the elastic closure automatically to a
plastified DIC map. Such an extension is a separate phase.
