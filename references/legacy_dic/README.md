# Legacy DIC identification sources

These files preserve the historical image-to-map workflow supplied in 2026.
They are provenance sources, not modules executed by the maintained
`fem_inhouse` package.

## Credits and roles

- `dic_displacement_fields.py`: dense OpenCV DIS optical flow. The source
  credits Eddidoune (2021) as original author and `kilincadil` as adaptor.
- `yield_stress_hardening.py`: pixel-wise yield-onset and hardening-map
  identification. The source credits Qi Hu as original author and
  `kilincadil` for restructuring.
- `LEGACY_README.md`: the README received with the two scripts.
- `requirements.txt`: the received dependency list.

## Byte-preserving archival

The four received files were moved without changing their content.

| Archived file | SHA-256 before move | SHA-256 after move |
|---|---|---|
| `dic_displacement_fields.py` | `fdbd9f6c7a75750eedfabc203e98018631ebe7094e14008017b2a6c93a167855` | `fdbd9f6c7a75750eedfabc203e98018631ebe7094e14008017b2a6c93a167855` |
| `yield_stress_hardening.py` | `d6fdb6d460b0bf6f72d1516cc835893ac4f8de1a298c923ca3fc72a25d3a525c` | `d6fdb6d460b0bf6f72d1516cc835893ac4f8de1a298c923ca3fc72a25d3a525c` |
| `LEGACY_README.md` | `0cfa356f7e07cd77ced1b269c94aa504a0b4fb5f042e72769d94edd45b74ba59` | `0cfa356f7e07cd77ced1b269c94aa504a0b4fb5f042e72769d94edd45b74ba59` |
| `requirements.txt` | `e2f7ec3801a47c82fe6327f83cff2a3f42857c3cb8581193823fb3e3c4b6c8b1` | `e2f7ec3801a47c82fe6327f83cff2a3f42857c3cb8581193823fb3e3c4b6c8b1` |

## Component convention

OpenCV returns `flow[..., 0]` as column/X displacement and `flow[..., 1]`
as row/Y displacement. The script stores these as `U_<i>.npy` and
`V_<i>.npy`, respectively. In the received experiment, however, the
downstream data contract names the traction-axis field `U_40` and the
transverse field `V_40`. The maintained canonical mapping is therefore:

```text
historical V_40 -> canonical u_x (transverse, array axis 0)
historical U_40 -> canonical u_y (tensile, array axis 1)
```

This experiment-specific mapping is documented and tested in the maintained
pipeline; generic `U` and `V` names must not be interpreted without it.

## Explicit DISFlow requests

`dic_displacement_fields.py` creates the object with
`cv2.DISOpticalFlow_create()` and explicitly sets:

| Parameter | Value |
|---|---:|
| finest scale | 0 |
| patch size | 4 |
| patch stride | 1 |
| variational alpha | 100 |
| variational delta | 1 |
| variational gamma | 0 |
| Charbonnier epsilon | 0.002 |
| variational iterations | 30 |

It does **not** explicitly set:

- a factory preset;
- gradient-descent iterations;
- mean normalisation;
- spatial propagation.

Values returned for those settings by OpenCV 4.14 are observable defaults of
the present reproduction environment, not certified settings of the
historical executable.

## Historical paths and mask

The scripts contain personal Windows paths below
`C:\Users\adil.kilinc\Desktop\Thesis\3_data\21_DIC` and a custom OpenCV build
path below `C:\Users\adil.kilinc\opencv`. They are retained verbatim.

The referenced `mask.png` was not supplied. The historical script reads it as
grayscale and evaluates `uint8_image * mask`; its dtype, values and exact
arithmetic effect cannot currently be audited. Maintained experiments use a
separately declared all-valid binary mask and do not claim reproduction of
the unknown historical support.

## Known limitations

- historical OpenCV version and binary are unavailable;
- implicit DIS defaults are not certified;
- the acquisition log and image/load synchronisation are unavailable;
- the historical mask is unavailable;
- paths are machine-specific;
- yield-map identification includes hard-coded interpolation, smoothing,
  clipping and neighbour filling;
- the hardening output is normalised by `SLOPE_REF` and is not directly a
  modulus in MPa without an explicit conversion.

The directory is excluded from Ruff for byte preservation. No nominal package
code imports it.
