# Prepare the DIC arrays

This guide transforms the four received files into canonical solver inputs. It
never modifies raw data in place.

## Verify Git LFS

```bash
git lfs pull
git lfs status
```

Expected files:

```text
data/raw/case_study/
├── U_40.npy
├── V_40.npy
├── el_thresh50.npy
├── Hardening_coeff_el_Thresh50.npy
└── manifest.json
```

The manifest fixes their size, type, shape, and SHA-256. Preparation stops if
any contract changes.

## Prepare the nominal ROI

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

This command makes four scientific choices explicit:

- `V_40 → u_x` and `U_40 → u_y`;
- displacement multiplied by `0.00184 mm/pixel`;
- local hardening multiplier multiplied by `380 MPa`;
- nine non-finite values replaced by their nearest finite neighbour.

The final row and column are duplicated to convert the `3600 × 3100` pixel
support into `3601 × 3101` nodes.

## Refuse automatic repair

To audit a new source before selecting a policy:

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/audit-only \
  --nonfinite-policy error
```

The command fails on the nine known values. Use `nearest` only after checking
that the reported indices agree with the raw manifest.

## Create a reproducible crop

```bash
.venv/bin/fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study-40x30 \
  --crop-nx 40 \
  --crop-ny 30 \
  --nonfinite-policy nearest
```

The crop is centred. Its bounds in the original ROI are stored in the manifest;
no implicit visual selection is performed.

## Check outputs without loading everything

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("data/processed/case-study/manifest.json").read_text()
)
for name, metadata in manifest["outputs"].items():
    print(name, metadata["shape"], metadata["dtype"], metadata["sha256"])
PY
```

See {doc}`../reference/input_contract` for expected shapes, units, and value
domains.

## Reuse an existing preparation

Repeating the exact command verifies and reuses the existing files. The command
refuses to write a different configuration into a non-empty directory. Use a
new directory for every variant:

```text
data/processed/
├── case-study-k380/
├── case-study-k396-historical/
└── case-study-40x30/
```

This separation prevents two scientific contracts from being mixed silently.

