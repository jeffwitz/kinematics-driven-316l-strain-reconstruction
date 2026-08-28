# Prepare a DIC case

**Mode:** how-to  
**Domain:** dic

## Prerequisites

Keep the raw displacement and local-descriptor arrays under
`data/raw/case_study`, and verify their units and row/column convention before
running the command. Do not start from an already filtered or transposed copy.

## Prepare the inputs

```bash
fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

Add `--crop-nx` and `--crop-ny` for a registered crop. The command preserves
the temporal path and valid mask and refuses incompatible shapes or unresolved
non-finite values.

## Verify the artifact

```bash
python - <<'PY'
import json
from pathlib import Path
manifest = Path("data/processed/case-study/manifest.json")
print(json.dumps(json.loads(manifest.read_text()), indent=2))
PY
```

The expected artifact is a manifest plus nodal displacements in millimetres,
material maps in MPa and recorded transformation/hash information. Check that
no hidden transpose, filtering or baseline subtraction is present.

Use {doc}`../../reference/data/input_contract` and
{doc}`../../reference/data/dic_axis_conventions` for the exact fields. The
case-preparation command is recorded in the campaign manifest; no hidden
transpose, filtering or baseline subtraction is permitted.
