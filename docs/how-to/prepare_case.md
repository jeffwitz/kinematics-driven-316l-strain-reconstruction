# Prepare a DIC case

**Category: How-to.**

## Prerequisites

- raw displacement and local-descriptor arrays are available;
- their axes and units are known;
- Git LFS objects have been fetched.

## Create canonical inputs

```bash
fem-inhouse prepare-case \
  --raw data/raw/case_study \
  --output data/processed/case-study \
  --hardening-scale-mpa 380 \
  --nonfinite-policy nearest \
  --nodal-completion edge-pad-upper
```

Use `fem-inhouse prepare-case --help` for crop and repair options. The command
writes nodal displacements in millimetres, element material maps in MPa and a
manifest containing transformations and hashes.

## Inspect the result

```bash
python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("data/processed/case-study/manifest.json").read_text())
print(json.dumps(manifest, indent=2))
PY
```

Expected failures include unresolved Git LFS pointers, incompatible shapes,
non-finite values without an explicit policy, and an already occupied output
directory.

See {doc}`../reference/input_contract` for the exact contract and
{doc}`../explanation/from_dic_to_mechanics` for the scientific role of these
inputs.
