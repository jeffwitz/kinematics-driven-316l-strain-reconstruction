# Run the TET2 SRIX comparison

Use the repository qualification entry point for the registered comparison:

```bash
python scripts/qualify_spectral2d_against_newton.py --help
```

The complete 12x12 command is:

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
python scripts/qualify_spectral2d_against_newton.py \
  --mesh 12 --increments 8 --repeats 1 --tolerance 1e-8 \
  --anderson-target polarization --update-safeguard published_none \
  --reference-parameter-mode projected \
  --output validation/_generated/spectral2d_registered
```

The summary and JSONL traces are written below the selected output directory.
Archive them with the commit SHA and the MFront library SHA.

Record the commit, grid, tolerance, reference parameters, thread count and
the JSON/NPZ output paths. The qualified comparison is TET2 against CPS4; EBI
is a separate falsification experiment.
