# Run the TET2 SRIX comparison

Use the repository qualification entry point for the registered comparison:

```bash
python scripts/qualify_spectral2d_against_newton.py --help
```

Record the commit, grid, tolerance, reference parameters, thread count and
the JSON/NPZ output paths. The qualified comparison is TET2 against CPS4; EBI
is a separate falsification experiment.
