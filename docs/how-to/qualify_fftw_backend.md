# Qualify the optional FFTW backend

Install the optional dependency in an isolated environment:

```bash
python -m pip install -e '.[dev,fftw]'
```

Run the numerical contracts with deterministic planning:

```bash
pytest -q \
  tests/unit/spectral2d/test_fftw_transforms.py \
  tests/unit/spectral2d/test_fftw_wisdom.py \
  tests/unit/spectral2d/test_preconditioner_backends.py
```

Run the reproducible benchmark separately from CI:

```bash
python scripts/benchmark_spectral_transforms.py \
  --meshes 12 24 48 96 192 \
  --backends scipy fftw \
  --fftw-threads 1 2 4 \
  --output validation/_generated/fftw/benchmark.json
```

The JSON separates planning, forward/inverse transforms, and the complete
`DST-I -> B0 -> inverse DST-I` preconditioner. Its median steady-state time is
the primary performance statistic. Do not claim a speedup until a benchmark
record is archived with the machine, library versions, planner effort, wisdom
state, and thread count.

The standard project installation and CI use SciPy and do not require pyFFTW.
