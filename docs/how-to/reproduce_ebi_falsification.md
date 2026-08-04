# Reproduce the EBI falsification

Run the state-sharing qualification script and compare the three fields:

```text
CPS4
traditional TET2 with two SRIX states per pixel
EBI-TET with one shared SRIX state per pixel
```

The archived 24x24 experiment is reproduced with:

```bash
MFRONT_BEHAVIOUR_LIBRARY=build/mfront/src/libBehaviour.so \
python scripts/qualify_ebi_state_sharing.py \
  --mesh 24 --increments 8 --tolerance 1e-8 \
  --output validation/_generated/ebi_tet/state_sharing_m24_reproduced.json
```

The report contains field errors, verified residuals, state counts, Newton and
GMRES iterations, and timings. The decisive archived values are
`E_Gamma(EBI,TET2)=0.05393` and `E_Gamma(TET2,CPS4)=0.00722`.

The decisive metric is the accumulated-slip difference between EBI and TET2
at identical TET2 kinematics. The registered result is
`experimental_falsified_for_registered_SRIX_case`; do not generalize it to
other constitutive laws or loading paths.
