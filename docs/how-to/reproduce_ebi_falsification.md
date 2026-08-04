# Reproduce the EBI falsification

Run the state-sharing qualification script and compare the three fields:

```text
CPS4
traditional TET2 with two SRIX states per pixel
EBI-TET with one shared SRIX state per pixel
```

The decisive metric is the accumulated-slip difference between EBI and TET2
at identical TET2 kinematics. The registered result is
`experimental_falsified_for_registered_SRIX_case`; do not generalize it to
other constitutive laws or loading paths.
