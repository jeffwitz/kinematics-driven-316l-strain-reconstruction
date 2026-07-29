# P43 measured history: state-bridge indexing amendment

Date: 2026-07-30

This amendment supersedes only the selected state in
`dic_multistep_p0043_state4_bridge_preregistration.md`.

## Detected indexing error

For a history `u[0], ..., u[40]`, NumPy entry

```text
diff(u, n=2)[i] = u[i+2] - 2 u[i+1] + u[i]
```

is centred on state `i+1`. The exploratory script labelled it as `i+2`.
Consequently, the value

```text
4.213850188534707e-4 mm
```

belongs to state 3, not state 4. The generated state-4 artefact was rejected
before any mechanical solve and is not evidence.

## Corrected selection

The single bridged state is therefore state 3:

```text
u_3 = 0.5 * (u_2 + u_4)
```

Every other rule, solver parameter, constitutive-state rule, predictor
comparison and interpretation boundary from the original preregistration
remains unchanged.

The conditioning report must independently verify that state 3 is the largest
RMS second-difference state before a full mechanical run is authorised.
