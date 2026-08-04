# Newton-GMRES contract

Unknowns are the two displacement components at interior nodes:

```text
2 * (nx - 1) * (ny - 1)
```

The Jacobian is a matrix-free action. GMRES receives the exact current
right-hand side and uses the DST-I/B0 inverse as preconditioner. A candidate
must be integrated from the last committed constitutive state. Rejected trials
are reverted. The final accepted field is independently re-integrated after
`revert()` before `commit()`.

The solve records Newton iterations, GMRES iterations, material evaluations,
line-search evaluations, residual verification and transaction failures.
