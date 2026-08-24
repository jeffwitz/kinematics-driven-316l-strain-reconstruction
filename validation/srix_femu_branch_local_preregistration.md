# E-SRIX-FEMU-BRANCH-002A — preregistration

This is a bounded diagnostic around the failed refined interval
`[0.236328125, 0.23828125]`. It does not launch identification or P43.

The parent path is replayed to `f0=0.234375` and the endpoint is `f2=0.23828125`.
Local partitions use `alpha={0.25,0.40,0.50,0.60,0.75}`. The same oracle
configuration, SRIX preset, EBSD map and plane-stress backend are used for all
runs. Convergent endpoints are compared in displacement, stress and plastic
strain tensor against the coarse endpoint.

For the midpoint partition, two global initial-guess diagnostics are also
run: linear extrapolation from `f0,f1`, and the coarse endpoint displacement.
The coarse endpoint is used only as a displacement initial guess; its material
state is never copied.

The result is classified as a continuation issue only if all convergent local
partitions agree within numerical tolerance. Distinct convergent endpoints are
reported as a possible branch/discretization ambiguity. Failure of all local
partitions is reported as unresolved, not repaired by selecting a preferred
partition.
