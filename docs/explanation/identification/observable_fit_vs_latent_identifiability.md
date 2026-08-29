# Observable fit versus latent identifiability

The inverse studies in this repository establish a boundary that is easy to
miss: reproducing the measured kinematics does not, by itself, identify the
constitutive state that produced them. This page gathers the conclusion from
the tensor, local-coefficient and TANN/inverse-closure campaigns. They are
historical studies, but the methodological lesson is part of the current
identification strategy.

## The question tested

The first experiments asked whether one could infer an inelastic field directly
from displacement data:

```text
measured or synthetic kinematics
          ↓
infer a free latent inelastic field
          ↓
obtain an excellent displacement reconstruction
          ↓
ask whether that latent field is unique
```

This is a different question from fitting a qualified constitutive law. The
forward map can be accurate while its inverse remains many-to-one.

## A. Linear non-identifiability of a free field

The tensor-inverse campaign allowed several independent tensor components at
each spatial patch. Its parameter-to-displacement operator had 192 latent
directions but only 173 effective singular directions; the condition number was
about $3.5\times10^{16}$. Uniform eigenstrain is an especially transparent
null mode: it creates uniform eigenstress, whose divergence is zero, and thus
produces exactly zero displacement in the tested setting.

The consequence is not a weak optimiser. In the registered twin, the free
tensor family reached an objective of $2.43\times10^{-15}$, while the recovered
plastic field still had roughly 80% gauge error at the least-squares optimum.
Removing the spatial mean improved the floor only to about 52%. A large change
in the latent field can therefore be invisible to the observable displacement.

This is **linear non-identifiability**: the mapping from latent field to
observable has a substantial nullspace. Truncating the SVD can regularise the
solution, but it cannot manufacture information in those missing directions.
The detailed gates and nullspace construction remain in
`validation/tensor_local_inverse_results.md` and the associated evidence
records.

## B. A compact representation can still be non-unique

The next campaign restricted the latent field to a smaller local-coefficient
parameterisation. At the degree-zero setting, the parameter-to-observable map
was well behaved (condition number about $2.0\times10^2$), and a synthetic twin
was recovered to $2.1\times10^{-4}$ relative field error. This is a useful
qualification of the inverse plumbing and of that restricted representation;
it is not evidence that an unrestricted constitutive state is identifiable.

Enriching the same partition-of-unity basis exposed its own algebraic nullspace:
the degree-one representation had condition number about $4.1\times10^{15}$
and 23 numerically absent directions. The degeneracy came from the assembly
map, not from a failure of the mechanical solve.

More generally, even a compact representation with a healthy local spectrum
does not guarantee a unique nonlinear inverse. A low residual only says that
one latent state explains the observable. It does not say that another latent
state, or another constitutive history, cannot explain it equally well. The
linear spectrum is therefore necessary information, but not sufficient proof
of constitutive identifiability.

## C. What the TANN and local-closure tests add

The TANN/FCC and inverse-closure campaigns tested whether the inferred
inelastic correction could be turned directly into a local constitutive law.
The registered primary TANN run did not pass its held-out displacement gates
(`median(E_holdout)=1.052`), and the amended full-field trajectory remained an
incomplete solver/law-coupling experiment. The corrected recovery record
explicitly says that no trained scientific TANN model exists.

Those results do **not** show that TANNs cannot work. They show that the
particular formulation in which a kinematically inferred latent correction is
treated as a unique local state function was not established by the available
observables. Constitutive memory, loading path, unobserved through-thickness
organisation and stress-scale information all matter. Learning a variable
that is already non-unique in the inverse problem cannot repair that missing
information.

The FCC/shared-tensor experiments provide the same warning in a more physical
language: a tensorial combination can reproduce a measured field while the
amplitudes of individual slip systems remain unresolved. This page does not
claim that FCC slip is unobservable; it records that the registered kinematics
do not uniquely determine every latent slip contribution.

## The methodological conclusion

The two failure mechanisms must remain separate:

```text
linear field observability
    → a large nullspace makes distinct latent fields indistinguishable

reduced/nonlinear inverse
    → a good local spectrum does not guarantee a unique global latent state
```

Together they support the rule:

> Linear observability is necessary, but not sufficient, for constitutive
> identifiability.

This is why the current programme does not proceed from

```text
kinematics → arbitrary latent field → learned constitutive law
```

Instead it keeps the constitutive model explicit and qualified:

```text
constitutive law
       ↓
qualified mechanical forward
       ↓
observable prediction
       ↓
FEMU sensitivity / SVD
       ↓
identify only observable parameter combinations
```

SRIX, MFront and the native backend are therefore used in a forward problem
whose conventions, plane-stress closure and observation model can be qualified
independently. The parametric SVD then asks a narrower and defensible question:
which combinations of $(\tau_0,R,Q,b)$ change the observed residual? In the
registered records, strong $\tau_0/R$-dominated directions coexist with a weak
$Q-b$ direction. That is a statement about the chosen experiment and
parameterisation, not a claim that every material parameter is recoverable.

## What this authorises

These studies justify:

* using SVD and explicit regularisation before interpreting a fit;
* separating field observability from parametric FEMU observability;
* keeping a constitutive prior or a qualified forward law in the inverse loop;
* reporting latent-field and parameter errors separately from displacement
  agreement;
* treating memory and loading path as part of the state, not as optional input
  decoration.

They do not justify a material claim, a TANN constitutive claim, or a unique
recovery of individual slip amplitudes from the registered displacement data.
The real-data FEMU workflow remains a separate open item, and the full
DIC-weighted parametric SVD is still blocked until its qualified spatial
whitener is available.

## Historical sources absorbed

The conclusions above consolidate the relevant parts of
`validation/tensor_local_inverse_results.md`,
`validation/local_coefficient_inverse_results.md`,
`validation/tann_fcc_primary_run_results.md`,
`validation/tann_fcc_recovery_strategy.md`,
`validation/shared_tensor_generator_preregistration.md` and
`validation/_generated/shared_tensor_generator/tann_fcc_smoke_25.json`.
Those artefacts remain the primary records; this page is the canonical
scientific synthesis, not a replacement for their registered gates.

