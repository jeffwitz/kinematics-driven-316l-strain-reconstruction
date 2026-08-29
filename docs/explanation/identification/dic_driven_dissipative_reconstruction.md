# DIC-driven dissipative plastic reconstruction

## Question

Can the plastic correction required by measured kinematics be reconstructed
before choosing a constitutive law?  This method sits between two familiar
extremes:

```text
prescribed constitutive model  ->  FEMU parameter update
fully flexible learned law    ->  constitutive training
```

It first asks what an observed mechanical defect requires from a plastic
field, then imposes only minimal physical admissibility.  It is therefore not
a conventional reduced-order model or a claim that the reconstructed field is
the material's unique internal state.

## Data-guided modes

At each increment, use a mechanical reference solution with the same boundary
conditions and define the observed defect schematically as

```{math}
r_n=\varepsilon_{\mathrm{DIC},n}-\varepsilon_{\mathrm{ref},n}.
```

Represent a correction in a reduced field space,

```{math}
\varepsilon^p_n=\Phi a_n,
\qquad
\varepsilon_{\mathrm{sim},n}
=\varepsilon_{\mathrm{ref},n}+A\Phi a_n.
```

The modes are generated conceptually from a block-Krylov construction,

```{math}
\Phi\sim K_r(A^Tr),
```

where $A$ maps a plastic/eigenstrain perturbation to the observed mechanical
field and $A^T$ maps an observable defect back to mechanically effective field
directions.  Thus the modes are not selected by a material map or an a priori
POD: they are directions capable of changing the measured defect through
equilibrium.  This is a direct use of the qualified full-field operator and
adjoint described in {doc}`../spectral_mechanics/plastic_inverse_reuse`.

```text
DIC / mechanical defect
          |
          v
         A^T
          |
          v
  observable plastic modes
          |
          v
   reduced reconstruction
          |
          v
positive-dissipation projection
          |
          v
   dissipative plastic history
          |
       +--+----------------+
       v                   v
 FCC slip coordinates   convex-potential
       |                 compatibility
       +--------+----------+
                v
      constitutive hypotheses
```

## A good fit can still be unphysical

The registered rank-16 comparison makes the distinction concrete:

| reconstruction | held-out defect error $E$ |
|---|---:|
| raw Krylov | $\simeq 0.245$ |
| dissipatively projected Krylov | $\simeq 0.402$ |
| learned dissipative generator | $\simeq 0.587$ |
| J2-imposed direction | $\simeq 0.855$ |

The raw Krylov fit is excellent, but about 47% of its plastic power is
negative.  Its kinematic success is therefore partly obtained through
directions that are not physically dissipative.  This is a direct example of

```text
field fit != physical plastic history.
```

## Dissipative projection

For each local increment, project onto the half-space

```{math}
\sigma:\Delta\varepsilon^p\ge 0.
```

The registered projection uses the declared plastic metric $G_p$; the exact
metric details are kept in `validation/krylov_projected_control_results.md`.
At rank 16, predictor-stress negative power is zero to numerical precision
($D_-^{\mathrm{pred}}\approx5\times10^{-17}$), while the projected equivalent
plastic amplitude remains about $1.01$ times the raw value.  The projection
therefore redirects the field rather than suppressing its plastic magnitude.

This correction is not a constitutive law.  Approximately $f_0\simeq0.47$ of
the plastically active points lie on the zero-work boundary
$\sigma:\Delta\varepsilon^p\simeq0$.  Consequently,

```text
kinematics + equilibrium + D >= 0
still do not select a constitutive direction.
```

Positive dissipation removes anti-dissipative directions, but the material law
must still select a direction inside the admissible half-space.  A dissipative
direction can still be materially incorrect.

## From a history to a law hypothesis

The reduced history contains increments $\Delta a_n$ and state-like
coordinates $X_n$.  A possible next step is to test whether these pairs are
compatible with a common convex dissipative potential, for example through
cyclic monotonicity.  The recorded work establishes this as a route for
constitutive-law discovery; it does not establish that a complete law has
already been identified.  History and path dependence remain part of the
question.

## From plastic strain to FCC coordinates

Using the declared EBSD orientation and specimen-frame Schmid tensors, write

```{math}
\Delta\varepsilon^p
=\sum_{\alpha=1}^{12}\Delta\gamma^\alpha P^\alpha.
```

Under the currently registered EBSD orientation assignment, the
observable-projected increment has FCC representation error
$e_{\mathrm{FCC}}\simeq2\times10^{-12}$, effectively exact, whereas the raw
field has $e_{\mathrm{FCC}}\simeq0.353$.  The observable correction therefore
lies in the local FCC tensor span even though the unfiltered latent field does
not.  This is a geometric compatibility result under the registered mapping,
not independent validation of the physical DIC--EBSD co-registration.

Imposing thermodynamic sign compatibility system by system raises the recorded
median error to about $0.546$ (represented share $\rho\simeq0.836$).  The
correct wording is therefore **FCC-compatible slip activities** or
**2-D-projection-compatible slip decomposition**, never “true slips
recovered.”  The decomposition is a kinematic representation subject to a
thermodynamic test, not an experimental identification of individual systems.

Resolved shear does organise these activities: Spearman correlation between
$\tau^\alpha$ and $\Delta\gamma^\alpha$ is about $0.70$--$0.83$ by system and
$0.76$ pooled.  The relation is nevertheless broad, changes with increment and
does not define a unique local constitutive law.

## Status and boundaries

### Demonstrated

- data-guided modal construction through $A^Tr$;
- substantial reduced reconstruction of the observed defect;
- positive predictor-stress dissipation after projection, at roundoff;
- preservation of plastic amplitude under the projection;
- machine-precision FCC-span compatibility of the observable increment;
- strong organisation of FCC-compatible activities by resolved shear.

### Not demonstrated

- uniqueness of the reconstructed plastic field or history;
- recovery of experimentally true slip activities;
- identification of a constitutive law from this history;
- independently proven P43 DIC--EBSD physical registration;
- transfer to multiple curated experimental cases.

The primary records are `validation/krylov_projected_control_results.md` and
`validation/fcc_slip_decomposition_results.md`, with the reduced-basis and
learned-flow context in `validation/adaptive_reduced_basis_learned_flow.md`.
They document a method and its limits, not a completed P43 material claim.

## Position in the inverse architecture

FEMU starts with a law and asks which parameters explain the observations.
This reconstruction starts with the observations and asks which mechanically
effective, dissipative plastic corrections they permit:

```text
FEMU: law -> parameters -> predicted fields -> comparison

This method: fields -> admissible plastic correction -> candidate law
```

It is therefore a constructive intermediate layer between the qualified
mechanical/adjoint core and constitutive validation.  Its next scientific test
requires better-curated DIC, EBSD and loading histories, not another attempt to
optimise the present P43 record.
