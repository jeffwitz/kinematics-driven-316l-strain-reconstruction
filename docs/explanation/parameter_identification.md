# Identifying coupling strength and spatial length

**Category: Explanation.** What experiment can distinguish $H_\chi$ from
$\ell$ without an exhaustive high-fidelity grid?

## The identifiability problem

When the observed spatial frequencies satisfy $\ell k\ll1$, the model responds
mainly through $A_\chi=H_\chi\ell^2$. A valley of nearly equivalent pairs can
then replace a closed optimum. The strategy must test whether changing
$\ell$ produces morphology that cannot be reproduced by changing
$H_\chi$ alone.

## Three fidelities, three claims

```{graphviz}
digraph identification {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef5fb", color="#2980b9"];
  f0 [label="F0\nfrozen-field DCT screen"];
  f1 [label="F1\nreduced coupled mechanics"];
  f2 [label="F2\nhigh-fidelity confirmation"];
  transfer [label="transfer\nunchanged parameters"];
  f0 -> f1 [label="discard weak/redundant"];
  f1 -> f2 [label="select discriminants"];
  f2 -> transfer [label="at most a few candidates"];
}
```

**F0 — frozen-field screening.** Helmholtz is solved for many lengths on an
existing local PEEQ field. One DCT solve per length supports many values of
$H_\chi$ because the mismatch $p-\chi$ is independent of coupling strength
while $p$ is frozen. F0 diagnoses spectral sensitivity and rejects
uninformative regions; it does not predict coupled mechanics.

**F1 — reduced coupled mechanics.** The loading history is replayed on a
coarser but physically consistent grid. F1 ranks candidates, probes saturation
and runs experiments designed to separate parameters. It must reproduce the
ranking of existing high-fidelity cases before it can select new ones.

**F2 — high-fidelity confirmation.** Only a small set of discriminating pairs
is run on the full scientific region. F2 supplies the evidence from which a
scientific conclusion may be drawn. No F2 calculation is launched implicitly.

## Keep three metric families separate

| Family | Main observables | Primary sensitivity |
|---|---|---|
| amplitude | quantiles, standard deviation, RMSE, relative L2 | coupling strength |
| localization | relative and absolute-threshold IoU, active area, band position | placement and support |
| spatial scale | band width, spectrum, correlation length, gradients, total variation | spatial length |

A posteriori weighted sums can hide a bad compromise. Candidate selection
therefore uses Pareto dominance and reports the objectives separately.

## Discriminating experiments

The design asks four questions:

1. **Coupling saturation:** at fixed $\ell$, increase $\alpha$ until amplitude
   gains plateau or band morphology degrades.
2. **Constant $A_\chi$:** compare distant values of $\ell$ with
   $H_\chi\ell^2$ held constant. Identical fields would indicate that only the
   product is observed.
3. **Constant $\alpha$:** compare several lengths at the same coupling
   strength to isolate spatial-scale effects.
4. **Loading snapshots:** compare 25, 50, 75 and 100 percent of loading. A
   length should affect the formation and evolution of bands, not only the
   final image.

The FEM field is passed through the same configured DIC observation operator
before comparison. PEEQ remains a mechanical diagnostic and is never treated
as experimental PEEQ.

## Decision boundary

Separate identification requires an interior optimum or bounded plateau,
measurable discrimination at constant $A_\chi$, curvature in both log-parameter
directions, and unchanged-parameter transfer. If only $A_\chi$ is observable,
the result must be reported as an admissible interval or effective combination,
not as a unique material length.

## Conclusion

> F0 screens, F1 discriminates, F2 confirms, and transfer tests whether the
> result belongs to the material rather than one observed region.

Current outcomes are summarized in {doc}`current_evidence`; commands are in
{doc}`../how-to/run_identification`.
