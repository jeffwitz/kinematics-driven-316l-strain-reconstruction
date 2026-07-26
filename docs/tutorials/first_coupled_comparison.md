# First coupled comparison

**Category: Tutorial.** Compare a local and micromorphic solution without
trying to identify material parameters.

## 1. Start from one small prepared region

Complete {doc}`first_reconstruction` first. Choose padding large enough that
the retained centre is separated from the nonlocal boundary.

## 2. Run a local reference

Use {doc}`../how-to/run_local_reconstruction` to create one local campaign.
Record its mesh, loading and material hashes.

## 3. Add one moderate coupled candidate

Use {doc}`../how-to/run_coupled_reconstruction` with a documented
$\ell>0$ and $H_\chi>0$. Keep every local and numerical setting unchanged.

## 4. Compare four fields

Reconstruct EVM from DIC displacement and from both FEM displacement fields.
Then inspect:

1. DIC EVM;
2. local FEM EVM;
3. coupled FEM EVM;
4. local and coupled PEEQ on a separate scale.

Do not filter the coupled FEM EVM. You should observe that $H_\chi$ changes
how strongly plastic activity is redistributed, while $\ell$ defines the
neighbourhood over which $\chi$ differs from local $p$.

## What you learned

The coupled model changes plastic evolution, not merely the displayed output.
One comparison cannot identify either parameter. Identification requires the
discriminating F0/F1/F2 design in
{doc}`../explanation/parameter_identification`.
