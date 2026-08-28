# REGM screening and its present boundary

**Mode:** explanation  
**Domain:** identification

The reconditioned equilibrium-gap method (REGM) is a pre-screening tool. A
digital twin can rank candidates when the mechanical field is directly
available, but the current DIC observation operator can destroy the transfer
of that ranking. This is a scientifically useful negative result: REGM is not
currently a validated replacement for full-field FEMU on the measured chain.

REGM replays a constitutive candidate into a reference equilibrium operator:

$$f(\theta)=B^TW\sigma(\theta),\qquad
\delta u(\theta)=-K_0^{-1}f(\theta),$$

then compares the observable prediction

$$r(\theta)=W_D O(\delta u(\theta)).$$

In exact mechanical space the registered ranking is strong (Spearman
\(0.866\), log-Pearson \(0.878\)). After the DIC operator it falls to
Spearman \(0.326\) and log-Pearson \(0.276\). The current chain is therefore
a NO-GO for replacing FEMU. Placement, POD upper bounds, mechanical
projection, information geometry, algorithmic tangents and sequential
corrections were retained as diagnostics; none restores the missing transfer.

Claim boundaries are indexed in {doc}`../../reference/evidence/claims_matrix`
and the reproduction workflow is in
{doc}`../../how-to/identification/run_regm_screening`.
