# REGM numerical contract

**Mode:** reference  
**Domain:** identification

For a candidate parameter vector ``theta``, the registered equilibrium-gap
screening contract is

$$f(\theta)=B^T W\sigma(\theta),\qquad
\delta u(\theta)=-K_0^{-1}f(\theta),\qquad
r(\theta)=W_D O(\delta u(\theta)).$$

The manifest must record the frozen state, parameter ordering and scales,
reference operator ``K0``, equilibrium weighting ``W``, observation operator
``O``, displacement weighting ``WD``, finite-difference stencil and step, and
the candidate ranking metric. A result is valid only within those choices.

The exact-space twin and observed-DIC transfer are separate evidence claims;
their statuses and metrics are recorded under IDs ``E-SRIX-REGM-001`` and
``E-SRIX-REGM-002`` and must not be conflated.
