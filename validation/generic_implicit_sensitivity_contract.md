# Generic local constitutive sensitivity contract

## Purpose

The coupled Newton pilot needs the derivatives of a local observable `y` and
of the stress with respect to the non-local field. These derivatives must not
be derived from a J2-specific return-mapping formula: the same mechanism must
work for a crystal-plasticity law with an arbitrary number of slip, hardening,
or auxiliary unknowns.

For a converged local implicit system

```text
F(z, q) = 0,
```

where `z` contains every local integration unknown and `q` contains the
external parameters being differentiated, the required sensitivities are

```text
F_z dz/dq = -F_q
dy/dq = y_q + y_z dz/dq.
```

The solve uses the local Jacobian factorisation and several right-hand sides;
it does not form `F_z^-1`. The number or meaning of the components of `z` is
irrelevant. In particular, the same operation applies to a one-variable J2
system and to an 18-variable SRIX system containing elastic strains and slip
increments.

The reusable algebra is implemented in
`fem_inhouse.core.implicit_sensitivities.solve_implicit_sensitivities` and is
covered by a scalar test and a batched 18-unknown/13-observable test.

## Required constitutive export

The host-side adapter needs, at a converged trial point:

```text
F_z                    local implicit Jacobian
F_q                    residual derivatives with respect to q
y_z                    observable derivatives with respect to local unknowns
y_q                    direct observable derivatives with respect to q
```

For the micromorphic pilot, `q` contains the local value of `chi` and the
three imposed in-plane strain components. `y` contains the equivalent
plastic strain and the in-plane stress. Once these blocks are available, the
off-diagonal Newton actions are purely algebraic and require no constitutive
probe at each GMRES operation.

## Crystal-plasticity requirement

The contract deliberately does not request `dg`, `Nss`, Schmid tensors, a
J2 multiplier, or any named hardening variable. A crystal law may therefore
provide a local system with any dimension. Its observable can be accumulated
slip, a slip-family measure, or another explicitly declared scalar/vector;
the sensitivity engine only consumes the derivatives of that observable.

## Current integration boundary

**Superseded on 2026-08-10. Read this section with the correction below.**

The installed MGIS 5.1 interface exposes gradients, thermodynamic forces,
internal and external state variables, and the consistent tangent operator,
but it does not expose the converged local residual/Jacobian or derivatives of
arbitrary observables. Consequently, the current Python pilot cannot obtain
this contract from an arbitrary MFront behaviour without an additional
MFront/TFEL export hook.

### Correction: no export hook is needed

The statement about MGIS is accurate. The conclusion drawn from it is not.

The host never needed `F_z` and `F_q`. It needs `dy/dq`, and MFront already
performs that solve internally: the `ImplicitGenericBehaviour` DSL accepts
several gradient/force pairs, `@TangentOperatorBlocks` names the cross
derivatives wanted, and `getIntegrationVariablesDerivatives_*` obtains them
from the converged implicit Jacobian without refactorising it. MGIS then
publishes them through `tangent_operator_blocks`, which is public API.

Demonstrated on the micromorphic J2 law in
`validation/micromorphic_generic_tangent_blocks.md`: all four blocks match
central finite differences, and the complete coupled linearisation costs 11 %
over a bare integration against roughly `9×` for the finite-difference route.

The generic Python algebra in `implicit_sensitivities` is not wasted — it
remains the right consumer for any law that cannot express its observable as a
declared force, and the finite-difference adapter remains the oracle that
qualifies either route. But the MFront-facing step described below is narrower
than it appeared: it is a behaviour rewrite, not a request upstream.

The finite-difference probes remain the validation fallback. They must not be
presented as the generic implementation. ~~The next MFront-facing step is to
export these blocks from the generated local integration context, or to add a
small DSL contract allowing a behaviour to declare the observable and its
partial derivatives.~~ The next MFront-facing step is to rewrite the behaviour
under `ImplicitGenericBehaviour` and declare the blocks; the DSL contract this
paragraph asked for already exists. It must still be exercised on both SRIX and
Méric–Cailletaud before replacing the probes in the coupled driver, and the
J2 demonstration is tridimensional while production is plane stress.

## Verified MGIS boundary

The installed headers were checked directly under
`/home/jeff/.local/include/MGIS/Behaviour`. `State` stores gradients,
thermodynamic forces, material properties and state-variable arrays, while
`MaterialDataManager` exposes the consistent tangent operator and the
transactional state operations. The public API does not expose the local
implicit residual or its Jacobian, but that is not required when the law is
written as an `ImplicitGenericBehaviour`: MFront performs the local implicit
solve and MGIS returns the requested tangent blocks.

The remaining boundary is the choice of observable. An arbitrary state
observable cannot be differentiated by MGIS merely because it is present in an
internal-variable array. It must be declared as a second generic force, as in
the micromorphic probe, or be handled by the finite-difference oracle. This is
why the generic block mechanism is now demonstrated for J2, while SRIX and
Méric still require their own `ImplicitGenericBehaviour` reformulations before
the production FD probes can be removed.
