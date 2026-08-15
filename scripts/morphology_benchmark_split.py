#!/usr/bin/env python3
"""The one split definition both benchmarks import.

A comparison in which POD and the network see different states is not a
comparison, so the split lives in one place rather than being restated twice.

## Why the holdout is spread over the loading and not taken from its end

The earlier choice, training on states 1 to 30 and testing on 31 to 40,
answered the wrong question. The strongest plastic morphologies appear late, so
it asked a model to learn a family of shapes while withholding the richest part
of that family, then judged it on exactly that part. A failure would have meant
out-of-distribution extrapolation, not that the convolutional representation is
uncompact -- two different questions collapsed into one number.

This milestone asks only whether the observed morphologies admit a compact
representation. Training must therefore span every mechanical regime, heavy
plasticity included, and the held-out states must be drawn from all of them.

The holdout is made of two-state blocks rather than isolated states because
consecutive states are strongly correlated: a lone state 25 is nearly the
average of 24 and 26, and interpolating it would prove nothing. A block forces
the model across two missing increments.

State 1 to 30 against 31 to 40 keeps its place later, as a predictive test of
an identified mechanical law once a tensor decoder, equilibrium and
thermodynamics are in the loop. It is not a criterion for choosing between two
reduced representations.
"""

from __future__ import annotations

#: Two-state blocks spread over the early, intermediate, yielding and heavily
#: plastic parts of the loading.
TEMPORAL_HOLDOUT = (8, 9, 18, 19, 28, 29, 37, 38)


def split_states(states: list[int]) -> tuple[list[int], list[int]]:
    """Positions within `states` for training and for the temporal holdout."""

    train = [index for index, state in enumerate(states) if state not in TEMPORAL_HOLDOUT]
    test = [index for index, state in enumerate(states) if state in TEMPORAL_HOLDOUT]
    if not test:
        raise ValueError("the temporal holdout is empty for this state list")
    return train, test
