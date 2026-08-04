import numpy as np
import pytest

from fem_inhouse.spectral2d import DisplacementAndersonAccelerator


def test_anderson_starts_with_fixed_point_and_then_accelerates() -> None:
    accelerator = DisplacementAndersonAccelerator(memory=4)
    state = np.array([0.0, 0.0])
    image = np.array([1.0, 2.0])
    first = accelerator.propose(state, image, image - state)
    np.testing.assert_allclose(first, image)

    second_state = image
    second_image = np.array([1.5, 2.5])
    second = accelerator.propose(second_state, second_image, second_image - second_state)
    assert second.shape == image.shape
    assert accelerator.diagnostics.proposals == 2
    assert accelerator.diagnostics.accelerated_proposals == 1


def test_anderson_reset_discards_history() -> None:
    accelerator = DisplacementAndersonAccelerator()
    accelerator.propose([0.0], [1.0], [1.0])
    accelerator.reset()
    proposal = accelerator.propose([1.0], [2.0], [1.0])
    np.testing.assert_allclose(proposal, [2.0])
    assert accelerator.diagnostics.resets == 1


def test_anderson_rejects_inconsistent_or_nonfinite_inputs() -> None:
    accelerator = DisplacementAndersonAccelerator()
    with pytest.raises(ValueError, match="sizes"):
        accelerator.propose([0.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        accelerator.propose([0.0], [np.inf], [1.0])
