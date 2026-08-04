import numpy as np
import pytest

from fem_inhouse.config import SolverConfig
from fem_inhouse.core.global_broyden import GlobalInverseBroyden


def test_global_broyden_uses_a_secant_direction() -> None:
    accelerator = GlobalInverseBroyden(memory=2)
    accelerator.begin_increment()
    accelerator.observe([0.0, 0.0], [2.0, 0.0])
    accelerator.observe([1.0, 0.0], [1.0, 0.0])

    direction = accelerator.direction(
        [-1.0, 1.0],
        [1.0, 0.0],
        lambda columns: columns,
    )

    assert np.allclose(direction, [1.0, 1.0])
    assert accelerator.diagnostics["global_broyden_directions_used"] == 1.0


def test_global_broyden_rejects_an_unbounded_direction() -> None:
    accelerator = GlobalInverseBroyden(memory=1, maximum_step_factor=1.1)
    accelerator.observe([0.0], [1.0])
    accelerator.observe([1.0], [0.9])

    direction = accelerator.direction([1.0], [-100.0], lambda columns: columns)

    assert np.array_equal(direction, [1.0])
    assert accelerator.diagnostics["global_broyden_directions_rejected"] == 1.0


def test_global_broyden_requires_line_search() -> None:
    with pytest.raises(ValueError, match="newton_line_search"):
        SolverConfig(
            element_formulation="cps4r_as",
            jacobian_correction="global_broyden",
        )

    config = SolverConfig(
        element_formulation="cps4r_as",
        jacobian_correction="global_broyden",
        newton_line_search=True,
    )
    assert config.jacobian_correction == "global_broyden"
