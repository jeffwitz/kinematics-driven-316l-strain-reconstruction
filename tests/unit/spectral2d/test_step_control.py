import pytest

from fem_inhouse.spectral2d import (
    AdaptiveLoadStepController,
    AdaptiveStepConfig,
    LoadStepObservation,
)


def test_controller_grows_easy_steps_without_crossing_path_end() -> None:
    controller = AdaptiveLoadStepController(
        AdaptiveStepConfig(initial_increment_fraction=0.25, maximum_increment_fraction=0.5)
    )
    assert controller.propose(0.0, segment_end=0.3) == pytest.approx(0.25)
    decision = controller.accept(LoadStepObservation(converged=True, newton_iterations=4))
    assert decision.reason == "accepted_easy_step"
    assert decision.next_increment_fraction == pytest.approx(0.375)
    assert controller.propose(0.3, segment_end=1.0) == pytest.approx(0.675)


def test_controller_keeps_normal_steps_and_cuts_back_difficult_steps() -> None:
    controller = AdaptiveLoadStepController(AdaptiveStepConfig(initial_increment_fraction=0.25))
    normal = controller.accept(LoadStepObservation(converged=True, newton_iterations=6))
    assert normal.reason == "accepted_normal_step"
    assert normal.next_increment_fraction == pytest.approx(0.25)
    difficult = controller.accept(
        LoadStepObservation(converged=True, newton_iterations=8, minimum_line_search_factor=0.5)
    )
    assert difficult.reason == "accepted_difficult_step"
    assert difficult.next_increment_fraction == pytest.approx(0.125)


def test_controller_rejects_with_bounded_cutbacks_and_resets_after_acceptance() -> None:
    controller = AdaptiveLoadStepController(
        AdaptiveStepConfig(initial_increment_fraction=0.25, maximum_cutbacks_per_step=2)
    )
    first = controller.reject("newton_failure")
    second = controller.reject("constitutive_failure")
    assert not first.accepted and first.cutbacks_for_current_step == 1
    assert second.next_increment_fraction == pytest.approx(0.0625)
    with pytest.raises(RuntimeError, match="maximum cutbacks"):
        controller.reject()
    controller.accept(LoadStepObservation(converged=True, newton_iterations=6))
    assert controller.cutbacks_for_current_step == 0


def test_controller_rejects_invalid_policy_and_observation() -> None:
    with pytest.raises(ValueError, match="growth factor"):
        AdaptiveStepConfig(increment_growth_factor=1.0)
    controller = AdaptiveLoadStepController(AdaptiveStepConfig())
    with pytest.raises(ValueError, match="converged observation"):
        controller.accept(LoadStepObservation(converged=False, newton_iterations=1))
    with pytest.raises(ValueError, match="path fractions"):
        controller.propose(0.8, segment_end=0.2)
