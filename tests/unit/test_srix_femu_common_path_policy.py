from scripts.qualify_srix_femu_direct_sensitivity import (
    _oracle_config,
    _path_search_config,
    _reference_config,
    _seed_config,
)


def test_common_path_policies_keep_scientific_defaults_separate() -> None:
    reference = _reference_config()
    seed = _seed_config()
    search = _path_search_config()
    oracle = _oracle_config()

    assert reference.adaptive_stepping_enabled
    assert seed.verify_final_state is False
    assert seed.maximum_newton_iterations == 12
    assert seed.adaptive_step.line_search_difficult_threshold == 0.25
    assert search.adaptive_stepping_enabled is False
    assert search.maximum_newton_iterations == 12
    assert search.gmres_maximum_iterations == 40
    assert oracle.verify_final_state
    assert oracle.maximum_newton_iterations == 80
    assert oracle.maximum_line_search_reductions == 20

