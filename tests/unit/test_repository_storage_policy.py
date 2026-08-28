from scripts.check_repository_storage import is_golden


def test_only_explicit_golden_paths_are_whitelisted() -> None:
    assert is_golden("validation/golden/srix_m20.npz")
    assert not is_golden("validation/reference_data/srix_m20.npz")
