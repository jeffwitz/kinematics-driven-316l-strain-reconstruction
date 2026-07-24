from fem_inhouse.core import solver_legacy


def test_historical_biaxial_verification(capsys) -> None:
    solver_legacy._verify()
    output = capsys.readouterr().out
    assert "PASSED" in output
