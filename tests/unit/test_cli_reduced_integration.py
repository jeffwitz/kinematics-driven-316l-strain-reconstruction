"""CLI coverage for reduced-integration campaign controls."""

from fem_inhouse.cli import _parser


def test_partition_cli_exposes_cps4r_controls() -> None:
    args = _parser().parse_args(
        [
            "partition",
            "--input",
            "inputs",
            "--output",
            "campaign",
            "--count",
            "25",
            "--element-formulation",
            "cps4r",
            "--hourglass-scale",
            "0.25",
            "--hourglass-energy-warning-ratio",
            "0.02",
            "--hourglass-energy-failure-ratio",
            "0.05",
            "--list-pending",
        ]
    )

    assert args.element_formulation == "cps4r"
    assert args.hourglass_scale == 0.25
    assert args.hourglass_energy_warning_ratio == 0.02
    assert args.hourglass_energy_failure_ratio == 0.05


def test_partition_cli_keeps_cps4_as_the_default() -> None:
    args = _parser().parse_args(
        [
            "partition",
            "--input",
            "inputs",
            "--output",
            "campaign",
            "--count",
            "25",
            "--list-pending",
        ]
    )

    assert args.element_formulation == "cps4"
    assert args.hourglass_scale == 1.0
