"""Command-line interface limited to the supported case study."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from fem_inhouse.config import (
    CaseStudyConfig,
    MaterialConfig,
    MeshConfig,
    NonlocalPlasticityConfig,
    SolverConfig,
)
from fem_inhouse.data_preparation import PreparationConfig, prepare_case_study
from fem_inhouse.examples import (
    reduced_biaxial_case,
    save_reduced_example,
    validate_reduced_case,
)
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.postprocessing import (
    FieldAcceptanceThresholds,
    evaluate_field_comparison,
    signed_difference_field,
)
from fem_inhouse.solver import linear_solver_backend, require_pypardiso
from fem_inhouse.workflows import (
    CoupledValidationThresholds,
    PartitionWorkflow,
    collect_identification_results,
    estimate_reference_hardening_from_campaign,
    generate_high_fidelity_manifest,
    generate_joint_identification_report,
    inspect_joint_identification,
    load_decision_thresholds,
    load_joint_identification_config,
    plot_coupled_alpha_fields,
    prepare_transfer_validation,
    profile_coupling_modulus,
    run_low_fidelity,
    run_nonlocality_diagnostic,
    scan_dic_partition_heterogeneity,
    screen_frozen_field,
    select_identification_candidates,
    validate_coupled_nonlocal_campaign,
    write_dic_partition_heterogeneity_report,
)

PARTITION_FIELDS = (
    "U",
    "S",
    "S_3D",
    "E",
    "E_3D",
    "EE_3D",
    "PE",
    "PE_3D",
    "PEEQ",
    "S33_RESIDUAL_MPA",
    "PLANE_STRESS_RESIDUAL_MPA",
    "PEEQ_NONLOCAL",
    "PEEQ_MISMATCH",
    "NONLOCAL_HARDENING_MPA",
    "YIELD_SURFACE_RADIUS_MPA",
    "NONLOCAL_RESIDUAL",
    "RF",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fem-inhouse",
        description="Kinematics-driven 316L case-study reconstruction tools.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable progress logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backend", help="verify and print the sparse solver backend")

    validate = commands.add_parser("validate", help="run the reduced analytical case")
    validate.add_argument("--nx", type=int, default=10)
    validate.add_argument("--ny", type=int, default=10)
    validate.add_argument(
        "--constitutive-backend",
        choices=(
            "python",
            "mfront",
            "mfront-native-plane-stress",
            "mfront-3d-condensed-plane-stress",
        ),
        default="mfront",
    )
    validate.add_argument(
        "--mfront-library",
        default="build/mfront/src/libBehaviour.so",
    )
    validate.add_argument("--mfront-threads", type=int, default=1)

    example = commands.add_parser("example", help="run and save the reduced example")
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--nx", type=int, default=10)
    example.add_argument("--ny", type=int, default=10)
    example.add_argument(
        "--constitutive-backend",
        choices=(
            "python",
            "mfront",
            "mfront-native-plane-stress",
            "mfront-3d-condensed-plane-stress",
        ),
        default="mfront",
    )
    example.add_argument(
        "--mfront-library",
        default="build/mfront/src/libBehaviour.so",
    )
    example.add_argument("--mfront-threads", type=int, default=1)

    prepare = commands.add_parser(
        "prepare-case",
        help="verify and convert the versioned raw DIC data to canonical solver inputs",
    )
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--pixel-size-um", type=float, default=1.84)
    prepare.add_argument("--hardening-scale-mpa", type=float, default=380.0)
    prepare.add_argument("--crop-nx", type=int)
    prepare.add_argument("--crop-ny", type=int)
    prepare.add_argument(
        "--nonfinite-policy",
        choices=("error", "nearest"),
        default="error",
        help="explicit policy for non-finite hardening multipliers",
    )
    prepare.add_argument(
        "--nodal-completion",
        choices=("edge-pad-upper",),
        default="edge-pad-upper",
        help="explicit rule used to obtain the final nodal row and column",
    )

    layout = commands.add_parser("layout", help="write an article partition manifest")
    layout.add_argument("--count", type=int, choices=(25, 100))
    layout.add_argument("--parts-x", type=int)
    layout.add_argument("--parts-y", type=int)
    layout.add_argument("--padding", type=int, default=150)
    layout.add_argument("--output", type=Path, required=True)

    partition = commands.add_parser(
        "partition",
        help="inspect, solve, or stitch the article partition workflow",
    )
    partition.add_argument("--input", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--count", type=int, choices=(25, 100))
    partition.add_argument("--parts-x", type=int)
    partition.add_argument("--parts-y", type=int)
    partition.add_argument("--padding", type=int, default=150)
    partition.add_argument("--base-pixel-mm", type=float, default=0.001)
    partition.add_argument("--scale-factor", type=float, default=1.84)
    partition.add_argument("--young-modulus-mpa", type=float, default=205_000.0)
    partition.add_argument("--poisson-ratio", type=float, default=0.3)
    partition.add_argument("--hardening-exponent", type=float, default=0.245)
    partition.add_argument("--increments", type=int, default=20)
    partition.add_argument("--max-newton-iterations", type=int, default=15)
    partition.add_argument("--residual-tolerance", type=float, default=1e-6)
    partition.add_argument("--minimum-step-divisor", type=int, default=1_024)
    partition.add_argument(
        "--constitutive-backend",
        choices=(
            "python",
            "mfront",
            "mfront-native-plane-stress",
            "mfront-3d-condensed-plane-stress",
        ),
        default="mfront",
    )
    partition.add_argument(
        "--mfront-library",
        default="build/mfront/src/libBehaviour.so",
    )
    partition.add_argument("--mfront-threads", type=int, default=1)
    partition.add_argument("--nonlocal-plasticity", action="store_true")
    partition.add_argument("--nonlocal-length-um", type=float, default=58.88)
    partition.add_argument("--nonlocal-coupling-modulus-mpa", type=float, default=0.0)
    partition.add_argument("--nonlocal-relaxation", type=float, default=0.5)
    partition.add_argument(
        "--nonlocal-relaxation-strategy",
        choices=("fixed", "aitken"),
        default="fixed",
    )
    partition.add_argument("--nonlocal-minimum-relaxation", type=float, default=0.05)
    partition.add_argument("--nonlocal-maximum-relaxation", type=float, default=0.8)
    partition.add_argument(
        "--nonlocal-aitken-residual-growth-factor",
        type=float,
        default=1.25,
    )
    partition.add_argument("--nonlocal-tolerance", type=float, default=1e-6)
    partition.add_argument("--nonlocal-max-iterations", type=int, default=15)
    partition.add_argument(
        "--nonlocal-record-iteration-history",
        action="store_true",
    )
    action = partition.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-pending", action="store_true")
    action.add_argument("--partition-id", type=int)
    action.add_argument("--solve-pending", action="store_true")
    action.add_argument("--stitch", choices=PARTITION_FIELDS)
    partition.add_argument(
        "--field-output",
        type=Path,
        help="optional output path used with --stitch",
    )

    reference = commands.add_parser(
        "estimate-nonlocal-reference",
        help="compute H_ref from a completed local partition core",
    )
    reference.add_argument("--input", type=Path, required=True)
    reference.add_argument("--campaign", type=Path, required=True)
    reference.add_argument("--partition-id", type=int, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
    )
    reference.add_argument("--overwrite", action="store_true")

    validate_coupled = commands.add_parser(
        "validate-coupled-nonlocal",
        help="compare raw local and coupled partition fields with DIC on the core",
    )
    validate_coupled.add_argument("--input", type=Path, required=True)
    validate_coupled.add_argument("--local-campaign", type=Path, required=True)
    validate_coupled.add_argument("--coupled-campaign", type=Path, required=True)
    validate_coupled.add_argument("--partition-id", type=int, required=True)
    validate_coupled.add_argument("--output", type=Path, required=True)
    validate_coupled.add_argument("--overwrite", action="store_true")

    select_dic = commands.add_parser(
        "select-dic-partition",
        help="rank DIC partitions by coherent EVM band morphology",
    )
    select_dic.add_argument("--input", type=Path, required=True)
    select_dic.add_argument("--output", type=Path, required=True)
    select_dic.add_argument("--parts-x", type=int, default=10)
    select_dic.add_argument("--parts-y", type=int, default=10)
    select_dic.add_argument("--padding", type=int, default=150)
    select_dic.add_argument("--overwrite", action="store_true")

    plot_alpha = commands.add_parser(
        "plot-coupled-alpha-fields",
        help="plot raw EVM and PEEQ fields for one local and three coupled campaigns",
    )
    plot_alpha.add_argument("--input", type=Path, required=True)
    plot_alpha.add_argument("--local-campaign", type=Path, required=True)
    plot_alpha.add_argument(
        "--coupled-campaign",
        action="append",
        nargs=2,
        metavar=("ALPHA", "PATH"),
        help="repeat exactly three times; for example: --coupled-campaign 4 results/...-a400",
    )
    plot_alpha.add_argument("--campaign-a050", type=Path)
    plot_alpha.add_argument("--campaign-a100", type=Path)
    plot_alpha.add_argument("--campaign-a200", type=Path)
    plot_alpha.add_argument("--partition-id", type=int, required=True)
    plot_alpha.add_argument("--output", type=Path, required=True)
    plot_alpha.add_argument("--dpi", type=int, default=180)
    plot_alpha.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    plot_alpha.add_argument("--strain-vmax-percentile", type=float)
    plot_alpha.add_argument("--peeq-vmax-percentile", type=float)
    plot_alpha.add_argument("--difference-vmax-percentile", type=float)
    plot_alpha.add_argument("--include-optional-fields", action="store_true")
    plot_alpha.add_argument("--overwrite", action="store_true")

    compare = commands.add_parser(
        "compare-fields",
        help="compare two co-registered fields against pre-declared thresholds",
    )
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--prediction", type=Path, required=True)
    compare.add_argument("--mask", type=Path)
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--difference", type=Path, required=True)
    compare.add_argument("--top-fraction", type=float, default=0.1)
    compare.add_argument("--max-rmse", type=float, required=True)
    compare.add_argument("--max-mae", type=float, required=True)
    compare.add_argument("--min-correlation", type=float, required=True)
    compare.add_argument("--min-localization-iou", type=float, required=True)

    diagnose = commands.add_parser(
        "diagnose-nonlocality",
        help="filter one saved padded partition and compare its spatial width with DIC",
    )
    diagnose.add_argument("--input", type=Path, required=True)
    diagnose.add_argument("--campaign", type=Path, required=True)
    diagnose.add_argument("--partition-id", type=int, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    length_group = diagnose.add_mutually_exclusive_group(required=True)
    length_group.add_argument("--lengths-mm", nargs="+", type=float)
    length_group.add_argument("--lengths-um", nargs="+", type=float)
    length_group.add_argument("--lengths-pixels", nargs="+", type=float)
    diagnose.add_argument("--include-peeq", action="store_true")
    diagnose.add_argument(
        "--mode",
        choices=("exploratory", "confirmatory"),
        default="exploratory",
    )
    diagnose.add_argument("--decision-thresholds", type=Path)
    diagnose.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=(0.05, 0.1, 0.2),
    )
    diagnose.add_argument(
        "--dic-quantiles",
        nargs="+",
        type=float,
        default=(0.8, 0.9, 0.95),
    )
    diagnose.add_argument("--minimum-padding-length-ratio", type=float, default=4.0)
    diagnose.add_argument(
        "--save-fields",
        choices=("all", "best", "none"),
        default="all",
    )
    diagnose.add_argument("--overwrite", action="store_true")

    identify = commands.add_parser(
        "identify-nonlocal",
        help="inspect and run explicit stages of joint ell/H_chi identification",
    )
    identify.add_argument(
        "action",
        choices=(
            "inspect",
            "screen-frozen",
            "run-low-fidelity",
            "profile-h",
            "select-candidates",
            "generate-high-fidelity-manifest",
            "collect-results",
            "report",
            "prepare-transfer-validation",
        ),
    )
    identify.add_argument("--config", type=Path, required=True)
    identify.add_argument("--dry-run", action="store_true")
    identify.add_argument("--workers", type=int, default=1)
    identify.add_argument(
        "--design",
        action="store_true",
        help="run the configured sparse F1 design instead of validation points",
    )
    identify.add_argument(
        "--point",
        action="append",
        help="optional point selector for resumable low-fidelity actions",
    )
    return parser


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _load_partition_field(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing partition input: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {values.shape}")
    return values


def _partition_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve legacy square counts or an explicit rectangular layout."""

    if args.count is not None:
        if args.parts_x is not None or args.parts_y is not None:
            raise ValueError("--count cannot be combined with --parts-x/--parts-y")
        side = 5 if args.count == 25 else 10
        return side, side
    if args.parts_x is None or args.parts_y is None:
        raise ValueError("provide either --count or both --parts-x and --parts-y")
    if args.parts_x < 1 or args.parts_y < 1:
        raise ValueError("--parts-x and --parts-y must be positive")
    return args.parts_x, args.parts_y


def _partition_workflow(args: argparse.Namespace) -> PartitionWorkflow:
    displacement_x = _load_partition_field(args.input, "displacement_x_mm")
    displacement_y = _load_partition_field(args.input, "displacement_y_mm")
    yield_stress = _load_partition_field(args.input, "yield_stress_mpa")
    hardening = _load_partition_field(args.input, "hardening_coefficient_mpa")
    if yield_stress.shape != hardening.shape:
        raise ValueError("material maps must have the same shape")
    nx, ny = yield_stress.shape
    partition_shape = _partition_shape(args)
    config = CaseStudyConfig(
        mesh=MeshConfig(
            nx=nx,
            ny=ny,
            base_pixel_size_mm=args.base_pixel_mm,
            scale_factor=args.scale_factor,
        ),
        material=MaterialConfig(
            young_modulus_mpa=args.young_modulus_mpa,
            poisson_ratio=args.poisson_ratio,
            hardening_exponent=args.hardening_exponent,
        ),
        solver=SolverConfig(
            increments=args.increments,
            max_newton_iterations=args.max_newton_iterations,
            residual_tolerance=args.residual_tolerance,
            minimum_step_divisor=args.minimum_step_divisor,
            constitutive_backend=args.constitutive_backend,
            mfront_library=args.mfront_library,
            mfront_threads=args.mfront_threads,
        ),
        nonlocal_plasticity=NonlocalPlasticityConfig(
            enabled=args.nonlocal_plasticity,
            length_scale_mm=args.nonlocal_length_um / 1_000.0,
            coupling_modulus_mpa=args.nonlocal_coupling_modulus_mpa,
            relaxation=args.nonlocal_relaxation,
            relaxation_strategy=args.nonlocal_relaxation_strategy,
            minimum_relaxation=args.nonlocal_minimum_relaxation,
            maximum_relaxation=args.nonlocal_maximum_relaxation,
            aitken_residual_growth_factor=(
                args.nonlocal_aitken_residual_growth_factor
            ),
            relative_tolerance=args.nonlocal_tolerance,
            maximum_iterations=args.nonlocal_max_iterations,
            record_iteration_history=args.nonlocal_record_iteration_history,
        ),
    )
    return PartitionWorkflow(
        config=config,
        layout=PartitionLayout((nx, ny), partition_shape, padding=args.padding),
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=hardening,
        output_directory=args.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one supported command and return a process exit status."""

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command == "backend":
        require_pypardiso()
        print(linear_solver_backend())
        return 0
    if args.command == "layout":
        layout = PartitionLayout(
            (3_600, 3_100),
            _partition_shape(args),
            padding=args.padding,
        )
        layout.write_manifest(args.output)
        print(args.output)
        return 0
    if args.command == "validate":
        _result, report = validate_reduced_case(
            reduced_biaxial_case(
                nx=args.nx,
                ny=args.ny,
                constitutive_backend=args.constitutive_backend,
                mfront_library=args.mfront_library,
                mfront_threads=args.mfront_threads,
            )
        )
        _print_json(asdict(report))
        return 0 if report.passed else 1
    if args.command == "example":
        report = save_reduced_example(
            args.output,
            nx=args.nx,
            ny=args.ny,
            constitutive_backend=args.constitutive_backend,
            mfront_library=args.mfront_library,
            mfront_threads=args.mfront_threads,
        )
        _print_json(asdict(report))
        return 0 if report.passed else 1
    if args.command == "prepare-case":
        manifest = prepare_case_study(
            args.raw,
            args.output,
            config=PreparationConfig(
                pixel_size_um=args.pixel_size_um,
                hardening_scale_mpa=args.hardening_scale_mpa,
                nonfinite_policy=args.nonfinite_policy,
                nodal_completion=args.nodal_completion,
                crop_nx=args.crop_nx,
                crop_ny=args.crop_ny,
            ),
        )
        _print_json(manifest)
        return 0
    if args.command == "partition":
        workflow = _partition_workflow(args)
        if args.list_pending:
            _print_json({"pending": workflow.pending_partition_ids()})
            return 0
        if args.partition_id is not None:
            _print_json(workflow.solve_partition(args.partition_id))
            return 0
        if args.solve_pending:
            solved = workflow.solve_pending()
            _print_json(
                {
                    "remaining": workflow.pending_partition_ids(),
                    "solved": solved,
                }
            )
            return 0
        output = workflow.stitch(args.stitch, output_path=args.field_output)
        print(output.filename)
        return 0
    if args.command == "estimate-nonlocal-reference":
        reference_report = estimate_reference_hardening_from_campaign(
            input_directory=args.input,
            campaign_directory=args.campaign,
            partition_id=args.partition_id,
            output_path=args.output,
            alpha_values=tuple(args.alphas),
            overwrite=args.overwrite,
        )
        _print_json(asdict(reference_report))
        return 0
    if args.command == "validate-coupled-nonlocal":
        coupled_report = validate_coupled_nonlocal_campaign(
            input_directory=args.input,
            local_campaign_directory=args.local_campaign,
            coupled_campaign_directory=args.coupled_campaign,
            partition_id=args.partition_id,
            output_path=args.output,
            thresholds=CoupledValidationThresholds(),
            overwrite=args.overwrite,
        )
        _print_json(coupled_report)
        return 0 if coupled_report["passed"] else 2
    if args.command == "select-dic-partition":
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
        selection_report = scan_dic_partition_heterogeneity(
            input_directory=args.input,
            parts_x=args.parts_x,
            parts_y=args.parts_y,
            padding=args.padding,
        )
        write_dic_partition_heterogeneity_report(selection_report, args.output)
        _print_json(selection_report)
        return 0
    if args.command == "plot-coupled-alpha-fields":
        coupled_campaigns = (
            tuple((float(alpha), Path(path)) for alpha, path in args.coupled_campaign)
            if args.coupled_campaign is not None
            else None
        )
        plot_report = plot_coupled_alpha_fields(
            input_directory=args.input,
            local_campaign=args.local_campaign,
            coupled_campaigns=coupled_campaigns,
            campaign_a050=args.campaign_a050,
            campaign_a100=args.campaign_a100,
            campaign_a200=args.campaign_a200,
            partition_id=args.partition_id,
            output_directory=args.output,
            dpi=args.dpi,
            formats=args.formats,
            strain_vmax_percentile=args.strain_vmax_percentile,
            peeq_vmax_percentile=args.peeq_vmax_percentile,
            difference_vmax_percentile=args.difference_vmax_percentile,
            include_optional_fields=args.include_optional_fields,
            overwrite=args.overwrite,
        )
        _print_json(plot_report)
        return 0
    if args.command == "compare-fields":
        reference = np.load(args.reference, mmap_mode="r", allow_pickle=False)
        prediction = np.load(args.prediction, mmap_mode="r", allow_pickle=False)
        mask = (
            np.load(args.mask, mmap_mode="r", allow_pickle=False) if args.mask is not None else None
        )
        thresholds = FieldAcceptanceThresholds(
            maximum_rmse=args.max_rmse,
            maximum_mae=args.max_mae,
            minimum_correlation=args.min_correlation,
            minimum_localization_iou=args.min_localization_iou,
        )
        comparison_report = evaluate_field_comparison(
            reference,
            prediction,
            thresholds,
            top_fraction=args.top_fraction,
            mask=mask,
        )
        difference = signed_difference_field(reference, prediction, mask=mask)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.difference.parent.mkdir(parents=True, exist_ok=True)
        report_data = asdict(comparison_report)
        args.report.write_text(
            json.dumps(report_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.save(args.difference, difference)
        _print_json(report_data)
        return 0 if comparison_report.passed else 1
    if args.command == "diagnose-nonlocality":
        if args.mode == "confirmatory" and args.decision_thresholds is None:
            raise ValueError(
                "confirmatory mode requires --decision-thresholds supplied before calculation"
            )
        length_unit: Literal["mm", "um", "pixels"]
        if args.lengths_mm is not None:
            length_values, length_unit = args.lengths_mm, "mm"
        elif args.lengths_um is not None:
            length_values, length_unit = args.lengths_um, "um"
        else:
            length_values, length_unit = args.lengths_pixels, "pixels"
        decision_thresholds = (
            load_decision_thresholds(args.decision_thresholds)
            if args.decision_thresholds is not None
            else None
        )
        nonlocality_report = run_nonlocality_diagnostic(
            input_directory=args.input,
            campaign_directory=args.campaign,
            partition_id=args.partition_id,
            output_directory=args.output,
            length_values=length_values,
            length_unit=length_unit,
            include_peeq=args.include_peeq,
            mode=args.mode,
            decision_thresholds=decision_thresholds,
            top_fractions=args.top_fractions,
            dic_quantiles=args.dic_quantiles,
            minimum_padding_length_ratio=args.minimum_padding_length_ratio,
            save_fields=args.save_fields,
            overwrite=args.overwrite,
        )
        _print_json(nonlocality_report)
        return 0
    if args.command == "identify-nonlocal":
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        identification_config = load_joint_identification_config(args.config)
        if args.action == "inspect":
            _print_json(inspect_joint_identification(identification_config))
            return 0
        if args.action == "screen-frozen":
            _print_json(
                screen_frozen_field(
                    identification_config,
                    dry_run=args.dry_run,
                )
            )
            return 0
        if args.action == "run-low-fidelity":
            low_fidelity_report = run_low_fidelity(
                identification_config,
                point_selectors=tuple(args.point or ()),
                dry_run=args.dry_run,
                maximum_workers=args.workers,
                use_sparse_design=args.design,
            )
            _print_json(low_fidelity_report)
            return 2 if low_fidelity_report.get("failure_count", 0) else 0
        if args.action == "collect-results":
            _print_json(collect_identification_results(identification_config))
            return 0
        if args.action == "profile-h":
            _print_json(profile_coupling_modulus(identification_config))
            return 0
        if args.action == "select-candidates":
            _print_json(select_identification_candidates(identification_config))
            return 0
        if args.action == "generate-high-fidelity-manifest":
            _print_json(
                generate_high_fidelity_manifest(
                    identification_config,
                    dry_run=args.dry_run,
                )
            )
            return 0
        if args.action == "report":
            _print_json(
                generate_joint_identification_report(
                    identification_config,
                    dry_run=args.dry_run,
                )
            )
            return 0
        if args.action == "prepare-transfer-validation":
            _print_json(
                prepare_transfer_validation(
                    identification_config,
                    dry_run=args.dry_run,
                )
            )
            return 0
        raise RuntimeError(
            f"identification action {args.action!r} is not implemented yet; "
            "no calculation was launched"
        )
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover
