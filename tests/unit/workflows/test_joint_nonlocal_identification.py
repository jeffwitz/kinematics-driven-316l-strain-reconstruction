from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.workflows.joint_nonlocal_identification import (
    _analyze_discriminating_f1_design,
    _pareto_indices,
    _pareto_knee,
    _proposed_f2_candidates,
    _quadratic_identifiability_fit,
    _same_physical_point,
    inspect_joint_identification,
    load_joint_identification_config,
    run_low_fidelity,
    screen_frozen_field,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_configuration(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic'\n", encoding="utf-8")
    campaign = tmp_path / "local"
    manifest = {
        "config": {
            "mesh": {
                "nx": 8,
                "ny": 6,
                "base_pixel_size_mm": 0.001,
                "scale_factor": 2.0,
            },
            "material": {
                "first_positive_plastic_strain": 1.0e-6,
            },
            "solver": {"mfront_threads": 1},
            "nonlocal_plasticity": {"enabled": False, "coupling_modulus_mpa": 0.0},
        },
        "inputs": {"synthetic": "same"},
        "layout": {
            "global_shape": [8, 6],
            "partition_shape": [1, 1],
            "padding": 0,
            "partitions": [
                {
                    "partition_id": 0,
                    "index": [0, 0],
                    "core_bounds": [0, 8, 0, 6],
                    "solve_bounds": [0, 8, 0, 6],
                    "core_shape": [8, 6],
                    "solve_shape": [8, 6],
                }
            ],
        },
    }
    manifest_path = campaign / "manifest.json"
    _json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)[:, None]
    peeq = 0.01 + 0.005 * np.cos(x) * np.ones((1, 6))
    field_path = campaign / "partitions" / "0000" / "PEEQ.npy"
    field_path.parent.mkdir(parents=True)
    np.save(field_path, peeq)
    _json(
        field_path.with_name("status.json"),
        {
            "complete": True,
            "partition_id": 0,
            "manifest_sha256": manifest_sha,
            "outputs": {"PEEQ": fingerprint_file(field_path)},
            "diagnostics": {},
        },
    )
    _json(
        campaign / "HREF.json",
        {
            "partition_id": 0,
            "source_campaign_manifest_sha256": manifest_sha,
            "source_peeq_sha256": fingerprint_file(field_path),
            "reference_hardening_modulus_mpa": 4_000.0,
        },
    )
    prepared = tmp_path / "inputs"
    prepared.mkdir()
    prepared_arrays = {
        "displacement_x_mm": np.zeros((9, 7)),
        "displacement_y_mm": np.zeros((9, 7)),
        "yield_stress_mpa": np.ones((8, 6)) * 200.0,
        "hardening_coefficient_mpa": np.ones((8, 6)) * 400.0,
    }
    prepared_outputs: dict[str, object] = {}
    for name, values in prepared_arrays.items():
        path = prepared / f"{name}.npy"
        np.save(path, values)
        prepared_outputs[name] = {
            "filename": path.name,
            "sha256": fingerprint_file(path),
            "shape": list(values.shape),
        }
    _json(prepared / "manifest.json", {"outputs": prepared_outputs})
    config = tmp_path / "configs" / "identification.yaml"
    config.parent.mkdir()
    config.write_text(
        """
campaign:
  name: synthetic
  input: inputs
  output: output
  local_campaign: local
  partition_id: 0
  h_ref: local/HREF.json
  max_new_high_fidelity_runs: 5
  existing_high_fidelity: []
parameters:
  ell_um: {min: 20.0, max: 40.0, samples: 2}
  alpha: {min: 1.0, max: 2.0, samples: 2}
  h_ref_source: campaign_metadata
fidelity:
  low:
    spatial_reduction: 2
    temporal_increments: 2
    minimum_elements_per_ell: 3
    residual_tolerance: 3.0e-6
    sparse_design:
      ell_um: [20.0, 40.0]
      alpha: [1.0, 2.0]
observation:
  grid_mapping: identity
  grid_reduction: 1
  spatial_filter: none
  use_core_mask_only: true
""",
        encoding="utf-8",
    )
    return config


def test_frozen_screen_is_cached_and_uses_one_local_point(tmp_path: Path) -> None:
    config = load_joint_identification_config(_synthetic_configuration(tmp_path))
    inspection = inspect_joint_identification(config)
    assert inspection["f0"]["helmholtz_solves"] == 2
    assert inspection["f0"]["parameter_pair_count"] == 5

    report = screen_frozen_field(config)
    assert report["status"] == "completed"
    with (config.output_directory / "f0" / "frozen_screen.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert sum(row["is_local"] == "True" for row in rows) == 1
    assert all(float(row["helmholtz_residual_relative"]) < 1.0e-11 for row in rows)

    reused = screen_frozen_field(config)
    assert reused["status"] == "reused"
    validation = json.loads(
        (config.output_directory / "f0" / "proxy_validation.json").read_text()
    )
    assert validation["status"] == "not_available"


def test_frozen_screen_dry_run_performs_no_output_write(tmp_path: Path) -> None:
    config = load_joint_identification_config(_synthetic_configuration(tmp_path))
    report = screen_frozen_field(config, dry_run=True)
    assert report["status"] == "dry_run"
    assert not config.output_directory.exists()


def test_low_fidelity_dry_run_preserves_extent_and_reduces_padding(
    tmp_path: Path,
) -> None:
    config = load_joint_identification_config(_synthetic_configuration(tmp_path))
    report = screen_frozen_field(config, dry_run=True)
    assert report["status"] == "dry_run"

    plan = run_low_fidelity(
        config,
        point_selectors=("local", "1:20"),
        dry_run=True,
    )
    assert plan["status"] == "dry_run"
    assert plan["point_count"] == 2
    reduction = plan["reduction"]
    assert reduction["source_element_shape"] == [8, 6]
    assert reduction["reduced_element_shape"] == [4, 3]
    assert reduction["reduced_spacing_mm"] == pytest.approx(0.004)
    assert reduction["physical_extent_preserved"] is True

    design = run_low_fidelity(config, dry_run=True, use_sparse_design=True)
    assert design["point_count"] == 4


def test_low_fidelity_fixed_point_controls_are_explicitly_loaded(
    tmp_path: Path,
) -> None:
    path = _synthetic_configuration(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "    sparse_design:\n",
        """    maximum_newton_iterations: 25
    minimum_step_divisor: 4096
    fixed_point:
      strategy: aitken
      relaxation: 0.2
      minimum_relaxation: 0.05
      maximum_relaxation: 0.8
      residual_growth_factor: 1.25
      maximum_iterations: 50
      record_iteration_history: true
    sparse_design:
""",
    )
    path.write_text(text, encoding="utf-8")

    config = load_joint_identification_config(path)

    assert config.low_maximum_newton_iterations == 25
    assert config.low_minimum_step_divisor == 4096
    assert config.low_nonlocal_relaxation_strategy == "aitken"
    assert config.low_nonlocal_relaxation == pytest.approx(0.2)
    assert config.low_nonlocal_maximum_iterations == 50
    assert config.low_record_iteration_history is True


def test_discriminating_design_unions_replay_saturation_and_constant_a(
    tmp_path: Path,
) -> None:
    path = _synthetic_configuration(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "ell_um: {min: 20.0, max: 40.0, samples: 2}",
        "ell_um: {min: 20.0, max: 60.0, samples: 3}",
    ).replace(
        "alpha: {min: 1.0, max: 2.0, samples: 2}",
        "alpha: {min: 1.0, max: 12.0, samples: 4}",
    )
    text += """
identifiability_design:
  homogeneous_replay_points:
    - {ell_um: 20.0, alpha: 1.0}
    - {ell_um: 40.0, alpha: 2.0}
  saturation:
    ell_um: [20.0, 40.0, 60.0]
    alpha: [6.0, 9.0, 12.0]
    thresholds:
      maximum_relative_amplitude_objective_gain: 0.03
      maximum_correlation_gain: 0.005
      maximum_relative_l2_change: 0.02
      maximum_band_width_relative_degradation: 0.05
  constant_a:
    anchor: {ell_um: 20.0, alpha: 6.0}
    ell_um: [20.0, 30.0, 40.0]
  fixed_alpha:
    alpha: 6.0
    ell_um: [20.0, 40.0, 60.0]
"""
    text = text.replace(
        "    residual_tolerance: 3.0e-6\n",
        "    residual_tolerance: 3.0e-6\n"
        "    maximum_newton_iterations: 25\n"
        "    snapshot_fractions: [0.25, 0.5, 0.75, 1.0]\n",
    )
    path.write_text(text, encoding="utf-8")
    config = load_joint_identification_config(path)

    report = run_low_fidelity(
        config,
        dry_run=True,
        use_identifiability_design=True,
    )

    assert config.low_maximum_newton_iterations == 25
    assert config.low_snapshot_fractions == (0.25, 0.5, 0.75, 1.0)
    assert report["point_count"] == 14
    by_pair = {
        (round(float(point["length_scale_um"]), 6), round(float(point["alpha"]), 6)): point
        for point in report["points"]
        if not bool(point["is_local"])
    }
    assert (20.0, 6.0) in by_pair
    assert set(by_pair[(20.0, 6.0)]["design_roles"]) == {
        "alpha_saturation",
        "constant_a_chi",
        "fixed_alpha_length_discrimination",
    }
    assert (30.0, 2.666667) in by_pair
    constant_a_points = [
        point for point in report["points"] if "constant_a_chi" in point["design_roles"]
    ]
    a_chi = [float(point["a_chi_mpa_mm2"]) for point in constant_a_points]
    assert max(a_chi) == pytest.approx(min(a_chi), rel=1.0e-12)

    rows = []
    for point in report["points"]:
        if bool(point["is_local"]):
            continue
        row = _identification_row(
            fidelity="F1_low",
            ell_um=float(point["length_scale_um"]),
            alpha=float(point["alpha"]),
            amplitude=1.0 / float(point["alpha"]),
            localization=0.5,
        )
        row.update(
            {
                "campaign_id": point["point_id"],
                "design_roles": point["design_roles"],
                "source": str(tmp_path / "missing"),
            }
        )
        rows.append(row)
    analysis = _analyze_discriminating_f1_design(config, rows)
    assert analysis["status"] == "complete"
    assert analysis["constant_a"]["complete"] is True
    assert analysis["constant_a"]["a_chi_relative_spread"] == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert analysis["saturation"]["plateau_reached_for_all_lengths"] is False


def test_pareto_detection_excludes_dominated_points_and_finds_knee() -> None:
    rows = [
        {"j_amplitude": 1.0, "j_localization": 4.0},
        {"j_amplitude": 2.0, "j_localization": 2.0},
        {"j_amplitude": 4.0, "j_localization": 1.0},
        {"j_amplitude": 3.0, "j_localization": 3.0},
    ]
    indices = _pareto_indices(rows)
    assert indices == [0, 1, 2]
    assert _pareto_knee(rows, indices) == 1


def test_configuration_rejects_local_alpha_in_positive_domain(tmp_path: Path) -> None:
    path = _synthetic_configuration(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "alpha: {min: 1.0, max: 2.0, samples: 2}",
        "alpha: {min: 0.0, max: 2.0, samples: 2}",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="start above zero"):
        load_joint_identification_config(path)


def _identification_row(
    *,
    fidelity: str,
    ell_um: float,
    alpha: float,
    amplitude: float,
    localization: float,
) -> dict[str, object]:
    h_ref = 4_000.0
    h_chi = alpha * h_ref
    ell_mm = ell_um / 1_000.0
    a_chi = h_chi * ell_mm**2
    return {
        "fidelity": fidelity,
        "is_local": False,
        "alpha": alpha,
        "h_ref_mpa": h_ref,
        "h_chi_mpa": h_chi,
        "length_scale_um": ell_um,
        "length_scale_mm": ell_mm,
        "a_chi_mpa_mm2": a_chi,
        "a_chi_mpa_um2": h_chi * ell_um**2,
        "theta_h_log_mpa": np.log(h_chi),
        "theta_a_log_mpa_mm2": np.log(a_chi),
        "j_amplitude": amplitude,
        "j_localization": localization,
        "relative_l2": amplitude,
        "correlation": 1.0 - amplitude,
        "absolute_q90_iou": 1.0 - localization,
        "band_width_reference_mm": 0.02,
        "band_width_prediction_mm": 0.02 + 0.0001 * ell_um,
        "band_width_error_mm": 0.0001 * ell_um,
        "band_orientation_error_deg": 0.1 * ell_um,
        "band_axis_offset_mm": 0.001,
        "band_centroid_distance_mm": 0.001,
        "correlation_length_x_reference_mm": 0.1,
        "correlation_length_x_prediction_mm": 0.1 + ell_um / 10_000.0,
        "correlation_length_y_reference_mm": 0.05,
        "correlation_length_y_prediction_mm": 0.05 + ell_um / 20_000.0,
        "spectral_centroid_reference_cycles_per_mm": 10.0,
        "spectral_centroid_prediction_cycles_per_mm": 10.0 + ell_um / 100.0,
        "radial_spectrum_l2": ell_um / 1_000.0,
        "wall_time": 100.0 + 10.0 * alpha,
    }


def test_f2_proposal_is_bounded_deduplicated_and_closes_alpha_boundary() -> None:
    rows = [
        _identification_row(
            fidelity="F2_high",
            ell_um=58.88,
            alpha=4.0,
            amplitude=0.2,
            localization=0.7,
        ),
        *[
            _identification_row(
                fidelity="F1_low",
                ell_um=ell,
                alpha=alpha,
                amplitude=1.2 / alpha / (ell / 20.0),
                localization=0.75 - 0.01 * alpha + 0.001 * abs(ell - 40.0),
            )
            for ell in (20.0, 40.0, 60.0)
            for alpha in (1.0, 3.5, 6.0)
        ],
    ]
    candidates = _proposed_f2_candidates(
        {"rows": rows},
        h_ref_mpa=4_000.0,
        maximum=5,
    )
    assert 1 <= len(candidates) <= 5
    physical = [
        (float(row["length_scale_um"]), float(row["alpha"]))
        for row, _, _ in candidates
    ]
    assert (58.88, 6.0) in physical
    assert len(set(physical)) == len(physical)
    assert (58.88, 4.0) not in physical


def test_quadratic_identifiability_fit_is_explicitly_diagnostic() -> None:
    rows = [
        _identification_row(
            fidelity="F1_low",
            ell_um=ell,
            alpha=alpha,
            amplitude=(np.log(alpha * 4_000.0) - 9.5) ** 2
            + 0.5 * (np.log(alpha * 4_000.0 * (ell / 1_000.0) ** 2) + 1.0) ** 2,
            localization=0.7,
        )
        for ell in (20.0, 40.0, 60.0)
        for alpha in (1.0, 3.5, 6.0)
    ]
    report = _quadratic_identifiability_fit(rows, objective="j_amplitude")
    assert report["status"] == "diagnostic_only"
    assert report["design_rank"] == 6
    assert report["positive_definite"] is True
    assert report["r_squared"] == pytest.approx(1.0)


def test_physical_point_comparison_uses_both_length_and_alpha() -> None:
    reference = _identification_row(
        fidelity="F1_low",
        ell_um=40.0,
        alpha=2.0,
        amplitude=0.5,
        localization=0.7,
    )
    same = dict(reference)
    different_length = {**reference, "length_scale_um": 60.0}
    different_alpha = {**reference, "alpha": 3.0}
    assert _same_physical_point(reference, same)
    assert not _same_physical_point(reference, different_length)
    assert not _same_physical_point(reference, different_alpha)
