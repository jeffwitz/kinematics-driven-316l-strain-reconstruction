from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fem_inhouse import solver
from fem_inhouse.config import CaseStudyConfig, MeshConfig, SolverConfig


def _fields(nx: int = 2, ny: int = 2):
    nodal = np.zeros((nx + 1, ny + 1))
    return nodal, nodal.copy(), np.full((nx, ny), 250.0), np.full((nx, ny), 500.0)


def _raw_result(nx: int = 2, ny: int = 2):
    return {
        "U": np.zeros((nx + 1, ny + 1, 2)),
        "S": np.zeros((nx, ny, 3)),
        "E": np.zeros((nx, ny, 3)),
        "PE": np.zeros((nx, ny, 3)),
        "PEEQ": np.zeros((nx, ny)),
        "RF": np.zeros((nx + 1, ny + 1, 2)),
        "diagnostics": {
            "backend": "test backend",
            "elapsed_seconds": 0.1,
            "initialization_seconds": 0.01,
            "elastic_assembly_seconds": 0.01,
            "constitutive_seconds": 0.02,
            "tangent_assembly_seconds": 0.02,
            "linear_solve_seconds": 0.03,
            "output_seconds": 0.01,
            "attempted_increments": 2,
            "converged_increments": 2,
            "cutbacks": 0,
            "total_newton_iterations": 4,
            "maximum_newton_iterations": 2,
            "final_residual_norm": 1e-12,
            "final_relative_residual": 1e-8,
            "final_convergence_criterion": "absolute_residual",
        },
        "frames": {
            0.5: {
                "U": np.zeros((nx + 1, ny + 1, 2)),
                "S": np.zeros((nx, ny, 3)),
                "E": np.zeros((nx, ny, 3)),
                "PEEQ": np.zeros((nx, ny)),
            }
        },
    }


def test_backend_reports_pypardiso() -> None:
    solver.require_pypardiso()
    assert solver.linear_solver_backend().startswith("pypardiso")


def test_missing_pypardiso_fails_explicitly(monkeypatch) -> None:
    def fail_import(_name):
        raise ImportError

    monkeypatch.setattr(solver, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="PyPardiso/MKL is required"):
        solver.require_pypardiso()


def test_typed_api_validates_and_forwards_configuration(monkeypatch) -> None:
    config = CaseStudyConfig(
        MeshConfig(nx=2, ny=2),
        solver=SolverConfig(
            increments=7,
            max_newton_iterations=9,
            residual_tolerance=1e-7,
            minimum_step_divisor=64,
        ),
    )
    captured = SimpleNamespace(args=None, kwargs=None)

    def fake_run_fem(*args, **kwargs):
        captured.args = args
        captured.kwargs = kwargs
        return _raw_result()

    monkeypatch.setattr(solver.nonlinear, "run_fem", fake_run_fem)
    ux, uy, yield_map, hardening_map = _fields()
    result = solver.run_case_study(
        config,
        displacement_x_mm=ux,
        displacement_y_mm=uy,
        yield_stress_mpa=yield_map,
        hardening_coefficient_mpa=hardening_map,
        snapshots=(1.0, 0.5),
    )

    assert result.stress_mpa.shape == (2, 2, 3)
    assert result.frames[0.5].displacement_mm.shape == (3, 3, 2)
    assert result.diagnostics is not None
    assert result.diagnostics.backend == "test backend"
    assert result.diagnostics.final_convergence_criterion == "absolute_residual"
    assert captured.args[4] == config.material.hardening_exponent
    assert captured.kwargs["N_inc"] == 7
    assert captured.kwargs["max_nr"] == 9
    assert captured.kwargs["minimum_step_divisor"] == 64
    assert captured.kwargs["first_positive_plastic_strain"] == 1e-6
    assert captured.kwargs["snapshot_fractions"] == (0.5, 1.0)
    assert captured.kwargs["hardening"] == "tabular"


@pytest.mark.parametrize(
    ("field_index", "replacement", "message"),
    [
        (0, np.zeros((2, 2)), "displacement_x_mm has shape"),
        (1, np.full((3, 3), np.nan), "displacement_y_mm contains non-finite"),
        (2, np.zeros((2, 2)), "yield_stress_mpa must be strictly positive"),
        (3, np.full((2, 2), -1.0), "hardening_coefficient_mpa must be nonnegative"),
    ],
)
def test_invalid_input_fields_are_rejected(
    monkeypatch,
    field_index,
    replacement,
    message,
) -> None:
    monkeypatch.setattr(solver.nonlinear, "run_fem", lambda *args, **kwargs: _raw_result())
    fields = list(_fields())
    fields[field_index] = replacement

    with pytest.raises(ValueError, match=message):
        solver.run_case_study(
            CaseStudyConfig(MeshConfig(nx=2, ny=2)),
            displacement_x_mm=fields[0],
            displacement_y_mm=fields[1],
            yield_stress_mpa=fields[2],
            hardening_coefficient_mpa=fields[3],
        )


@pytest.mark.parametrize("snapshots", [(0.0,), (1.1,), (np.nan,), (0.5, 0.5)])
def test_invalid_snapshot_fractions_are_rejected(snapshots) -> None:
    ux, uy, yield_map, hardening_map = _fields()
    with pytest.raises(ValueError, match="snapshot fractions"):
        solver.run_case_study(
            CaseStudyConfig(MeshConfig(nx=2, ny=2)),
            displacement_x_mm=ux,
            displacement_y_mm=uy,
            yield_stress_mpa=yield_map,
            hardening_coefficient_mpa=hardening_map,
            snapshots=snapshots,
        )


def test_non_finite_solver_output_is_rejected(monkeypatch) -> None:
    raw = _raw_result()
    raw["S"][0, 0, 0] = np.nan
    monkeypatch.setattr(solver.nonlinear, "run_fem", lambda *args, **kwargs: raw)
    ux, uy, yield_map, hardening_map = _fields()

    with pytest.raises(RuntimeError, match="non-finite final fields"):
        solver.run_case_study(
            CaseStudyConfig(MeshConfig(nx=2, ny=2)),
            displacement_x_mm=ux,
            displacement_y_mm=uy,
            yield_stress_mpa=yield_map,
            hardening_coefficient_mpa=hardening_map,
        )


def test_solver_can_skip_backend_requirement_for_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(solver.nonlinear, "run_fem", lambda *args, **kwargs: _raw_result())

    def unexpected_check():
        raise AssertionError("backend requirement should have been skipped")

    monkeypatch.setattr(solver, "require_pypardiso", unexpected_check)
    ux, uy, yield_map, hardening_map = _fields()
    result = solver.run_case_study(
        CaseStudyConfig(
            MeshConfig(nx=2, ny=2),
            solver=SolverConfig(require_pypardiso=False),
        ),
        displacement_x_mm=ux,
        displacement_y_mm=uy,
        yield_stress_mpa=yield_map,
        hardening_coefficient_mpa=hardening_map,
    )
    assert result.stress_mpa.shape == (2, 2, 3)
