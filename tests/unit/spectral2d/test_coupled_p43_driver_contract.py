from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_coupled_j2_p43 as coupled_driver  # noqa: E402
from benchmark_coupled_j2_p43 import (  # noqa: E402
    _history_at_time_fractions,
    _load_ebsd_rotations,
    _refine_history,
    _solve_sequence_with_local_cutback,
    _time_increment,
)
from plot_nonlocal_effects_p43 import _validate_crystal_pair  # noqa: E402

from fem_inhouse.spectral2d.grid import StructuredGrid2D  # noqa: E402
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D  # noqa: E402


def test_refinement_preserves_total_time() -> None:
    history = np.arange(9.0)[:, None]
    refined = _refine_history(history, 4)

    assert refined.shape[0] == 33
    assert _time_increment(history) == pytest.approx(1.0 / 8.0)
    assert _time_increment(refined) == pytest.approx(1.0 / 32.0)
    assert _time_increment(refined) * (len(refined) - 1) == pytest.approx(1.0)
    assert _time_increment(refined, 12.0) == pytest.approx(12.0 / 32.0)


def test_explicit_time_path_preserves_dic_knots_and_interpolates() -> None:
    history = np.arange(3.0)[:, None]
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    interpolated = _history_at_time_fractions(history, fractions)

    np.testing.assert_allclose(interpolated[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])


def test_explicit_time_path_rejects_a_missing_dic_knot() -> None:
    with pytest.raises(ValueError, match="preserve DIC knot"):
        _history_at_time_fractions(
            np.arange(3.0)[:, None], np.array([0.0, 0.25, 0.75, 1.0])
        )


def test_ebsd_crop_is_replicated_per_pixel_without_permutation(tmp_path: Path) -> None:
    path = tmp_path / "orientations.h5"
    phi1 = np.array([[0.0, 10.0], [20.0, 30.0]])
    phi = np.array([[5.0, 15.0], [25.0, 35.0]])
    phi2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("orientation")
        group.create_dataset("phi1", data=phi1)
        group.create_dataset("Phi", data=phi)
        group.create_dataset("phi2", data=phi2)

    rotations, provenance = _load_ebsd_rotations(path, (0, 2, 0, 2), states_per_pixel=2)

    assert rotations.shape == (8, 3, 3)
    for pixel in range(4):
        np.testing.assert_allclose(rotations[2 * pixel], rotations[2 * pixel + 1])
    assert not np.allclose(rotations[0], rotations[2])
    assert provenance["unique_orientations"] == 4
    assert len(str(provenance["angles_sha256"])) == 64


def _write_archive(
    path: Path,
    *,
    material: str,
    coupling: float,
    orientation_digest: str = "a" * 64,
) -> None:
    np.savez_compressed(
        path,
        metadata_material=np.asarray(material),
        metadata_coupling_modulus_mpa=np.asarray(coupling),
        metadata_length_scale_mm=np.asarray(0.05888),
        metadata_effective_increments=np.asarray(8),
        metadata_time_increment=np.asarray(0.125),
        metadata_time_history_kind=np.asarray("normalized_pseudo_time"),
        metadata_total_duration=np.asarray(1.0),
        metadata_time_path_sha256=np.asarray("c" * 64),
        metadata_physical_time_history=np.asarray(False),
        metadata_orientation_sha256=np.asarray(orientation_digest),
    )


def test_crystal_plot_pair_requires_same_qualified_setup(tmp_path: Path) -> None:
    local = tmp_path / "local.npz"
    coupled = tmp_path / "coupled.npz"
    _write_archive(local, material="srix", coupling=0.0)
    _write_archive(coupled, material="srix", coupling=5168.0)

    contract = _validate_crystal_pair(local, coupled, "srix")

    assert contract["orientation_sha256"] == "a" * 64


def test_crystal_plot_pair_rejects_different_orientation_maps(tmp_path: Path) -> None:
    local = tmp_path / "local.npz"
    coupled = tmp_path / "coupled.npz"
    _write_archive(local, material="meric", coupling=0.0)
    _write_archive(coupled, material="meric", coupling=5168.0, orientation_digest="b" * 64)

    with pytest.raises(ValueError, match="orientation_sha256 differs"):
        _validate_crystal_pair(local, coupled, "meric")


def test_selective_cutback_preserves_dic_knots_and_regrows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeMaterial:
        def __init__(self) -> None:
            self.reverts = 0

        def revert(self) -> None:
            self.reverts += 1

    material = FakeMaterial()
    monkeypatch.setattr(coupled_driver, "_make_coupled_material", lambda **_kwargs: material)

    accepted_durations: list[float] = []

    def fake_solve(**kwargs: object) -> dict[str, object]:
        duration = float(kwargs["total_duration"])
        if duration > 0.26:
            raise RuntimeError("synthetic large-step failure")
        accepted_durations.append(duration)
        mechanical = np.asarray(kwargs["initial_mechanical"], dtype=float)
        chi = np.asarray(kwargs["initial_chi"], dtype=float)
        return {
            "method": "monolithic",
            "elapsed_seconds": 0.0,
            "newton_iterations": 3,
            "krylov_iterations": [2],
            "krylov_total": 2,
            "outer_iterations": [1],
            "mechanical_iterations": [0],
            "material_tangent_evaluations": 1,
            "material_residual_evaluations": 1,
            "material_tangent_seconds": 0.0,
            "material_residual_seconds": 0.0,
            "final_mechanical_residual_norm": 1.0e-8,
            "final_nonlocal_residual_norm": 1.0e-9,
            "final_mechanical": mechanical,
            "final_chi": chi,
            "final_stress": np.zeros((2, 2, 2, 3)),
            "final_peeq": np.zeros((2, 2)),
            "final_source": np.zeros((2, 2)),
        }

    monkeypatch.setattr(coupled_driver, "_solve_sequence", fake_solve)
    history = np.zeros((3, 3, 3, 2))
    history[1, ..., 0] = 1.0
    history[2, ..., 0] = 3.0
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    progress = tmp_path / "progress.jsonl"

    result = _solve_sequence_with_local_cutback(
        history=history,
        total_duration=1.0,
        progress_path=progress,
        library=Path("unused.so"),
        backend="generic",
        material_model="srix",
        yield_stress=np.ones(4),
        hardening=np.ones(4),
        coupling_modulus_mpa=1.0,
        crystal_rotations=None,
        grid=grid,
        kinematics=TwoSubcellDiagnostic2D(grid),
        method="monolithic",
    )

    fractions = np.asarray(result["accepted_time_fractions"])
    assert np.any(np.isclose(fractions, 0.5))
    assert fractions[-1] == pytest.approx(1.0)
    assert max(accepted_durations) <= 0.26
    assert int(result["local_cutbacks"]) > 0
    assert material.reverts == int(result["local_cutbacks"])
    events = [json.loads(line)["event"] for line in progress.read_text().splitlines()]
    assert "rejected" in events and "accepted" in events
