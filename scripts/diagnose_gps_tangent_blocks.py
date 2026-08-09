"""Block-by-block comparison of the GPS and reference local Jacobians.

CdC section 12: the causal localisation showed ONE material point (point 96,
pixel (8,2), subcell 0) carries the whole 52-vs-46 penalty, and test E found
the tangents still differ by 1.9e-3 at the same internal variables. This
script compares the two LOCAL Jacobians block by block at the checkpoint
state, with a COMPLETE state transplant -- internal variables by name AND the
committed gradient rotated into the recipient's frame (the reference stores
it in the crystal frame, the GPS in the global frame).

For each of the top points (96, 95, 59) at the checkpoint increment 6:

1. the raw 6x6 Kelvin tangents of both formulations evaluated from their
   OWN committed states, and from the SAME transplanted complete state;
2. the four blocks C_aa, C_ab, C_ba, C_bb (a = in-plane [0,1,3],
   b = transverse [2,4,5]) of both tangents, in the global frame;
3. the plane-stress Schur complement of the reference tangent against the
   GPS projected tangent -- are the two formulations algebraically the
   same object at the same state, or do they differ?

Usage:

    .venv/bin/python scripts/diagnose_gps_tangent_blocks.py \
        --output validation/_generated/performance/gps_tangent_blocks_same_state_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
GPS = "mfront-native-generalised-plane-stress"
REFERENCE = "mfront-3d-condensed-plane-stress"
TOP_POINTS = (96, 95, 59)
_PLANE = (0, 1, 3)
_TRANSVERSE = (2, 4, 5)
_SQRT_TWO = np.sqrt(2.0)


def _kelvin_scale() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([1.0, 1.0, 1.0, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO]),
        np.array([1.0, 1.0, 1.0, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO]),
    )


def _rotate_gradient_to_crystal(
    gradient_global_kelvin: np.ndarray,
    q_global_to_material: np.ndarray,
) -> np.ndarray:
    """Rotate a 6-vector Kelvin gradient from the global to the crystal frame."""

    from fem_inhouse.core.tensor_reconstruction import kelvin_3d_to_tensor, tensor_to_kelvin_3d

    tensor = kelvin_3d_to_tensor(gradient_global_kelvin, quantity="strain")
    rotated = np.einsum("ij,jk,lk->il", q_global_to_material, tensor, q_global_to_material)
    return tensor_to_kelvin_3d(rotated, quantity="strain")


def _rotate_gradient_to_global(
    gradient_crystal_kelvin: np.ndarray,
    q_global_to_material: np.ndarray,
) -> np.ndarray:
    """Rotate a 6-vector Kelvin gradient from the crystal to the global frame."""

    from fem_inhouse.core.tensor_reconstruction import kelvin_3d_to_tensor, tensor_to_kelvin_3d

    tensor = kelvin_3d_to_tensor(gradient_crystal_kelvin, quantity="strain")
    rotated = np.einsum("ji,jk,kl->il", q_global_to_material, tensor, q_global_to_material)
    return tensor_to_kelvin_3d(rotated, quantity="strain")


def _load_case(arguments: argparse.Namespace):
    from scripts.benchmark_tri2_j2_krylov import _load_case as load

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = load(mesh, arguments.crop_nodes)
    return grid, yield_stress, coefficient, boundary


def _build_material(backend: str, arguments: argparse.Namespace, grid, yield_stress, coefficient):
    from scripts.diagnose_gps_tangent_localisation import _build_material as build

    return build(backend, arguments, grid, yield_stress, coefficient)


def _run_backend(backend, arguments, grid, yield_stress, coefficient, boundary):
    from scripts.diagnose_gps_tangent_localisation import _run_backend as run

    return run(backend, arguments, grid, yield_stress, coefficient, boundary)


def _checkpoint_calls(recording, increment: int) -> list[dict[str, object]]:
    from scripts.diagnose_gps_tangent_localisation import _checkpoint_calls as calls

    return calls(recording, increment)


def _evaluate_raw(
    material: object,
    snapshot: object,
    strain: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """Evaluate one backend from a snapshot; return the raw 3D quantities."""

    material.restore_state(snapshot)
    trial = material.evaluate(
        strain.reshape(-1, 3),
        time_increment=dt,
        consistent_tangent=True,
    )
    return {
        "trial": trial,
        "stress_in_plane": np.asarray(trial.stress_in_plane_mpa, dtype=float).copy(),
        "tangent_in_plane": np.asarray(trial.tangent_in_plane_mpa, dtype=float).copy(),
    }


def _native(obj: object) -> object:
    return getattr(obj, "_bridge", None) or obj


def _committed_point_state(
    material: object,
    snapshot: object,
    point: int,
) -> dict[str, np.ndarray]:
    """Export one point's complete committed state from a snapshot."""

    from fem_inhouse.core.mfront import _declared_internal_slices

    native = _native(material)
    manager = native._manager
    material.restore_state(snapshot)
    slices = _declared_internal_slices(
        native._mgis,
        native._behaviour,
        native._mgis.Hypothesis.Tridimensional,
        native._specification,
    )
    isv = np.asarray(manager.s0.internal_state_variables)[point, :]
    exported: dict[str, np.ndarray] = {
        "gradient": np.asarray(manager.s0.gradients)[point, :].copy(),
        "forces": np.asarray(manager.s0.thermodynamic_forces)[point, :].copy(),
    }
    for name, position in slices.items():
        exported[name] = isv[position].copy()
    return exported


def _bridge_snapshot(snapshot: object) -> object:
    """Return the 3-D snapshot carried by either backend."""

    return getattr(snapshot, "bridge", snapshot)


def _snapshot_arrays(snapshot: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract gradient, ISV, force and global-gradient arrays from a snapshot."""

    bridge = _bridge_snapshot(snapshot)
    if isinstance(bridge, tuple):
        return (
            np.asarray(bridge[0], dtype=float),
            np.asarray(bridge[1], dtype=float),
            np.asarray(bridge[2], dtype=float),
            np.asarray(bridge[3], dtype=float),
        )
    return (
        np.asarray(bridge.gradients_s0, dtype=float),
        np.asarray(bridge.internal_state_variables_s0, dtype=float),
        np.asarray(bridge.thermodynamic_forces_s0, dtype=float),
        np.asarray(bridge.committed_global_strain, dtype=float),
    )


def _copy_shared_isvs(
    source_material: object,
    source_snapshot: object,
    target_material: object,
    target_isv: np.ndarray,
    point: int,
) -> None:
    """Copy shared ISVs by declared name, never by assumed offsets."""

    from fem_inhouse.core.mfront import _declared_internal_slices

    source_native = _native(source_material)
    target_native = _native(target_material)
    source_slices = _declared_internal_slices(
        source_native._mgis,
        source_native._behaviour,
        source_native._mgis.Hypothesis.Tridimensional,
        source_native._specification,
    )
    target_slices = _declared_internal_slices(
        target_native._mgis,
        target_native._behaviour,
        target_native._mgis.Hypothesis.Tridimensional,
        target_native._specification,
    )
    source_isv = _snapshot_arrays(source_snapshot)[1]
    for name, target_slice in target_slices.items():
        if name in source_slices:
            source_slice = source_slices[name]
            if (target_slice.stop - target_slice.start) != (source_slice.stop - source_slice.start):
                raise ValueError(f"incompatible shared ISV size for {name}")
            target_isv[point, target_slice] = source_isv[point, source_slice]


def _make_transplanted_snapshot(
    material: object,
    target_snapshot: object,
    source_material: object,
    source_snapshot: object,
    point: int,
    *,
    q_global_to_material: np.ndarray,
) -> object:
    """Build a new target snapshot containing one physical source state.

    The returned object is the only state subsequently passed to an evaluator.
    In particular, no helper is allowed to restore the pre-transplant target
    snapshot after this function has been called.
    """

    from fem_inhouse.core.mfront import MFrontMaterialStateSnapshot

    target_bridge = _bridge_snapshot(target_snapshot)
    target_grad, target_isv0, target_forces, target_global = _snapshot_arrays(target_snapshot)
    source_grad, _, source_forces, source_global = _snapshot_arrays(source_snapshot)
    gradients = target_grad.copy()
    isv = target_isv0.copy()
    forces = target_forces.copy()
    committed_global = target_global.copy()

    source_native = _native(source_material)
    target_native = _native(material)
    source_is_gps = source_native.__class__.__name__ == "MFrontNativeGeneralisedPlaneStressBatch"
    target_is_gps = target_native.__class__.__name__ == "MFrontNativeGeneralisedPlaneStressBatch"
    physical_global = source_global[point].copy()
    if source_is_gps:
        gradient_target = (
            physical_global.copy()
            if target_is_gps
            else _rotate_gradient_to_crystal(physical_global, q_global_to_material)
        )
    else:
        source_crystal = source_grad[point]
        physical_global = _rotate_gradient_to_global(source_crystal, q_global_to_material)
        gradient_target = (
            _rotate_gradient_to_crystal(physical_global, q_global_to_material)
            if not target_is_gps
            else physical_global.copy()
        )
    gradients[point, :] = gradient_target
    committed_global[point, :] = physical_global

    # MGIS thermodynamic forces are stored in the material frame in both
    # bridges.  The two behaviours use the same material orientation here.
    forces[point, :] = source_forces[point]
    _copy_shared_isvs(source_material, source_snapshot, material, isv, point)

    if isinstance(target_bridge, tuple):
        values = list(target_bridge)
        values[0] = gradients
        values[1] = isv
        values[2] = forces
        values[3] = committed_global
        values[4] = np.asarray(values[4], dtype=float).copy()
        values[4][point, :] = physical_global[list(_TRANSVERSE)]
        if values[5] is not None:
            values[5] = np.asarray(values[5], dtype=float).copy()
            values[5][point, :] = physical_global[list(_PLANE)]
        return tuple(values)

    bridge = MFrontMaterialStateSnapshot(
        gradients_s0=gradients,
        internal_state_variables_s0=isv,
        thermodynamic_forces_s0=forces,
        committed_global_strain=committed_global,
        committed_nonlocal_values=target_bridge.committed_nonlocal_values,
    )
    if hasattr(target_snapshot, "accepted_transverse"):
        from fem_inhouse.core.mfront import MFrontCondensedStateSnapshot

        accepted = np.asarray(target_snapshot.accepted_transverse, dtype=float).copy()
        accepted[point, :] = physical_global[list(_TRANSVERSE)]
        accepted_in = (
            None
            if target_snapshot.accepted_in_plane is None
            else np.asarray(target_snapshot.accepted_in_plane, dtype=float).copy()
        )
        if accepted_in is not None:
            accepted_in[point, :] = physical_global[list(_PLANE)]
        return MFrontCondensedStateSnapshot(
            bridge=bridge,
            accepted_transverse=accepted,
            latest_transverse=None,
            has_accepted_global_trial=False,
            last_in_plane=None,
            last_time_increment=None,
            accepted_in_plane=accepted_in,
            accepted_cbb=target_snapshot.accepted_cbb,
            accepted_cba=target_snapshot.accepted_cba,
        )
    return bridge


def _assert_same_physical_committed_state(
    gps_material: object,
    gps_snapshot: object,
    ref_material: object,
    ref_snapshot: object,
    point: int,
    q_global_to_material: np.ndarray,
    tolerance: float = 1.0e-13,
) -> dict[str, float]:
    """Assert physical equality after explicit frame conversion."""

    _, gps_isv, gps_forces, gps_global = _snapshot_arrays(gps_snapshot)
    ref_grad, ref_isv, ref_forces, ref_global = _snapshot_arrays(ref_snapshot)
    gps_global_point = gps_global[point]
    try:
        ref_global_point = _rotate_gradient_to_global(ref_grad[point], q_global_to_material)
    except ValueError as exc:
        raise ValueError(
            f"reference gradient not rotatable at point {point}: "
            f"{ref_grad[point]!r}; q={q_global_to_material!r}"
        ) from exc
    shared = _declared_shared_names(gps_material, gps_snapshot, ref_material, ref_snapshot)
    diffs = {
        "gradient": float(np.max(np.abs(gps_global_point - ref_global_point))),
        "committed_global_strain": float(np.max(np.abs(gps_global_point - ref_global[point]))),
        "stress_material_frame": float(
            np.max(np.abs(gps_forces[point] - ref_forces[point]))
        ),
    }
    gps_slices, ref_slices = shared
    for name in gps_slices:
        diffs[name] = float(
            np.max(
                np.abs(
                    gps_isv[point, gps_slices[name]]
                    - ref_isv[point, ref_slices[name]]
                )
            )
        )
    if max(diffs.values(), default=0.0) >= tolerance:
        raise AssertionError(f"transplanted physical state mismatch at point {point}: {diffs}")
    return diffs


def _declared_shared_names(
    source_material: object,
    source_snapshot: object,
    target_material: object,
    target_snapshot: object,
) -> tuple[dict[str, slice], dict[str, slice]]:
    from fem_inhouse.core.mfront import _declared_internal_slices

    source = _native(source_material)
    target = _native(target_material)
    source_slices = _declared_internal_slices(
        source._mgis,
        source._behaviour,
        source._mgis.Hypothesis.Tridimensional,
        source._specification,
    )
    target_slices = _declared_internal_slices(
        target._mgis,
        target._behaviour,
        target._mgis.Hypothesis.Tridimensional,
        target._specification,
    )
    names = set(source_slices).intersection(target_slices)
    return (
        {name: source_slices[name] for name in names},
        {name: target_slices[name] for name in names},
    )


def _rotations(material: object) -> np.ndarray | None:
    native = _native(material)
    rotations = getattr(native, "_rotations", None)
    return None if rotations is None else np.asarray(rotations, dtype=float)


def _raw_3d_tangent_gps(
    material: object, snapshot: object, strain: np.ndarray, dt: float
) -> np.ndarray:
    """The GPS DSL tangent in the GLOBAL frame, 6x6 Kelvin per point.

    The manager's K is the DSL tangent in the CRYSTAL frame (the law owns the
    rotation). The bridge post-processes a copy: post-multiply by the in-plane
    operator, then rotate every column crystal -> global. Replicate exactly
    that, so the result is comparable to the reference's global tangent. The
    material is EVALUATED first: K is the tangent of the last integration,
    and must be regenerated from the exact snapshot and strain being tested.
    """

    from fem_inhouse.core.mfront import _PLANE_STRESS_COMPONENTS

    native = _native(material)
    manager = native._manager
    material.restore_state(snapshot)
    material.evaluate(
        np.asarray(strain, dtype=float),
        time_increment=dt,
        consistent_tangent=True,
    )
    tangent = np.asarray(manager.K).copy()
    in_plane_operator = np.zeros((6, 6), dtype=float)
    in_plane_operator[_PLANE_STRESS_COMPONENTS, _PLANE_STRESS_COMPONENTS] = 1.0
    tangent = tangent @ in_plane_operator
    if native._mgis_rotations is not None:
        for column in range(6):
            flat = np.ascontiguousarray(tangent[:, :, column].reshape(-1))
            native._mgis.rotateThermodynamicForces(
                flat, native._behaviour, native._mgis_rotations
            )
            tangent[:, :, column] = flat.reshape(tangent.shape[0], 6)
    return tangent


def _raw_3d_tangent_reference(
    material: object,
    snapshot: object,
    strain: np.ndarray,
    dt: float,
    transverse_global: np.ndarray | None = None,
) -> np.ndarray:
    """The reference's 3D tangent at the CONVERGED closure, global Kelvin.

    The condensed batch keeps only the plane-stress Schur, but its local
    Newton ends at `_latest_transverse` -- the converged transverse strain in
    the global frame -- with the bridge holding the 3D tangent of THAT last
    evaluation. Re-running the bridge once with the converged transverse
    reproduces exactly the matrix whose Schur the batch returned. The
    transverse is passed from the caller when the reference is evaluated on a
    transplanted GPS state (the GPS's own converged transverses); otherwise
    the batch's own converged transverses are used.
    """

    from fem_inhouse.core.mfront import (
        _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        _PLANE_STRESS_COMPONENTS,
        _TRANSVERSE_COMPONENTS_3D,
    )

    bridge = getattr(material, "_bridge", None)
    if bridge is None:
        raise RuntimeError("reference backend exposes no _bridge")
    material.restore_state(snapshot)
    # Converge the closure with the condensed batch itself, so every warm
    # start and predictor is the batch's own.
    if transverse_global is None:
        material.evaluate(
            np.asarray(strain, dtype=float),
            time_increment=dt,
            consistent_tangent=True,
        )
        transverse_global = np.asarray(material._latest_transverse, dtype=float).copy()
        material.revert()
    total = np.zeros((material.point_count, 6), dtype=float)
    total[:, _PLANE_STRESS_COMPONENTS] = (
        np.asarray(strain, dtype=float) * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
    )
    total[:, _TRANSVERSE_COMPONENTS_3D] = transverse_global
    trial = bridge.evaluate(total, time_increment=dt, collect_observables=False)
    tangent = np.asarray(trial.consistent_tangent_kelvin_mpa, dtype=float).copy()
    bridge.revert()
    return tangent


def _raw_3d_tangent(
    material: object,
    snapshot: object,
    strain: np.ndarray,
    dt: float,
    *,
    is_reference: bool,
    transverse_global: np.ndarray | None = None,
) -> np.ndarray:
    if is_reference:
        return _raw_3d_tangent_reference(
            material, snapshot, strain, dt, transverse_global=transverse_global
        )
    return _raw_3d_tangent_gps(material, snapshot, strain, dt)


def _block_analysis(tangent: np.ndarray) -> dict[str, object]:
    caa = np.take(np.take(tangent, _PLANE, axis=-2), _PLANE, axis=-1)
    cab = np.take(np.take(tangent, _PLANE, axis=-2), _TRANSVERSE, axis=-1)
    cba = np.take(np.take(tangent, _TRANSVERSE, axis=-2), _PLANE, axis=-1)
    cbb = np.take(np.take(tangent, _TRANSVERSE, axis=-2), _TRANSVERSE, axis=-1)
    return {
        "caa": caa.tolist(),
        "cab": cab.tolist(),
        "cba": cba.tolist(),
        "cbb": cbb.tolist(),
        "cbb_condition": float(np.linalg.cond(cbb)),
        "cbb_min_singular": float(np.linalg.svd(cbb, compute_uv=False)[-1]),
    }


def _schur_plane_stress(tangent: np.ndarray) -> np.ndarray:
    """Caa - Cab Cbb^-1 Cba in Kelvin, then to engineering plane stress."""

    caa = np.take(np.take(tangent, _PLANE, axis=-2), _PLANE, axis=-1)
    cab = np.take(np.take(tangent, _PLANE, axis=-2), _TRANSVERSE, axis=-1)
    cba = np.take(np.take(tangent, _TRANSVERSE, axis=-2), _PLANE, axis=-1)
    cbb = np.take(np.take(tangent, _TRANSVERSE, axis=-2), _TRANSVERSE, axis=-1)
    condensed = caa - cab @ np.linalg.solve(cbb, cba)
    stress_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
    strain_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
    return condensed * stress_scale[:, None] * strain_scale[None, :]


def _gps_projected_tangent(tangent: np.ndarray) -> np.ndarray:
    """The GPS in-plane tangent: DSL tangent, in-plane operator, engineering.

    Accepts either a batch (..., 6, 6) or a single point (6, 6).
    """

    operator = np.zeros((6, 6), dtype=float)
    operator[_PLANE, _PLANE] = 1.0
    projected = tangent @ operator
    in_tangent = np.take(np.take(projected, _PLANE, axis=-2), _PLANE, axis=-1)
    stress_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
    strain_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
    return in_tangent * stress_scale[:, None] * strain_scale[None, :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--checkpoint-increment", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation/_generated/performance/gps_tangent_blocks_same_state_v2.json"
        ),
    )
    arguments = parser.parse_args()

    grid, yield_stress, coefficient, boundary = _load_case(arguments)
    material_gps, recording_gps, result_gps = _run_backend(
        GPS, arguments, grid, yield_stress, coefficient, boundary
    )
    material_ref, recording_ref, result_ref = _run_backend(
        REFERENCE, arguments, grid, yield_stress, coefficient, boundary
    )
    increment = arguments.checkpoint_increment
    calls_gps = _checkpoint_calls(recording_gps, increment)
    calls_ref = _checkpoint_calls(recording_ref, increment)
    strain_gps = np.asarray(calls_gps[0]["strain"], dtype=float)
    strain_ref = np.asarray(calls_ref[0]["strain"], dtype=float)
    dt = float(calls_gps[0]["time_increment"])
    snapshot_gps = recording_gps.committed_snapshots[increment - 2]
    snapshot_ref = recording_ref.committed_snapshots[increment - 2]

    rotations_gps = _rotations(material_gps)

    rows = []
    for point in TOP_POINTS:
        row: dict[str, object] = {"point": point}
        q = None if rotations_gps is None else rotations_gps[point]

        # --- own states: each backend from its own committed snapshot ---
        own_gps = _evaluate_raw(material_gps, snapshot_gps, strain_gps, dt)
        own_ref = _evaluate_raw(material_ref, snapshot_ref, strain_ref, dt)
        raw_gps_own = _raw_3d_tangent(
            material_gps, snapshot_gps, strain_gps, dt, is_reference=False
        )
        raw_ref_own = _raw_3d_tangent(material_ref, snapshot_ref, strain_ref, dt, is_reference=True)
        row["own_states_relative_tangent_difference"] = float(
            np.linalg.norm(raw_gps_own[point] - raw_ref_own[point])
            / max(np.linalg.norm(raw_ref_own[point]), 1.0e-30)
        )
        row["own_states_in_plane_relative"] = float(
            np.linalg.norm(own_gps["tangent_in_plane"][point] - own_ref["tangent_in_plane"][point])
            / max(np.linalg.norm(own_ref["tangent_in_plane"][point]), 1.0e-30)
        )

        # --- complete transplant: GPS state into the reference ---
        if q is None:
            raise RuntimeError("same-state transplant requires EBSD rotations")
        ref_on_gps_snapshot = _make_transplanted_snapshot(
            material_ref,
            snapshot_ref,
            material_gps,
            snapshot_gps,
            point,
            q_global_to_material=q,
        )
        state_diffs = _assert_same_physical_committed_state(
            material_gps,
            snapshot_gps,
            material_ref,
            ref_on_gps_snapshot,
            point,
            q,
        )
        ref_on_gps_state = _evaluate_raw(material_ref, ref_on_gps_snapshot, strain_gps, dt)
        raw_ref_on_gps = _raw_3d_tangent(
            material_ref,
            ref_on_gps_snapshot,
            strain_gps,
            dt,
            is_reference=True,
            transverse_global=material_gps.committed_transverse_strain_kelvin,
        )
        row["gps_state_into_ref_3d_relative"] = float(
            np.linalg.norm(raw_gps_own[point] - raw_ref_on_gps[point])
            / max(np.linalg.norm(raw_ref_on_gps[point]), 1.0e-30)
        )
        row["gps_state_into_ref_in_plane_relative"] = float(
            np.linalg.norm(
                own_gps["tangent_in_plane"][point]
                - ref_on_gps_state["tangent_in_plane"][point]
            )
            / max(np.linalg.norm(ref_on_gps_state["tangent_in_plane"][point]), 1.0e-30)
        )
        row["transplant_state_differences"] = state_diffs

        # --- complete transplant: reference state into the GPS ---
        gps_on_ref_snapshot = _make_transplanted_snapshot(
            material_gps,
            snapshot_gps,
            material_ref,
            snapshot_ref,
            point,
            q_global_to_material=q,
        )
        raw_gps_on_ref = _raw_3d_tangent(
            material_gps, gps_on_ref_snapshot, strain_ref, dt, is_reference=False
        )
        row["ref_state_into_gps_3d_relative"] = float(
            np.linalg.norm(raw_ref_own[point] - raw_gps_on_ref[point])
            / max(np.linalg.norm(raw_gps_on_ref[point]), 1.0e-30)
        )

        # --- block analysis on the same complete state (GPS state) ---
        # Evaluate the GPS first and read the transverse ITS closure converged
        # to at this strain; the reference is then evaluated at that SAME
        # transverse, so the Schur-vs-projected comparison is formulation-only.
        material_gps.restore_state(snapshot_gps)
        material_gps.evaluate(
            strain_gps.reshape(-1, 3),
            time_increment=dt,
            consistent_tangent=True,
        )
        gps_converged_transverse = np.asarray(
            material_gps._latest_transverse, dtype=float
        ).copy()
        raw_gps_same = _raw_3d_tangent(
            material_gps, snapshot_gps, strain_gps, dt, is_reference=False
        )
        # Re-evaluate the reference at the GPS-converged transverse, from the
        # GPS state transplant already in place.
        raw_ref_on_gps = _raw_3d_tangent(
            material_ref,
            ref_on_gps_snapshot,
            strain_gps,
            dt,
            is_reference=True,
            transverse_global=gps_converged_transverse,
        )
        row["blocks_gps"] = _block_analysis(raw_gps_same[point])
        row["blocks_reference_on_gps_state"] = _block_analysis(raw_ref_on_gps[point])
        schur_ref = _schur_plane_stress(raw_ref_on_gps[point])
        gps_projected = _gps_projected_tangent(raw_gps_same[point])
        row["same_state_schur_vs_gps_projected_relative"] = float(
            np.linalg.norm(schur_ref - gps_projected)
            / max(np.linalg.norm(schur_ref), 1.0e-30)
        )
        row["same_state_schur_vs_gps_returned_relative"] = float(
            np.linalg.norm(
                schur_ref - ref_on_gps_state["tangent_in_plane"][point]
            )
            / max(np.linalg.norm(schur_ref), 1.0e-30)
        )
        # The reference's own returned tangent vs its own Schur (consistency).
        schur_ref_own = _schur_plane_stress(raw_ref_own[point])
        row["ref_returned_vs_own_schur_relative"] = float(
            np.linalg.norm(schur_ref_own - own_ref["tangent_in_plane"][point])
            / max(np.linalg.norm(schur_ref_own), 1.0e-30)
        )
        rows.append(row)

    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "checkpoint_increment": increment,
        },
        "reference": {
            "gps_newton": int(sum(result_gps.diagnostics.iterations_per_increment)),
            "ref_newton": int(sum(result_ref.diagnostics.iterations_per_increment)),
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(f"=== point {row['point']}")
        print(f"  own states: 3D rel {row['own_states_relative_tangent_difference']:.3e}, "
              f"in-plane rel {row['own_states_in_plane_relative']:.3e}")
        print(f"  GPS state -> ref: 3D rel {row['gps_state_into_ref_3d_relative']:.3e}, "
              f"in-plane rel {row['gps_state_into_ref_in_plane_relative']:.3e}")
        print(f"  ref state -> GPS: 3D rel {row['ref_state_into_gps_3d_relative']:.3e}")
        print("  same-state Schur(ref) vs GPS projected: "
              f"{row['same_state_schur_vs_gps_projected_relative']:.3e}")
        print("  same-state Schur(ref) vs ref returned:  "
              f"{row['same_state_schur_vs_gps_returned_relative']:.3e}")
        print(f"  ref returned vs own Schur: {row['ref_returned_vs_own_schur_relative']:.3e}")
        print("  Cbb cond ref-on-gps-state: "
              f"{row['blocks_reference_on_gps_state']['cbb_condition']:.3e}, "
              f"min singular {row['blocks_reference_on_gps_state']['cbb_min_singular']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
