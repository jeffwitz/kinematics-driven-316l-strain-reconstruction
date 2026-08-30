#!/usr/bin/env python3
"""Build a candidate global EBSD-derived microstructure product.

The input Euler maps are documented as per-pixel grain-mean orientations.  No
physical misorientation threshold is used here: exact stored orientation
plateaus are first labelled, then split into four-connected components.  The
result is deliberately a candidate product, not a golden experimental input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.identification.grain_boundary_descriptors import (
    cubic_misorientation_angle,
    luster_morris_matrix,
    residual_burgers_matrix,
    rotated_fcc_slip_systems,
)

# The report intentionally contains long scientific definitions and paths.
# Keep Ruff's executable checks while allowing those human-readable lines.
# ruff: noqa: E501

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
DEFAULT_WORK = Path("/tmp/derived_ebsd_microstructure_v1")
M20_CROP = (1610, 1630, 1075, 1095)
TRACE_RADIUS = 3.0
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _orientation_labels(angles: NDArray[np.float64]) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Return exact triplet labels and their sorted stored values."""

    if angles.ndim != 3 or angles.shape[-1] != 3:
        raise ValueError(f"expected H,W,3 Euler map, got {angles.shape}")
    if not np.isfinite(angles).all():
        raise ValueError("orientation maps contain non-finite values")
    dtype = np.dtype([("phi1", "<f8"), ("Phi", "<f8"), ("phi2", "<f8")])
    records = np.empty(angles.shape[0] * angles.shape[1], dtype=dtype)
    flat = angles.reshape(-1, 3)
    records["phi1"], records["Phi"], records["phi2"] = flat.T
    unique, inverse = np.unique(records, return_inverse=True)
    unique_angles = np.column_stack([unique[name] for name in ("phi1", "Phi", "phi2")])
    return inverse.reshape(angles.shape[:2]).astype(np.int32), unique_angles


def _smallest_positive_neighbour_gap(angles: NDArray[np.float64]) -> float:
    """Measure the smallest non-zero neighbouring Euler component gap."""

    candidates: list[float] = []
    for first, second in ((angles[:, :-1], angles[:, 1:]), (angles[:-1], angles[1:])):
        difference = np.max(np.abs(first - second), axis=-1)
        positive = difference[difference > 0.0]
        if positive.size:
            candidates.append(float(positive.min()))
    if not candidates:
        return 0.0
    return min(candidates)


def _orientation_preflight(angles: NDArray[np.float64]) -> dict[str, object]:
    """Audit whether exact orientation plateaus can be segmented defensibly."""

    differences = np.concatenate(
        (
            np.max(np.abs(angles[:, :-1] - angles[:, 1:]), axis=-1).ravel(),
            np.max(np.abs(angles[:-1] - angles[1:]), axis=-1).ravel(),
        )
    )
    positive = np.sort(differences[differences > 0.0])
    quantiles = np.quantile(positive, [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0])
    # A numerical quantisation gap must be an empty interval in log space,
    # with substantial support on both sides.  A smooth positive tail is not
    # evidence for a tolerance and must block automatic segmentation.
    clear_gap = False
    gap_bounds: list[float] = []
    if positive.size >= 100:
        unique_positive = np.unique(positive)
        ratios = unique_positive[1:] / unique_positive[:-1]
        # A candidate numerical gap must be an early, order-of-magnitude
        # separation.  The large gap at hundreds of degrees in this export is
        # an Euler-angle wrap/outlier and is deliberately not accepted.
        for index in np.flatnonzero((ratios >= 10.0) & (unique_positive[:-1] <= 5.0)):
            left = np.count_nonzero(positive <= unique_positive[index])
            right = np.count_nonzero(positive >= unique_positive[index + 1])
            if left >= 100 and right >= 100:
                clear_gap = True
                gap_bounds = [float(unique_positive[index]), float(unique_positive[index + 1])]
                break
    return {
        "neighbour_pairs": int(differences.size),
        "exact_equal_fraction": float(np.mean(differences == 0.0)),
        "positive_difference_count": int(positive.size),
        "positive_difference_quantiles_deg": quantiles.tolist(),
        "smallest_positive_difference_deg": float(positive.min()) if positive.size else 0.0,
        "clear_numeric_gap_detected": clear_gap,
        "candidate_log10_gap": gap_bounds,
        "decision": "diagnostic only; no segmentation gate",
    }


def _build_grain_ids(
    orientation_label: NDArray[np.int32], unique_angles: NDArray[np.float64]
) -> tuple[NDArray[np.int32], dict[str, NDArray[np.generic]]]:
    """Split exact orientation labels into four-connected grain components."""

    shape = orientation_label.shape
    grain_ids = np.full(shape, -1, dtype=np.int32)
    flat = orientation_label.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=len(unique_angles))
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    records: list[tuple[int, int, float, float, float, int, int, int, int, int, bool]] = []
    grain_id = 0
    for orientation_id, count in enumerate(counts):
        if count == 0:
            continue
        indices = order[offsets[orientation_id] : offsets[orientation_id + 1]]
        rows = indices // shape[1]
        cols = indices % shape[1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[rows - r0, cols - c0] = True
        components, n_components = ndimage.label(local, structure=FOUR_CONNECTED)
        for component in range(1, n_components + 1):
            local_indices = np.flatnonzero(components == component)
            component_rows, component_cols = np.unravel_index(local_indices, local.shape)
            global_rows = component_rows + r0
            global_cols = component_cols + c0
            grain_ids[global_rows, global_cols] = grain_id
            area = int(local_indices.size)
            records.append(
                (
                    grain_id,
                    orientation_id,
                    float(unique_angles[orientation_id, 0]),
                    float(unique_angles[orientation_id, 1]),
                    float(unique_angles[orientation_id, 2]),
                    area,
                    int(global_rows.min()),
                    int(global_rows.max()) + 1,
                    int(global_cols.min()),
                    int(global_cols.max()) + 1,
                    bool(
                        global_rows.min() == 0
                        or global_rows.max() == shape[0] - 1
                        or global_cols.min() == 0
                        or global_cols.max() == shape[1] - 1
                    ),
                )
            )
            grain_id += 1
    if np.any(grain_ids < 0):
        raise RuntimeError("some pixels were not assigned a grain")
    dtype = np.dtype(
        [
            ("grain_id", "i4"),
            ("orientation_id", "i4"),
            ("phi1", "f8"),
            ("Phi", "f8"),
            ("phi2", "f8"),
            ("area_px", "i8"),
            ("r0", "i4"),
            ("r1", "i4"),
            ("c0", "i4"),
            ("c1", "i4"),
            ("touches_border", "?")
        ]
    )
    table = np.asarray(records, dtype=dtype)
    return grain_ids, {name: table[name] for name in table.dtype.names or ()}


def _grain_dense(grain_ids: NDArray[np.int32], values: NDArray[np.generic]) -> NDArray[np.float64]:
    return np.asarray(values[grain_ids], dtype=np.float64)


def _edge_arrays(
    grain_ids: NDArray[np.int32],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Return unique interface key and source coordinates for right/down edges."""

    keys: list[NDArray[np.int64]] = []
    rows: list[NDArray[np.int32]] = []
    cols: list[NDArray[np.int32]] = []
    other_rows: list[NDArray[np.int32]] = []
    other_cols: list[NDArray[np.int32]] = []
    n_grains = int(grain_ids.max()) + 1
    for axis in (0, 1):
        first = np.take(grain_ids, np.arange(grain_ids.shape[axis] - 1), axis=axis)
        second = np.take(grain_ids, np.arange(1, grain_ids.shape[axis]), axis=axis)
        mask = first != second
        r, c = np.where(mask)
        a = first[mask].astype(np.int64)
        b = second[mask].astype(np.int64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keys.append(lo * n_grains + hi)
        rows.append(r.astype(np.int32))
        cols.append(c.astype(np.int32))
        if axis == 0:
            other_rows.append((r + 1).astype(np.int32))
            other_cols.append(c.astype(np.int32))
        else:
            other_rows.append(r.astype(np.int32))
            other_cols.append((c + 1).astype(np.int32))
    return (
        np.concatenate(keys),
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(other_rows),
        np.concatenate(other_cols),
    )


def _interface_geometry(
    points: NDArray[np.int32], radius: float = TRACE_RADIUS
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    """Estimate a stable 2-D tangent/normal from same-interface points."""

    points_unique = np.unique(points, axis=0)
    if len(points_unique) < 2:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0
    tree = cKDTree(points_unique.astype(float))
    sample_count = min(200, len(points_unique))
    sample_idx = np.linspace(0, len(points_unique) - 1, sample_count, dtype=int)
    tangents: list[NDArray[np.float64]] = []
    qualities: list[float] = []
    for index in sample_idx:
        neighbours = tree.query_ball_point(points_unique[index], radius)
        if len(neighbours) < 3:
            continue
        local = points_unique[np.asarray(neighbours, dtype=int)].astype(float)
        centered = local - local.mean(axis=0)
        covariance = centered.T @ centered / max(len(local) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, 1]
        quality = float(eigenvalues[1] / max(eigenvalues.sum(), np.finfo(float).eps))
        tangents.append(tangent)
        qualities.append(quality)
    if not tangents:
        centered = points_unique.astype(float) - points_unique.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangent = vh[0]
        quality = 0.0
        sample_count = 0
    else:
        reference = tangents[0]
        aligned = [t if np.dot(t, reference) >= 0 else -t for t in tangents]
        tangent = np.sum(aligned, axis=0)
        tangent /= max(np.linalg.norm(tangent), np.finfo(float).eps)
        quality = float(np.median(qualities))
    normal = np.array([-tangent[1], tangent[0]])
    return tangent.astype(float), normal.astype(float), quality, len(tangents)


def _write_dataset(group: h5py.Group, name: str, data: NDArray, **attrs: object) -> h5py.Dataset:
    kwargs: dict[str, object] = {}
    if data.ndim >= 2 and data.size > 10000:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    dataset = group.create_dataset(name, data=data, **kwargs)
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def _plot_qa(
    out: Path,
    grain_ids: NDArray[np.int32],
    diameter: NDArray[np.float64],
    xi: NDArray[np.float64],
    misorientation: NDArray[np.float32],
    mprime: NDArray[np.float32],
    burgers: NDArray[np.float32],
    tangent: NDArray[np.float32],
    quality: NDArray[np.float32],
    crossing: NDArray[np.float32],
) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    plots = [
        ("01_grains_boundaries.png", grain_ids, "Grain IDs", "tab20"),
        ("02_equivalent_diameter_px.png", diameter, "Equivalent grain diameter [px]", "viridis"),
        ("03_distance_to_gb_over_deq.png", xi, "Distance to nearest GB / d_eq", "magma"),
        ("04_nearest_gb_misorientation_deg.png", misorientation, "Nearest GB misorientation [deg]", "viridis"),
        ("05_max_mprime.png", np.nanmax(mprime, axis=-1), "Max m' to nearest neighbour", "viridis"),
        ("06_min_residual_burgers.png", np.nanmin(burgers, axis=-1), "Min sign-invariant residual Burgers", "magma"),
        ("07_trace_quality.png", quality, "Nearest GB trace PCA quality", "viridis"),
        ("08_trace_crossing_system0.png", crossing[..., 0], "Slip trace crossing factor, system 0", "coolwarm"),
    ]
    for filename, image, title, cmap in plots:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sample = image[::8, ::8] if image.shape[0] > 1000 else image
        ax.imshow(sample, origin="upper", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("EBSD column (downsampled)" if sample is not image else "EBSD column")
        ax.set_ylabel("EBSD row (downsampled)" if sample is not image else "EBSD row")
        path = out / filename
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path))
    # A small vector QA view in the upper-left region with valid local traces.
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    r1, c1 = min(500, grain_ids.shape[0]), min(500, grain_ids.shape[1])
    ax.imshow(grain_ids[:r1, :c1], origin="upper", cmap="tab20")
    step = 20
    yy, xx = np.mgrid[0:r1:step, 0:c1:step]
    ax.quiver(xx, yy, tangent[:r1:step, :c1:step, 1], tangent[:r1:step, :c1:step, 0], color="white", scale=30)
    ax.set_title("Local nearest-GB tangents (QA crop)")
    path = out / "09_trace_vectors_crop.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _plot_segmentation_qa(
    out: Path,
    orientation_label: NDArray[np.int32],
    grain_ids: NDArray[np.int32],
    crop: tuple[int, int, int, int] = M20_CROP,
) -> list[str]:
    """Write only the segmentation figures needed before product promotion."""

    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    images = [
        ("segmentation_orientation_labels_global.png", orientation_label, "Exact stored orientation labels"),
        ("segmentation_grain_ids_global.png", grain_ids, "4-connected grain IDs"),
    ]
    for filename, image, title in images:
        figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
        axis.imshow(image[::8, ::8], origin="upper", cmap="nipy_spectral")
        axis.set_title(title)
        axis.set_xlabel("EBSD column (downsampled)")
        axis.set_ylabel("EBSD row (downsampled)")
        path = out / filename
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(str(path))
    r0, r1, c0, c1 = crop
    for filename, image, title in (
        ("segmentation_orientation_labels_m20.png", orientation_label[r0:r1, c0:c1], "Exact labels, M20 crop"),
        ("segmentation_grain_ids_m20.png", grain_ids[r0:r1, c0:c1], "Grain IDs, M20 crop"),
    ):
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        axis.imshow(image, origin="upper", cmap="nipy_spectral", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("EBSD column")
        axis.set_ylabel("EBSD row")
        path = out / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _sample_cubic_neighbour_misorientation(
    orientation_label: NDArray[np.int32], unique_angles: NDArray[np.float64], sample_size: int = 2000
) -> NDArray[np.float64]:
    """Compute true cubic misorientation for a deterministic neighbour sample."""

    right_a, right_b = orientation_label[:, :-1], orientation_label[:, 1:]
    down_a, down_b = orientation_label[:-1], orientation_label[1:]
    pairs = np.concatenate(
        (
            np.column_stack((right_a[right_a != right_b], right_b[right_a != right_b])),
            np.column_stack((down_a[down_a != down_b], down_b[down_a != down_b])),
        )
    )
    if not len(pairs):
        return np.empty(0, dtype=np.float64)
    rotations = rotations_from_euler_bunge_deg(unique_angles)
    indices = np.linspace(0, len(pairs) - 1, min(sample_size, len(pairs)), dtype=int)
    return np.asarray(
        [cubic_misorientation_angle(rotations[int(i)], rotations[int(j)]) for i, j in pairs[indices]],
        dtype=np.float64,
    )


def build(source: Path, work: Path) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=True)
    h5_path = work / "derived_ebsd_microstructure_v1.h5"
    with h5py.File(source, "r") as handle:
        phi1 = np.asarray(handle["orientation/phi1"], dtype=np.float64)
        capital_phi = np.asarray(handle["orientation/Phi"], dtype=np.float64)
        phi2 = np.asarray(handle["orientation/phi2"], dtype=np.float64)
        orientation_attrs = {key: value for key, value in handle["orientation/phi1"].attrs.items()}
    angles = np.stack((phi1, capital_phi, phi2), axis=-1)
    preflight = _orientation_preflight(angles)
    neighbour_gap = _smallest_positive_neighbour_gap(angles)
    orientation_label, unique_angles = _orientation_labels(angles)
    del phi1, capital_phi, phi2, angles
    grain_ids, grain = _build_grain_ids(orientation_label, unique_angles)
    n_grains = int(grain_ids.max()) + 1
    area = np.asarray(grain["area_px"], dtype=np.int64)
    component_counts = np.bincount(
        np.asarray(grain["orientation_id"], dtype=np.int32), minlength=len(unique_angles)
    )
    cubic_neighbour_sample = _sample_cubic_neighbour_misorientation(orientation_label, unique_angles)
    rotations_for_qa = rotations_from_euler_bunge_deg(unique_angles)
    orthogonality_error = float(
        np.max(np.abs(np.einsum("nij,nkj->nik", rotations_for_qa, rotations_for_qa) - np.eye(3)))
    )
    determinant_range = [float(np.min(np.linalg.det(rotations_for_qa))), float(np.max(np.linalg.det(rotations_for_qa)))]
    segmentation_stats = {
        "n_orientation_labels": len(unique_angles),
        "n_grains_after_connectivity": n_grains,
        "extra_components": n_grains - len(unique_angles),
        "component_count_quantiles": np.quantile(component_counts, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "fraction_labels_with_one_component": float(np.mean(component_counts == 1)),
        "grain_area_quantiles_px": np.quantile(area, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "grain_area_mean_px": float(np.mean(area)),
        "fraction_grains_area_le_1_px": float(np.mean(area <= 1)),
        "fraction_grains_area_le_2_px": float(np.mean(area <= 2)),
        "fraction_grains_area_le_4_px": float(np.mean(area <= 4)),
        "fraction_pixels_in_area_le_1_px_grains": float(np.sum(area[area <= 1]) / area.sum()),
        "fraction_pixels_in_area_le_2_px_grains": float(np.sum(area[area <= 2]) / area.sum()),
        "fraction_pixels_in_area_le_4_px_grains": float(np.sum(area[area <= 4]) / area.sum()),
        "rotation_max_orthogonality_error": orthogonality_error,
        "rotation_determinant_range": determinant_range,
        "cubic_neighbour_misorientation_sample_deg": {
            "count": len(cubic_neighbour_sample),
            "min": float(np.min(cubic_neighbour_sample)) if len(cubic_neighbour_sample) else None,
            "median": float(np.median(cubic_neighbour_sample)) if len(cubic_neighbour_sample) else None,
            "max": float(np.max(cubic_neighbour_sample)) if len(cubic_neighbour_sample) else None,
            "quantiles": np.quantile(cubic_neighbour_sample, [0.01, 0.5, 0.99]).tolist() if len(cubic_neighbour_sample) else [],
        },
    }
    segmentation_figures = _plot_segmentation_qa(work / "segmentation_figures", orientation_label, grain_ids)
    # Exact labels that split into many one-pixel components are an export QA
    # finding, not a reason to silently introduce an angular segmentation rule.
    if segmentation_stats["fraction_grains_area_le_4_px"] > 0.9 and segmentation_stats["component_count_quantiles"][2] > 1:
        report = {
            "schema_version": "derived_ebsd_microstructure_v1_segmentation_qa",
            "candidate_product": False,
            "source": str(source),
            "source_sha256": _sha256(source),
            "grid_shape": list(grain_ids.shape),
            "source_orientation_datasets": ["orientation/phi1", "orientation/Phi", "orientation/phi2"],
            "source_orientation_dtype": "float64",
            "source_orientation_finite": True,
            "source_orientation_attrs": {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in orientation_attrs.items()},
            "raw_euler_component_difference": preflight,
            "orientation_convention": "Bunge ZXZ Euler angles, degrees (source metadata)",
            "rotation_validation": {"max_orthogonality_error": orthogonality_error, "determinant_range": determinant_range},
            "segmentation": segmentation_stats,
            "segmentation_figures": segmentation_figures,
            "decision": "pathological exact segmentation candidate; do not generate derived HDF5",
            "required_next_action": "audit the source export or define a provenance-backed plateau interpretation before building grain descriptors",
            "golden_status": "no product generated",
        }
        (work / "derived_ebsd_microstructure_v1_segmentation_qa.json").write_text(json.dumps(report, indent=2) + "\n")
        (work / "derived_ebsd_microstructure_v1_segmentation_qa.md").write_text(_markdown_segmentation_report(report))
        return report
    deq = 2.0 * np.sqrt(area / np.pi)
    deq_dense = _grain_dense(grain_ids, deq)

    edge_key, edge_rows, edge_cols, edge_other_rows, edge_other_cols = _edge_arrays(grain_ids)
    unique_keys, edge_pair_id = np.unique(edge_key, return_inverse=True)
    n_boundaries = len(unique_keys)
    pair_a = (unique_keys // n_grains).astype(np.int32)
    pair_b = (unique_keys % n_grains).astype(np.int32)
    pair_points: list[NDArray[np.int32]] = []
    pair_contact = np.zeros(n_boundaries, dtype=np.int64)
    pair_bbox = np.zeros((n_boundaries, 4), dtype=np.int32)
    pair_tangent = np.zeros((n_boundaries, 2), dtype=np.float64)
    pair_normal = np.zeros((n_boundaries, 2), dtype=np.float64)
    pair_quality = np.zeros(n_boundaries, dtype=np.float64)
    pair_samples = np.zeros(n_boundaries, dtype=np.int32)
    boundary_pair_map = np.full(grain_ids.shape, -1, dtype=np.int32)
    boundary_pixel_ids: list[NDArray[np.int64]] = []
    boundary_pixel_pairs: list[NDArray[np.int32]] = []
    edge_order = np.argsort(edge_pair_id, kind="stable")
    sorted_pair = edge_pair_id[edge_order]
    starts = np.flatnonzero(np.r_[True, sorted_pair[1:] != sorted_pair[:-1]])
    stops = np.r_[starts[1:], len(sorted_pair)]
    for pair_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        indices = edge_order[start:stop]
        r = edge_rows[indices]
        c = edge_cols[indices]
        opposite = np.column_stack((edge_other_rows[indices], edge_other_cols[indices]))
        points = np.concatenate((np.column_stack((r, c)), opposite), axis=0)
        points = np.unique(points, axis=0).astype(np.int32)
        pair_points.append(points)
        pair_contact[pair_id] = stop - start
        pair_bbox[pair_id] = (points[:, 0].min(), points[:, 0].max() + 1, points[:, 1].min(), points[:, 1].max() + 1)
        pair_tangent[pair_id], pair_normal[pair_id], pair_quality[pair_id], pair_samples[pair_id] = _interface_geometry(points)
        flat_points = np.ravel_multi_index((points[:, 0], points[:, 1]), grain_ids.shape)
        boundary_pixel_ids.append(flat_points)
        boundary_pixel_pairs.append(np.full(len(points), pair_id, dtype=np.int32))

    all_boundary_pixels = np.concatenate(boundary_pixel_ids)
    all_boundary_pairs = np.concatenate(boundary_pixel_pairs)
    boundary_order = np.lexsort((all_boundary_pairs, all_boundary_pixels))
    ordered_pixels = all_boundary_pixels[boundary_order]
    ordered_pairs = all_boundary_pairs[boundary_order]
    unique_pixel_starts = np.flatnonzero(np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]])
    unique_pixel_stops = np.r_[unique_pixel_starts[1:], len(ordered_pixels)]
    unique_pixels = ordered_pixels[unique_pixel_starts]
    selected_pairs = ordered_pairs[unique_pixel_starts]
    boundary_pair_map.ravel()[unique_pixels] = selected_pairs
    ambiguous = np.zeros(grain_ids.shape, dtype=bool)
    ambiguous.ravel()[unique_pixels] = np.array(
        [len(np.unique(ordered_pairs[s:e])) > 1 for s, e in zip(unique_pixel_starts, unique_pixel_stops, strict=True)],
        dtype=bool,
    )
    boundary_mask = boundary_pair_map >= 0
    triple_junction = ambiguous.copy()

    distance, nearest_indices = ndimage.distance_transform_edt(~boundary_mask, return_indices=True)
    nearest_pair = boundary_pair_map[nearest_indices[0], nearest_indices[1]]
    nearest_pair = np.where(grain_ids >= 0, nearest_pair, -1).astype(np.int32)
    nearest_neighbor = np.full(grain_ids.shape, -1, dtype=np.int32)
    for pair_id in range(n_boundaries):
        mask = nearest_pair == pair_id
        nearest_neighbor[mask & (grain_ids == pair_a[pair_id])] = pair_b[pair_id]
        nearest_neighbor[mask & (grain_ids == pair_b[pair_id])] = pair_a[pair_id]
    nearest_distance = distance.astype(np.float32)
    xi = (distance / deq_dense).astype(np.float32)

    rotations = rotations_from_euler_bunge_deg(np.column_stack([grain["phi1"], grain["Phi"], grain["phi2"]]))
    slip_normals = np.empty((n_grains, 12, 3), dtype=np.float64)
    slip_trace = np.full((n_grains, 12, 2), np.nan, dtype=np.float64)
    slip_trace_valid = np.zeros((n_grains, 12), dtype=bool)
    ez = np.array([0.0, 0.0, 1.0])
    for gid in range(n_grains):
        _, slip_normals[gid] = rotated_fcc_slip_systems(rotations[gid])
        traces = np.cross(ez[None, :], slip_normals[gid])
        norms = np.linalg.norm(traces, axis=1)
        valid = norms > 1e-12
        slip_trace[gid, valid] = traces[valid, :2] / norms[valid, None]
        slip_trace_valid[gid] = valid

    misorientation = np.zeros(n_boundaries, dtype=np.float32)
    mprime_matrix = np.zeros((n_boundaries, 12, 12), dtype=np.float32)
    burgers_matrix = np.zeros((n_boundaries, 12, 12), dtype=np.float32)
    mprime_a = np.zeros((n_boundaries, 12), dtype=np.float32)
    mprime_b = np.zeros((n_boundaries, 12), dtype=np.float32)
    mprime_partner_a = np.zeros((n_boundaries, 12), dtype=np.int8)
    mprime_partner_b = np.zeros((n_boundaries, 12), dtype=np.int8)
    burgers_a = np.zeros((n_boundaries, 12), dtype=np.float32)
    burgers_b = np.zeros((n_boundaries, 12), dtype=np.float32)
    burgers_partner_a = np.zeros((n_boundaries, 12), dtype=np.int8)
    burgers_partner_b = np.zeros((n_boundaries, 12), dtype=np.int8)
    for pair_id in range(n_boundaries):
        ga, gb = int(pair_a[pair_id]), int(pair_b[pair_id])
        misorientation[pair_id] = cubic_misorientation_angle(rotations[ga], rotations[gb])
        matrix = luster_morris_matrix(rotations[ga], rotations[gb])
        rb = residual_burgers_matrix(rotations[ga], rotations[gb])
        mprime_matrix[pair_id] = matrix
        burgers_matrix[pair_id] = rb
        mprime_a[pair_id], mprime_partner_a[pair_id] = matrix.max(axis=1), matrix.argmax(axis=1)
        mprime_b[pair_id], mprime_partner_b[pair_id] = matrix.max(axis=0), matrix.argmax(axis=0)
        burgers_a[pair_id], burgers_partner_a[pair_id] = rb.min(axis=1), rb.argmin(axis=1)
        burgers_b[pair_id], burgers_partner_b[pair_id] = rb.min(axis=0), rb.argmin(axis=0)

    nearest_misorientation = np.full(grain_ids.shape, np.nan, dtype=np.float32)
    mprime_dense = np.full((*grain_ids.shape, 12), np.nan, dtype=np.float32)
    mprime_partner_dense = np.full((*grain_ids.shape, 12), -1, dtype=np.int8)
    burgers_dense = np.full((*grain_ids.shape, 12), np.nan, dtype=np.float32)
    burgers_partner_dense = np.full((*grain_ids.shape, 12), -1, dtype=np.int8)
    nearest_tangent = np.full((*grain_ids.shape, 2), np.nan, dtype=np.float32)
    nearest_normal = np.full((*grain_ids.shape, 2), np.nan, dtype=np.float32)
    nearest_quality = np.full(grain_ids.shape, np.nan, dtype=np.float32)
    for pair_id in range(n_boundaries):
        mask = nearest_pair == pair_id
        side_a = mask & (grain_ids == pair_a[pair_id])
        side_b = mask & (grain_ids == pair_b[pair_id])
        nearest_misorientation[mask] = misorientation[pair_id]
        mprime_dense[side_a] = mprime_a[pair_id]
        mprime_dense[side_b] = mprime_b[pair_id]
        mprime_partner_dense[side_a] = mprime_partner_a[pair_id]
        mprime_partner_dense[side_b] = mprime_partner_b[pair_id]
        burgers_dense[side_a] = burgers_a[pair_id]
        burgers_dense[side_b] = burgers_b[pair_id]
        burgers_partner_dense[side_a] = burgers_partner_a[pair_id]
        burgers_partner_dense[side_b] = burgers_partner_b[pair_id]
        nearest_tangent[mask] = pair_tangent[pair_id]
        nearest_normal[mask] = pair_normal[pair_id]
        nearest_quality[mask] = pair_quality[pair_id]

    trace_angle = np.full((*grain_ids.shape, 12), np.nan, dtype=np.float32)
    trace_crossing = np.full((*grain_ids.shape, 12), np.nan, dtype=np.float32)
    trace_valid_dense = np.zeros((*grain_ids.shape, 12), dtype=bool)
    for gid in range(n_grains):
        mask = (grain_ids == gid) & (nearest_pair >= 0) & np.isfinite(nearest_quality)
        if not np.any(mask):
            continue
        traces = slip_trace[gid]
        valid = slip_trace_valid[gid]
        tangent = nearest_tangent[mask]
        normal = nearest_normal[mask]
        dots_t = np.abs(tangent[:, None, :] @ traces[:, :, None, :].transpose(0, 1, 3, 2))
        # The compact einsum below is equivalent to |t_GB dot t_alpha|.
        dot_t = np.abs(np.einsum("ni,ai->na", tangent, traces))
        dot_n = np.abs(np.einsum("ni,ai->na", normal, traces))
        angles = np.degrees(np.arccos(np.clip(dot_t, 0.0, 1.0)))
        rows, cols = np.where(mask)
        trace_angle[rows, cols] = np.where(valid[None, :], angles, np.nan)
        trace_crossing[rows, cols] = np.where(valid[None, :], dot_n, np.nan)
        trace_valid_dense[rows, cols] = valid[None, :]
        del dots_t

    # Ensure output is created only after all dense calculations succeeded.
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(
            schema_version="derived_ebsd_microstructure_v1",
            candidate_product=True,
            created_utc=datetime.now(UTC).isoformat(),
            source_sha256=_sha256(source),
            generation_git_sha=_git_sha(),
            orientation_convention="Bunge ZXZ Euler angles, degrees; Q_global_to_material",
            crystal_symmetry="cubic proper signed permutations (24)",
            physical_ebsd_scale_verified=False,
            source_pixel_size_um=1.84,
            source_pixel_size_semantics="unresolved native EBSD acquisition scale",
        )
        provenance = handle.create_group("provenance")
        _write_dataset(provenance, "source_path", np.bytes_(str(source)))
        _write_dataset(provenance, "source_sha256", np.bytes_(_sha256(source)))
        _write_dataset(provenance, "generation_git_sha", np.bytes_(_git_sha()))
        _write_dataset(provenance, "grid_shape", np.asarray(grain_ids.shape, dtype=np.int32), units="pixels")
        _write_dataset(provenance, "m20_crop_absolute", np.asarray(M20_CROP, dtype=np.int32), convention="first_axis_start_stop, second_axis_start_stop")
        fields = handle.create_group("fields")
        _write_dataset(fields, "valid_mask", np.ones(grain_ids.shape, dtype=bool), definition="finite source Euler triplet")
        _write_dataset(fields, "orientation_label", orientation_label, definition="exact stored Euler triplet label")
        _write_dataset(fields, "grain_id", grain_ids, definition="4-connected component of an exact orientation plateau")
        _write_dataset(fields, "grain_area_px", area[grain_ids], units="pixel^2")
        _write_dataset(fields, "grain_equivalent_diameter_px", deq_dense, units="pixel")
        _write_dataset(fields, "inverse_sqrt_grain_size_proxy", 1.0 / np.sqrt(deq_dense), units="pixel^-1/2", definition="geometric proxy, not calibrated Hall-Petch strength")
        _write_dataset(fields, "distance_to_gb_px", nearest_distance, units="pixel")
        _write_dataset(fields, "distance_to_gb_over_deq", xi, units="1")
        _write_dataset(fields, "nearest_boundary_pair_id", nearest_pair)
        _write_dataset(fields, "nearest_neighbor_grain_id", nearest_neighbor)
        _write_dataset(fields, "nearest_boundary_ambiguous", ambiguous)
        _write_dataset(fields, "triple_junction_mask", triple_junction, definition="boundary pixel touched by multiple interface pairs")
        _write_dataset(fields, "nearest_gb_misorientation_deg", nearest_misorientation, units="degree")
        _write_dataset(fields, "nearest_gb_tangent_xy", nearest_tangent, units="1", definition="PCA tangent of selected same-interface pixels")
        _write_dataset(fields, "nearest_gb_normal_xy", nearest_normal, units="1", definition="2-D perpendicular to selected GB trace tangent")
        _write_dataset(fields, "nearest_gb_trace_quality", nearest_quality, units="1", definition="PCA lambda_max / trace(lambda)")
        _write_dataset(fields, "mprime_max", mprime_dense, units="1", definition="best geometric Luster-Morris compatibility to nearest neighbour")
        _write_dataset(fields, "mprime_best_partner", mprime_partner_dense, definition="neighbour-system index achieving mprime_max")
        _write_dataset(fields, "residual_burgers_min", burgers_dense, units="1", definition="sign-invariant normalized residual Burgers minimum")
        _write_dataset(fields, "residual_burgers_best_partner", burgers_partner_dense)
        _write_dataset(fields, "slip_trace_to_gb_angle_deg", trace_angle, units="degree", definition="2-D slip trace / GB trace angle")
        _write_dataset(fields, "slip_trace_crossing_factor", trace_crossing, units="1", definition="absolute 2-D slip-trace dot GB-normal")
        _write_dataset(fields, "slip_trace_descriptor_valid", trace_valid_dense)
        grains = handle.create_group("grains")
        for name in ("grain_id", "orientation_id", "phi1", "Phi", "phi2", "area_px", "touches_border"):
            _write_dataset(grains, name, grain[name])
        _write_dataset(grains, "euler", np.column_stack([grain["phi1"], grain["Phi"], grain["phi2"]]), units="degree", convention="phi1,Phi,phi2")
        _write_dataset(grains, "rotation_global_to_material", rotations, convention="Q_global_to_material")
        _write_dataset(grains, "equivalent_diameter_px", deq, units="pixel")
        centroids = np.zeros((n_grains, 2), dtype=np.float64)
        for gid in range(n_grains):
            rr, cc = np.where(grain_ids == gid)
            centroids[gid] = (rr.mean(), cc.mean())
        _write_dataset(grains, "centroid_rc_px", centroids, units="pixel")
        _write_dataset(grains, "bbox_r0_r1_c0_c1", np.column_stack([grain["r0"], grain["r1"], grain["c0"], grain["c1"]]))
        neighbor_count = np.zeros(n_grains, dtype=np.int32)
        for ga, gb in zip(pair_a, pair_b, strict=True):
            neighbor_count[ga] += 1
            neighbor_count[gb] += 1
        _write_dataset(grains, "neighbor_count", neighbor_count)
        _write_dataset(grains, "slip_plane_trace_xy", slip_trace, units="1", definition="surface trace from e_z cross slip-plane normal")
        _write_dataset(grains, "slip_plane_trace_valid", slip_trace_valid)
        boundaries = handle.create_group("boundaries")
        _write_dataset(boundaries, "boundary_pair_id", np.arange(n_boundaries, dtype=np.int32))
        _write_dataset(boundaries, "grain_a", pair_a)
        _write_dataset(boundaries, "grain_b", pair_b)
        _write_dataset(boundaries, "contact_length_px", pair_contact, units="pixel-edge count")
        _write_dataset(boundaries, "bbox_r0_r1_c0_c1", pair_bbox)
        _write_dataset(boundaries, "misorientation_deg", misorientation, units="degree")
        _write_dataset(boundaries, "mprime_matrix", mprime_matrix, units="1")
        _write_dataset(boundaries, "mprime_max_side_a", mprime_a, units="1")
        _write_dataset(boundaries, "mprime_max_side_b", mprime_b, units="1")
        _write_dataset(boundaries, "mprime_best_partner_side_a", mprime_partner_a)
        _write_dataset(boundaries, "mprime_best_partner_side_b", mprime_partner_b)
        _write_dataset(boundaries, "residual_burgers_matrix", burgers_matrix, units="1")
        _write_dataset(boundaries, "residual_burgers_min_side_a", burgers_a, units="1")
        _write_dataset(boundaries, "residual_burgers_min_side_b", burgers_b, units="1")
        _write_dataset(boundaries, "residual_burgers_best_partner_side_a", burgers_partner_a)
        _write_dataset(boundaries, "residual_burgers_best_partner_side_b", burgers_partner_b)
        _write_dataset(boundaries, "gb_tangent_xy", pair_tangent, units="1", definition="local same-interface PCA tangent")
        _write_dataset(boundaries, "gb_normal_xy", pair_normal, units="1")
        _write_dataset(boundaries, "gb_trace_quality", pair_quality, units="1")
        _write_dataset(boundaries, "gb_trace_sample_count", pair_samples)
        contact_offsets = np.zeros(n_boundaries + 1, dtype=np.int64)
        contact_offsets[1:] = np.cumsum([len(points) for points in pair_points], dtype=np.int64)
        contact_flat = np.concatenate(pair_points, axis=0) if pair_points else np.empty((0, 2), dtype=np.int32)
        _write_dataset(boundaries, "contact_pixel_offsets", contact_offsets, units="index into contact_pixels_rc")
        _write_dataset(boundaries, "contact_pixels_rc", contact_flat, units="pixel coordinates", definition="concatenated variable-length interface-side pixels")

    figures = _plot_qa(work / "figures", grain_ids, deq_dense, xi, nearest_misorientation, mprime_dense, burgers_dense, nearest_tangent, nearest_quality, trace_crossing)
    m20 = orientation_label[M20_CROP[0] : M20_CROP[1], M20_CROP[2] : M20_CROP[3]]
    m20_unique = int(np.unique(m20).size)
    report = {
        "schema_version": "derived_ebsd_microstructure_v1",
        "candidate_product": True,
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_orientation_attrs": {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in orientation_attrs.items()},
        "grid_shape": list(grain_ids.shape),
        "orientation_label_method": "exact stored float64 Euler triplets; no physical angular threshold; 4-connected components",
        "orientation_numeric_gap": {"smallest_positive_neighbour_max_abs_euler_difference_deg": neighbour_gap, "justification": "measured gap; exact plateaus used"},
        "n_orientation_labels": len(unique_angles),
        "n_grains": n_grains,
        "n_disconnected_duplicate_orientation_labels": int(n_grains - len(unique_angles)),
        "grain_area_px": {"min": int(area.min()), "median": float(np.median(area)), "max": int(area.max())},
        "n_global_border_grains": int(np.count_nonzero(grain["touches_border"])),
        "n_boundaries": n_boundaries,
        "misorientation_deg": {"min": float(misorientation.min()), "median": float(np.median(misorientation)), "max": float(misorientation.max())},
        "mprime": {"min": float(np.nanmin(mprime_dense)), "median": float(np.nanmedian(mprime_dense)), "max": float(np.nanmax(mprime_dense))},
        "residual_burgers": {"min": float(np.nanmin(burgers_dense)), "median": float(np.nanmedian(burgers_dense)), "max": float(np.nanmax(burgers_dense))},
        "trace": {"radius_px": TRACE_RADIUS, "interface_local_samples_median": float(np.median(pair_samples)), "quality_median": float(np.nanmedian(nearest_quality)), "valid_pixel_fraction": float(np.mean(np.isfinite(nearest_quality))), "ambiguous_pixel_fraction": float(np.mean(ambiguous)), "triple_junction_fraction": float(np.mean(triple_junction))},
        "m20_crop_absolute": list(M20_CROP),
        "m20_unique_orientation_labels": m20_unique,
        "m20_grain_ids": int(np.unique(grain_ids[M20_CROP[0] : M20_CROP[1], M20_CROP[2] : M20_CROP[3]]).size),
        "physical_ebsd_scale_verified": False,
        "source_pixel_size_um": 1.84,
        "figures": figures,
        "h5_path": str(h5_path),
        "h5_sha256": _sha256(h5_path),
        "golden_status": "candidate; not promoted to golden",
        "qa": {"valid_partition": bool(np.all(grain_ids >= 0)), "mprime_bounds": [float(np.nanmin(mprime_dense)), float(np.nanmax(mprime_dense))], "burgers_nonnegative": bool(np.nanmin(burgers_dense) >= -1e-7), "trace_angle_bounds": [float(np.nanmin(trace_angle)), float(np.nanmax(trace_angle))]},
    }
    (work / "derived_ebsd_microstructure_v1_report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "filename": str(h5_path),
        "size_bytes": h5_path.stat().st_size,
        "sha256": report["h5_sha256"],
        "source_sha256": report["source_sha256"],
        "schema_version": report["schema_version"],
        "generation_git_sha": _git_sha(),
        "grid_shape": report["grid_shape"],
        "n_grains": n_grains,
        "n_boundaries": n_boundaries,
        "datasets": ["provenance", "fields", "grains", "boundaries"],
        "generation_command": "python scripts/build_derived_ebsd_microstructure.py",
    }
    (work / "derived_ebsd_microstructure_v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (work / "derived_ebsd_microstructure_v1_report.md").write_text(_markdown_report(report))
    return report


def _markdown_report(report: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Candidate derived EBSD microstructure product v1",
            "",
            "> This is a reproducible derived candidate, not a golden dataset. No mechanical or inverse calculation is performed here.",
            "",
            f"- Source: `{report['source']}`",
            f"- Source SHA256: `{report['source_sha256']}`",
            f"- Global shape: `{report['grid_shape']}`",
            f"- Orientation labels: **{report['n_orientation_labels']}** exact stored Euler triplets",
            f"- Four-connected grain components: **{report['n_grains']}**",
            f"- Disconnected duplicate-label components: **{report['n_disconnected_duplicate_orientation_labels']}**",
            "",
            "## Segmentation contract",
            "",
            "Exact float64 `(phi1, Phi, phi2)` triplets define plateau labels. No physical angular threshold is used. Each label is split with four-connectivity, so disconnected regions retain distinct grain IDs.",
            "",
            f"Measured smallest positive neighbouring Euler difference: `{report['orientation_numeric_gap']['smallest_positive_neighbour_max_abs_euler_difference_deg']:.8g} deg`; this is recorded as an audit observation, not used as a segmentation tolerance.",
            "",
            "## Product QA summary",
            "",
            f"- Grain area [px²]: `{report['grain_area_px']}`",
            f"- Internal interfaces: **{report['n_boundaries']}**",
            f"- Misorientation [deg]: `{report['misorientation_deg']}`",
            f"- mprime range/median: `{report['mprime']}`",
            f"- Residual Burgers range/median: `{report['residual_burgers']}`",
            f"- Trace QA: `{report['trace']}`",
            f"- M20 crop `{report['m20_crop_absolute']}`: `{report['m20_unique_orientation_labels']}` orientation labels and `{report['m20_grain_ids']}` global grain IDs",
            "",
            "## Scale and status",
            "",
            "The source advertises `pixel_size_um = 1.84`, but native EBSD scale semantics are unresolved. The product therefore stores pixel-based geometric quantities and marks `physical_ebsd_scale_verified = false`; no micrometre reference fields are created.",
            "",
            f"HDF5 candidate: `{report['h5_path']}` (SHA256 `{report['h5_sha256']}`). **Not promoted to golden.**",
            "",
            "## QA figures",
            "",
            *[f"- `{path}`" for path in report["figures"]],
            "",
            "No k_perp screening, mechanics, FEMU, registration, or SRIX modification was run.",
            "",
        ]
    )


def _markdown_segmentation_report(report: dict[str, object]) -> str:
    segmentation = report["segmentation"]
    misorientation = segmentation["cubic_neighbour_misorientation_sample_deg"]
    return "\n".join(
        [
            "# Exact EBSD plateau segmentation QA",
            "",
            "> The raw Euler-component diagnostic is not a crystallographic misorientation and is not used as a segmentation gate.",
            "",
            f"- Source: `{report['source']}`",
            f"- Global grid: `{report['grid_shape']}`",
            f"- Exact stored triplets: **{segmentation['n_orientation_labels']}**",
            f"- Four-connected components: **{segmentation['n_grains_after_connectivity']}**",
            f"- Additional components beyond labels: **{segmentation['extra_components']}**",
            f"- Labels with one component: `{segmentation['fraction_labels_with_one_component']:.6f}`",
            f"- Grain-area quantiles [px²]: `{segmentation['grain_area_quantiles_px']}`",
            f"- Grain median/mean area [px²]: `{segmentation['grain_area_quantiles_px'][1]}` / `{segmentation['grain_area_mean_px']:.3f}`",
            f"- Fractions of grains with area ≤1/2/4 px²: `{segmentation['fraction_grains_area_le_1_px']:.6f}`, `{segmentation['fraction_grains_area_le_2_px']:.6f}`, `{segmentation['fraction_grains_area_le_4_px']:.6f}`",
            f"- Fractions of pixels in those grains: `{segmentation['fraction_pixels_in_area_le_1_px_grains']:.6f}`, `{segmentation['fraction_pixels_in_area_le_2_px_grains']:.6f}`, `{segmentation['fraction_pixels_in_area_le_4_px_grains']:.6f}`",
            "",
            "## Rotation and true neighbour misorientation checks",
            "",
            f"- Maximum rotation orthogonality error: `{segmentation['rotation_max_orthogonality_error']:.3e}`",
            f"- Rotation determinant range: `{segmentation['rotation_determinant_range']}`",
            f"- Cubic misorientation sample [deg]: `{misorientation}`",
            "",
            "The exact segmentation is statistically dominated by one-pixel and very small components (median area 1 px²). This is a pathological candidate for a grain-mean product and requires source-export/plateau curation before descriptor construction. No angular threshold was introduced.",
            "",
            "QA figures:",
            *[f"- `{path}`" for path in report["segmentation_figures"]],
            "",
            "No HDF5 product was generated or promoted to golden. No mechanics, k_perp screening, FEMU, or SRIX calculation was run.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    arguments = parser.parse_args()
    report = build(arguments.source, arguments.work)
    summary_keys = (
        ("grid_shape", "n_orientation_labels", "n_grains", "n_boundaries", "h5_path", "h5_sha256")
        if report.get("candidate_product")
        else ("grid_shape", "decision")
    )
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
