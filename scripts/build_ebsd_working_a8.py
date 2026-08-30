#!/usr/bin/env python3
"""Build the reversible working EBSD product after the A_min=8 cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from skimage.segmentation import find_boundaries

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.fcc_interaction_matrix import slip_systems
from fem_inhouse.identification.grain_boundary_descriptors import cubic_symmetry_matrices

# The report and schema use intentionally long scientific field names.
# ruff: noqa: E501

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
CLEANUP_DEFAULT = Path("/tmp/derived_ebsd_cleanup_v1/amin_8.npz")
WORK_DEFAULT = Path("/tmp/derived_ebsd_microstructure_working_A8_v1")
M20_CROP = (1610, 1630, 1075, 1095)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(group: h5py.Group, name: str, data: NDArray, **attrs: object) -> None:
    kwargs: dict[str, object] = {}
    if data.ndim >= 2 and data.size > 10_000:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    dataset = group.create_dataset(name, data=data, **kwargs)
    for key, value in attrs.items():
        dataset.attrs[key] = value


def _local_geometry(points: NDArray[np.int32], radius: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate tangent, normal and PCA quality at every interface point."""

    points = np.unique(points, axis=0).astype(np.float64)
    n_points = len(points)
    if n_points < 2:
        tangent = np.tile([1.0, 0.0], (n_points, 1))
        normal = np.tile([0.0, 1.0], (n_points, 1))
        return tangent, normal, np.zeros(n_points)
    tree = cKDTree(points)
    # A fixed nearest-neighbour stencil gives the same local-radius contract
    # without a Python query loop for every boundary pixel.  Neighbours beyond
    # the requested radius are discarded when enough in-radius points exist;
    # short interfaces deterministically use their available nearest points.
    k = min(max(int(np.ceil(np.pi * radius * radius)), 3), n_points)
    distances, indices = tree.query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    in_radius = distances <= radius + 1e-12
    valid_counts = np.count_nonzero(in_radius, axis=1)
    use = in_radius.copy()
    for row, count in enumerate(valid_counts):
        if count < 2:
            use[row, : min(2, k)] = True
    local = points[indices]
    weights = use.astype(np.float64)
    count = weights.sum(axis=1)
    mean = np.sum(local * weights[..., None], axis=1) / count[:, None]
    centered = local - mean[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered * weights[..., None], centered) / np.maximum(count - 1.0, 1.0)[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, :, 1]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), np.finfo(float).eps)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    quality = eigenvalues[:, 1] / np.maximum(eigenvalues.sum(axis=1), np.finfo(float).eps)
    return tangent, normal, quality


def _edge_arrays(grain_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    other_rows: list[np.ndarray] = []
    other_cols: list[np.ndarray] = []
    n_grains = int(grain_ids.max()) + 1
    for axis in (0, 1):
        first = np.take(grain_ids, np.arange(grain_ids.shape[axis] - 1), axis=axis)
        second = np.take(grain_ids, np.arange(1, grain_ids.shape[axis]), axis=axis)
        mask = first != second
        r, c = np.where(mask)
        lo = np.minimum(first[mask], second[mask]).astype(np.int64)
        hi = np.maximum(first[mask], second[mask]).astype(np.int64)
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


def _plot_fields(work: Path, fields: dict[str, np.ndarray]) -> list[str]:
    work.mkdir(parents=True, exist_ok=True)
    plots = [
        ("01_working_grain_ids.png", fields["working_grain_id"], "Working grain IDs", "nipy_spectral"),
        ("02_reassigned_pixels.png", fields["reassigned"], "Pixels reassigned by A_min=8 cleanup", "gray"),
        ("03_equivalent_diameter_px.png", fields["diameter"], "Working equivalent diameter [px]", "viridis"),
        ("04_distance_to_gb_over_deq.png", fields["xi"], "Distance to GB / equivalent diameter", "magma"),
        ("05_nearest_gb_misorientation_deg.png", fields["misorientation"], "Nearest GB cubic misorientation [deg]", "viridis"),
        ("06_max_mprime.png", np.nanmax(fields["mprime"], axis=-1), "Max Luster-Morris m'", "viridis"),
        ("07_min_residual_burgers.png", np.nanmin(fields["burgers"], axis=-1), "Min residual Burgers", "magma"),
        ("08_local_trace_quality.png", fields["quality"], "Nearest local GB trace quality", "viridis"),
    ]
    paths: list[str] = []
    for filename, image, title, cmap in plots:
        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        axis.imshow(image[::8, ::8] if image.shape[0] > 1000 else image, origin="upper", cmap=cmap)
        axis.set_title(title)
        path = work / filename
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    image = fields["working_grain_id"]
    axis.imshow(image[:500, :500], origin="upper", cmap="nipy_spectral")
    yy, xx = np.mgrid[0:500:20, 0:500:20]
    tangent = fields["tangent"][:500:20, :500:20]
    axis.quiver(xx, yy, tangent[..., 1], tangent[..., 0], color="white", scale=30)
    axis.set_title("Nearest local GB tangents (QA crop)")
    path = work / "09_local_trace_vectors_m20like_crop.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))
    return paths


def build(source: Path, cleanup_path: Path, work: Path) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=True)
    cleanup = np.load(cleanup_path, allow_pickle=False)
    raw_orientation = cleanup["raw_orientation_label"].astype(np.int32)
    raw_component = cleanup["raw_connected_component_id"].astype(np.int32)
    clean_component_raw = cleanup["clean_grain_id"].astype(np.int32)
    component_areas = np.bincount(raw_component.ravel())
    component_orientation = np.full(len(component_areas), -1, dtype=np.int32)
    component_orientation[raw_component.ravel()] = raw_orientation.ravel()
    raw_target = cleanup["cleanup_target_grain"].astype(np.int32)
    raw_ambiguous = cleanup["cleanup_ambiguous_component"].astype(bool)
    cleanup_reason = cleanup["cleanup_reason"].astype("S32")
    cleanup_fmax = cleanup["cleanup_fmax"].astype(np.float64)
    cleanup_angle = cleanup["cleanup_fusion_misorientation_deg"].astype(np.float64)

    unique_raw, working_inverse = np.unique(clean_component_raw, return_inverse=True)
    working_grain = working_inverse.reshape(raw_component.shape).astype(np.int32)
    n_grains = len(unique_raw)
    working_orientation = component_orientation[unique_raw]
    with h5py.File(source, "r") as handle:
        phi1 = handle["orientation/phi1"][:]
        capital_phi = handle["orientation/Phi"][:]
        phi2 = handle["orientation/phi2"][:]
    angles = np.stack((phi1, capital_phi, phi2), axis=-1)
    records = np.empty(angles.shape[0] * angles.shape[1], dtype=[("phi1", "<f8"), ("Phi", "<f8"), ("phi2", "<f8")])
    records["phi1"], records["Phi"], records["phi2"] = angles.reshape(-1, 3).T
    unique_angles = np.unique(records)
    euler = np.column_stack([unique_angles[name] for name in ("phi1", "Phi", "phi2")])
    rotations_by_label = rotations_from_euler_bunge_deg(euler)
    rotations = rotations_by_label[working_orientation]
    source_euler = euler
    del phi1, capital_phi, phi2, angles, records, unique_angles, rotations_by_label

    area = np.bincount(working_grain.ravel(), minlength=n_grains).astype(np.int64)
    deq = 2.0 * np.sqrt(area / np.pi)
    deq_dense = deq[working_grain]
    edge_key, edge_rows, edge_cols, edge_other_rows, edge_other_cols = _edge_arrays(working_grain)
    unique_keys, edge_pair_id = np.unique(edge_key, return_inverse=True)
    n_boundaries = len(unique_keys)
    pair_a = (unique_keys // n_grains).astype(np.int32)
    pair_b = (unique_keys % n_grains).astype(np.int32)
    pair_points: list[np.ndarray] = []
    pair_tangents: list[np.ndarray] = []
    pair_normals: list[np.ndarray] = []
    pair_quality: list[np.ndarray] = []
    pair_contact = np.zeros(n_boundaries, dtype=np.int64)
    pair_bbox = np.zeros((n_boundaries, 4), dtype=np.int32)
    boundary_pixels: list[np.ndarray] = []
    boundary_pairs: list[np.ndarray] = []
    edge_order = np.argsort(edge_pair_id, kind="stable")
    sorted_pairs = edge_pair_id[edge_order]
    starts = np.flatnonzero(np.r_[True, sorted_pairs[1:] != sorted_pairs[:-1]])
    stops = np.r_[starts[1:], len(sorted_pairs)]
    for pair_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        indices = edge_order[start:stop]
        points = np.unique(
            np.concatenate(
                (
                    np.column_stack((edge_rows[indices], edge_cols[indices])),
                    np.column_stack((edge_other_rows[indices], edge_other_cols[indices])),
                ),
                axis=0,
            ),
            axis=0,
        ).astype(np.int32)
        tangent, normal, quality = _local_geometry(points)
        pair_points.append(points)
        pair_tangents.append(tangent)
        pair_normals.append(normal)
        pair_quality.append(quality)
        pair_contact[pair_id] = stop - start
        pair_bbox[pair_id] = [points[:, 0].min(), points[:, 0].max() + 1, points[:, 1].min(), points[:, 1].max() + 1]
        flat = np.ravel_multi_index((points[:, 0], points[:, 1]), working_grain.shape)
        boundary_pixels.append(flat)
        boundary_pairs.append(np.full(len(points), pair_id, dtype=np.int32))
    all_pixels = np.concatenate(boundary_pixels)
    all_pairs = np.concatenate(boundary_pairs)
    all_tangent = np.concatenate(pair_tangents)
    all_normal = np.concatenate(pair_normals)
    all_quality = np.concatenate(pair_quality)
    order = np.lexsort((all_pairs, all_pixels))
    sorted_pixels = all_pixels[order]
    starts = np.flatnonzero(np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]])
    selected = order[starts]
    unique_pixels = sorted_pixels[starts]
    selected_pair = all_pairs[selected]
    selected_tangent = all_tangent[selected]
    selected_normal = all_normal[selected]
    selected_quality = all_quality[selected]
    boundary_pair_map = np.full(working_grain.shape, -1, dtype=np.int32)
    boundary_pair_map.ravel()[unique_pixels] = selected_pair
    boundary_tangent_map = np.full((*working_grain.shape, 2), np.nan, dtype=np.float32)
    boundary_normal_map = np.full((*working_grain.shape, 2), np.nan, dtype=np.float32)
    boundary_quality_map = np.full(working_grain.shape, np.nan, dtype=np.float32)
    boundary_tangent_map.reshape(-1, 2)[unique_pixels] = selected_tangent
    boundary_normal_map.reshape(-1, 2)[unique_pixels] = selected_normal
    boundary_quality_map.ravel()[unique_pixels] = selected_quality
    ambiguous_boundary = np.zeros(working_grain.shape, dtype=bool)
    for start, stop in zip(starts, np.r_[starts[1:], len(sorted_pixels)], strict=True):
        ambiguous_boundary.ravel()[sorted_pixels[start]] = len(np.unique(all_pairs[order[start:stop]])) > 1
    boundary_mask = find_boundaries(working_grain, connectivity=1, mode="thick")
    distance, nearest_indices = ndimage.distance_transform_edt(~boundary_mask, return_indices=True)
    nearest_pair = boundary_pair_map[nearest_indices[0], nearest_indices[1]]
    nearest_tangent = boundary_tangent_map[nearest_indices[0], nearest_indices[1]]
    nearest_normal = boundary_normal_map[nearest_indices[0], nearest_indices[1]]
    nearest_quality = boundary_quality_map[nearest_indices[0], nearest_indices[1]]
    valid_pair = nearest_pair >= 0
    safe_pair = np.clip(nearest_pair, 0, max(n_boundaries - 1, 0))
    nearest_neighbor = np.where(
        valid_pair,
        np.where(working_grain == pair_a[safe_pair], pair_b[safe_pair], pair_a[safe_pair]),
        -1,
    ).astype(np.int32)
    cleanup_ambiguous_dense = raw_ambiguous[clean_component_raw]
    triple = ambiguous_boundary

    systems = slip_systems()
    directions_material = np.asarray([system.burgers for system in systems], dtype=np.float64)
    normals_material = np.asarray([system.normal for system in systems], dtype=np.float64)
    material_to_global = np.swapaxes(rotations, 1, 2)
    directions = np.einsum("gij,sj->gsi", material_to_global, directions_material)
    normals = np.einsum("gij,sj->gsi", material_to_global, normals_material)
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    traces = np.cross(np.array([0.0, 0.0, 1.0]), normals)
    trace_norm = np.linalg.norm(traces, axis=2)
    slip_trace_valid = trace_norm > 1e-12
    slip_trace = np.full((n_grains, 12, 2), np.nan, dtype=np.float64)
    slip_trace[slip_trace_valid] = traces[slip_trace_valid][:, :2] / trace_norm[slip_trace_valid, None]

    relative = np.einsum("nij,njk->nik", rotations[pair_a], np.swapaxes(rotations[pair_b], 1, 2))
    symmetry = cubic_symmetry_matrices()
    traces_sym = np.einsum("sij,nji->ns", symmetry, relative)
    misorientation = np.degrees(np.arccos(np.clip((np.max(traces_sym, axis=1) - 1.0) / 2.0, -1.0, 1.0))).astype(np.float32)
    first_directions, second_directions = directions[pair_a], directions[pair_b]
    first_normals, second_normals = normals[pair_a], normals[pair_b]
    mprime_matrix = np.clip(np.abs(np.einsum("nai,nbi->nab", first_normals, second_normals)) * np.abs(np.einsum("nai,nbi->nab", first_directions, second_directions)), 0.0, 1.0).astype(np.float32)
    difference = np.linalg.norm(first_directions[:, :, None, :] - second_directions[:, None, :, :], axis=-1)
    sum_norm = np.linalg.norm(first_directions[:, :, None, :] + second_directions[:, None, :, :], axis=-1)
    burgers_matrix = np.minimum(difference, sum_norm).astype(np.float32)
    mprime_a, mprime_partner_a = mprime_matrix.max(axis=2), mprime_matrix.argmax(axis=2).astype(np.int8)
    mprime_b, mprime_partner_b = mprime_matrix.max(axis=1), mprime_matrix.argmax(axis=1).astype(np.int8)
    burgers_a, burgers_partner_a = burgers_matrix.min(axis=2), burgers_matrix.argmin(axis=2).astype(np.int8)
    burgers_b, burgers_partner_b = burgers_matrix.min(axis=1), burgers_matrix.argmin(axis=1).astype(np.int8)
    # Nearest-GB crystallographic descriptors are undefined for cleanup
    # ambiguities or pixels equidistant to several interfaces.
    descriptor_valid_dense = valid_pair & (~cleanup_ambiguous_dense) & (~ambiguous_boundary)
    flat_nearest = nearest_pair.ravel()
    valid = descriptor_valid_dense.ravel()
    indices = np.flatnonzero(valid)
    pair_indices = flat_nearest[indices]
    side_a = working_grain.ravel()[indices] == pair_a[pair_indices]
    nearest_misorientation = np.full(working_grain.size, np.nan, dtype=np.float32)
    nearest_misorientation[indices] = misorientation[pair_indices]
    mprime_dense = np.full((working_grain.size, 12), np.nan, dtype=np.float32)
    burgers_dense = np.full((working_grain.size, 12), np.nan, dtype=np.float32)
    mprime_partner_dense = np.full((working_grain.size, 12), -1, dtype=np.int8)
    burgers_partner_dense = np.full((working_grain.size, 12), -1, dtype=np.int8)
    mprime_dense[indices] = np.where(side_a[:, None], mprime_a[pair_indices], mprime_b[pair_indices])
    mprime_partner_dense[indices] = np.where(side_a[:, None], mprime_partner_a[pair_indices], mprime_partner_b[pair_indices])
    burgers_dense[indices] = np.where(side_a[:, None], burgers_a[pair_indices], burgers_b[pair_indices])
    burgers_partner_dense[indices] = np.where(side_a[:, None], burgers_partner_a[pair_indices], burgers_partner_b[pair_indices])
    tangent_flat = nearest_tangent.reshape(-1, 2)
    normal_flat = nearest_normal.reshape(-1, 2)
    trace_angle = np.full((working_grain.size, 12), np.nan, dtype=np.float32)
    trace_crossing = np.full((working_grain.size, 12), np.nan, dtype=np.float32)
    trace_valid = np.zeros((working_grain.size, 12), dtype=bool)
    for start in range(0, len(indices), 250_000):
        chunk = indices[start : start + 250_000]
        gids = working_grain.ravel()[chunk]
        valid_trace = slip_trace_valid[gids]
        dot_t = np.abs(np.einsum("ni,nai->na", tangent_flat[chunk], slip_trace[gids]))
        dot_n = np.abs(np.einsum("ni,nai->na", normal_flat[chunk], slip_trace[gids]))
        trace_angle[chunk] = np.where(valid_trace, np.degrees(np.arccos(np.clip(dot_t, 0.0, 1.0))), np.nan)
        trace_crossing[chunk] = np.where(valid_trace, dot_n, np.nan)
        trace_valid[chunk] = valid_trace
    trace_valid &= (~cleanup_ambiguous_dense.ravel())[:, None]
    trace_valid &= (~ambiguous_boundary.ravel())[:, None]
    trace_valid &= (~triple.ravel())[:, None]
    trace_valid &= np.isfinite(nearest_quality.ravel())[:, None] & (nearest_quality.ravel()[:, None] >= 0.8)

    final_orientation_dense = working_orientation[working_grain]
    grain_indices = np.arange(n_grains)
    rows, cols = np.indices(working_grain.shape)
    centroid = np.column_stack((np.bincount(working_grain.ravel(), weights=rows.ravel(), minlength=n_grains), np.bincount(working_grain.ravel(), weights=cols.ravel(), minlength=n_grains))) / area[:, None]
    bbox = np.zeros((n_grains, 4), dtype=np.int32)
    touches = np.zeros(n_grains, dtype=bool)
    for gid, location in enumerate(ndimage.find_objects(working_grain + 1)):
        if location is None:
            continue
        bbox[gid] = [location[0].start, location[0].stop, location[1].start, location[1].stop]
        touches[gid] = bool(bbox[gid, 0] == 0 or bbox[gid, 1] == working_grain.shape[0] or bbox[gid, 2] == 0 or bbox[gid, 3] == working_grain.shape[1])
    grain_euler = source_euler[working_orientation]
    neighbor_count = np.zeros(n_grains, dtype=np.int32)
    for ga, gb in zip(pair_a, pair_b, strict=True):
        neighbor_count[ga] += 1
        neighbor_count[gb] += 1

    working_fields = {
        "working_grain_id": working_grain,
        "reassigned": clean_component_raw != raw_component,
        "diameter": deq_dense,
        "xi": (distance / deq_dense).astype(np.float32),
        "misorientation": nearest_misorientation.reshape(working_grain.shape),
        "mprime": mprime_dense.reshape((*working_grain.shape, 12)),
        "burgers": burgers_dense.reshape((*working_grain.shape, 12)),
        "quality": nearest_quality,
        "tangent": nearest_tangent,
    }
    figures = _plot_fields(work / "figures", working_fields)
    h5_path = work / "derived_ebsd_microstructure_working_A8_v1.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="derived_ebsd_microstructure_working_A8_v1", product_status="WORKING_DERIVED_PRODUCT_NOT_GOLDEN", cleanup_Amin_px=8, cleanup_choice="pragmatic", source_sha256=sha256(source), physical_ebsd_scale_verified=False, source_pixel_size_um=1.84, source_pixel_size_semantics="unresolved native EBSD acquisition scale", created_utc=datetime.now(UTC).isoformat())
        provenance = handle.create_group("provenance")
        write_dataset(provenance, "source_path", np.bytes_(str(source)))
        write_dataset(provenance, "source_sha256", np.bytes_(sha256(source)))
        write_dataset(provenance, "cleanup_path", np.bytes_(str(cleanup_path)))
        write_dataset(provenance, "grid_shape", np.asarray(working_grain.shape, dtype=np.int32), units="pixels")
        write_dataset(provenance, "m20_crop_absolute", np.asarray(M20_CROP, dtype=np.int32))
        raw = handle.create_group("raw")
        write_dataset(raw, "orientation_label", raw_orientation, definition="exact stored Euler triplet label")
        write_dataset(raw, "connected_component_id", raw_component, definition="4-connected component of exact source label")
        cleanup_group = handle.create_group("cleanup")
        write_dataset(cleanup_group, "raw_component_id", raw_component)
        write_dataset(cleanup_group, "working_grain_id", working_grain)
        write_dataset(cleanup_group, "reassigned_mask", working_fields["reassigned"])
        write_dataset(cleanup_group, "ambiguous_mask", cleanup_ambiguous_dense)
        write_dataset(cleanup_group, "source_component_area_px", component_areas[raw_component], units="pixel^2")
        write_dataset(cleanup_group, "target_component_id", raw_target[raw_component], definition="-1 for retained/ambiguous components")
        write_dataset(cleanup_group, "reason", cleanup_reason[raw_component])
        write_dataset(cleanup_group, "shared_boundary_fraction", cleanup_fmax[raw_component])
        write_dataset(cleanup_group, "merge_misorientation_deg", cleanup_angle[raw_component], units="degree")
        fields = handle.create_group("fields")
        write_dataset(fields, "valid_mask", np.ones(working_grain.shape, dtype=bool), status="ASSUMED", definition="all finite source Euler pixels; source has no material/indexed mask")
        write_dataset(fields, "raw_component_id", raw_component)
        write_dataset(fields, "working_grain_id", working_grain)
        write_dataset(fields, "orientation_label", final_orientation_dense)
        write_dataset(fields, "cleanup_ambiguous", cleanup_ambiguous_dense)
        write_dataset(fields, "grain_area_px", area[working_grain], units="pixel^2")
        write_dataset(fields, "grain_equivalent_diameter_px", deq_dense, units="pixel")
        write_dataset(fields, "inverse_sqrt_grain_size_proxy", 1.0 / np.sqrt(deq_dense), units="pixel^-1/2")
        write_dataset(fields, "distance_to_gb_px", distance.astype(np.float32), units="pixel")
        write_dataset(fields, "distance_to_gb_over_deq", working_fields["xi"])
        write_dataset(fields, "nearest_boundary_id", nearest_pair)
        write_dataset(fields, "nearest_neighbor_grain_id", nearest_neighbor)
        write_dataset(fields, "nearest_boundary_ambiguous", ambiguous_boundary)
        write_dataset(fields, "nearest_gb_point_rc", np.stack((nearest_indices[0], nearest_indices[1]), axis=-1).astype(np.int32), units="pixel")
        write_dataset(fields, "nearest_gb_local_tangent_xy", nearest_tangent, units="1")
        write_dataset(fields, "nearest_gb_local_normal_xy", nearest_normal, units="1")
        write_dataset(fields, "nearest_gb_local_trace_quality", nearest_quality, units="1")
        write_dataset(fields, "nearest_gb_misorientation_deg", nearest_misorientation.reshape(working_grain.shape), units="degree")
        write_dataset(fields, "mprime_max", working_fields["mprime"], units="1")
        write_dataset(fields, "mprime_best_partner", mprime_partner_dense.reshape((*working_grain.shape, 12)))
        write_dataset(fields, "residual_burgers_min", working_fields["burgers"], units="1")
        write_dataset(fields, "residual_burgers_best_partner", burgers_partner_dense.reshape((*working_grain.shape, 12)))
        write_dataset(fields, "slip_trace_to_gb_angle_deg", trace_angle.reshape((*working_grain.shape, 12)), units="degree")
        write_dataset(fields, "slip_trace_crossing_factor", trace_crossing.reshape((*working_grain.shape, 12)), units="1")
        write_dataset(fields, "slip_trace_descriptor_valid", trace_valid.reshape((*working_grain.shape, 12)))
        grains = handle.create_group("grains")
        write_dataset(grains, "grain_id", grain_indices.astype(np.int32))
        write_dataset(grains, "source_component_id", unique_raw.astype(np.int32))
        write_dataset(grains, "orientation_id", working_orientation)
        write_dataset(grains, "euler", grain_euler, units="degree", convention="phi1,Phi,phi2")
        write_dataset(grains, "rotation_global_to_material", rotations, convention="Q_global_to_material")
        write_dataset(grains, "area_px", area, units="pixel^2")
        write_dataset(grains, "equivalent_diameter_px", deq, units="pixel")
        write_dataset(grains, "centroid_rc_px", centroid, units="pixel")
        write_dataset(grains, "bbox_r0_r1_c0_c1", bbox)
        write_dataset(grains, "touches_global_border", touches)
        write_dataset(grains, "neighbor_count", neighbor_count)
        write_dataset(grains, "slip_plane_trace_xy", slip_trace, units="1")
        write_dataset(grains, "slip_plane_trace_valid", slip_trace_valid)
        boundaries = handle.create_group("boundaries")
        write_dataset(boundaries, "boundary_id", np.arange(n_boundaries, dtype=np.int32))
        write_dataset(boundaries, "grain_a", pair_a)
        write_dataset(boundaries, "grain_b", pair_b)
        write_dataset(boundaries, "contact_length_px", pair_contact, units="pixel-edge count")
        write_dataset(boundaries, "bbox_r0_r1_c0_c1", pair_bbox)
        write_dataset(boundaries, "misorientation_deg", misorientation, units="degree")
        write_dataset(boundaries, "mprime_matrix", mprime_matrix)
        write_dataset(boundaries, "mprime_max_side_a", mprime_a)
        write_dataset(boundaries, "mprime_max_side_b", mprime_b)
        write_dataset(boundaries, "mprime_best_partner_side_a", mprime_partner_a)
        write_dataset(boundaries, "mprime_best_partner_side_b", mprime_partner_b)
        write_dataset(boundaries, "residual_burgers_matrix", burgers_matrix)
        write_dataset(boundaries, "residual_burgers_min_side_a", burgers_a)
        write_dataset(boundaries, "residual_burgers_min_side_b", burgers_b)
        write_dataset(boundaries, "residual_burgers_best_partner_side_a", burgers_partner_a)
        write_dataset(boundaries, "residual_burgers_best_partner_side_b", burgers_partner_b)
        write_dataset(boundaries, "contact_pixel_offsets", np.r_[0, np.cumsum([len(points) for points in pair_points], dtype=np.int64)])
        write_dataset(boundaries, "contact_pixels_rc", np.concatenate(pair_points, axis=0).astype(np.int32), units="pixel")
        write_dataset(boundaries, "contact_local_tangent_xy", np.concatenate(pair_tangents, axis=0).astype(np.float32), units="1")
        write_dataset(boundaries, "contact_local_normal_xy", np.concatenate(pair_normals, axis=0).astype(np.float32), units="1")
        write_dataset(boundaries, "contact_local_trace_quality", np.concatenate(pair_quality, axis=0).astype(np.float32), units="1")

    m20 = (slice(M20_CROP[0], M20_CROP[1]), slice(M20_CROP[2], M20_CROP[3]))
    report: dict[str, object] = {
        "schema_version": "derived_ebsd_microstructure_working_A8_v1",
        "status": "WORKING_DERIVED_PRODUCT_NOT_GOLDEN",
        "source": str(source),
        "source_sha256": sha256(source),
        "cleanup_source": str(cleanup_path),
        "cleanup_Amin_px": 8,
        "grid_shape": list(working_grain.shape),
        "raw_component_count": len(component_areas),
        "working_grain_count": int(n_grains),
        "reassigned_pixel_fraction": float(np.mean(working_fields["reassigned"])),
        "ambiguous_component_count": int(np.count_nonzero(raw_ambiguous)),
        "ambiguous_pixel_fraction": float(np.mean(cleanup_ambiguous_dense)),
        "n_boundaries": int(n_boundaries),
        "grain_area_px": {"min": int(area.min()), "median": float(np.median(area)), "max": int(area.max())},
        "misorientation_deg": {"min": float(misorientation.min()), "median": float(np.median(misorientation)), "max": float(misorientation.max())},
        "mprime": {"min": float(np.nanmin(mprime_dense)), "median": float(np.nanmedian(mprime_dense)), "max": float(np.nanmax(mprime_dense))},
        "residual_burgers": {"min": float(np.nanmin(burgers_dense)), "median": float(np.nanmedian(burgers_dense)), "max": float(np.nanmax(burgers_dense))},
        "trace": {"valid_fraction": float(np.mean(trace_valid)), "quality_median": float(np.nanmedian(nearest_quality)), "ambiguous_boundary_fraction": float(np.mean(ambiguous_boundary)), "local_radius_px": 3.0},
        "m20": {"working_grain_count": int(np.unique(working_grain[m20]).size), "reassigned_fraction": float(np.mean(working_fields["reassigned"][m20])), "global_area_inherited": True},
        "figures": figures,
        "h5_path": str(h5_path),
        "h5_sha256": sha256(h5_path),
        "no_mechanics_run": True,
    }
    (work / "derived_ebsd_microstructure_working_A8_v1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--cleanup", type=Path, default=CLEANUP_DEFAULT)
    parser.add_argument("--work", type=Path, default=WORK_DEFAULT)
    args = parser.parse_args()
    report = build(args.source, args.cleanup, args.work)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
