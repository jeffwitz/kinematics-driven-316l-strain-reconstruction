#!/usr/bin/env python3
"""Conservatively clean an already grain-mean EBSD orientation map.

The source Euler triplets are treated as upstream region colours.  Exact
colour regions are split only by spatial connectivity; no angular clustering
is introduced.  Small connected components are then assigned once, and only
to large seed components, using shared boundary contact first and cubic
misorientation as a deterministic tie-break.  The raw component map is kept
in every local variant artifact so the operation is reversible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from skimage.measure import label as skimage_label

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.identification.grain_boundary_descriptors import cubic_misorientation_angle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
DEFAULT_WORK = Path("/tmp/derived_ebsd_cleanup_v1")
M20_CROP = (1610, 1630, 1075, 1095)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_orientation(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(source, "r") as handle:
        angles = np.stack(
            [
                handle["orientation/phi1"][:],
                handle["orientation/Phi"][:],
                handle["orientation/phi2"][:],
            ],
            axis=-1,
        )
    records = np.empty(
        angles.shape[0] * angles.shape[1],
        dtype=[("phi1", "<f8"), ("Phi", "<f8"), ("phi2", "<f8")],
    )
    records["phi1"], records["Phi"], records["phi2"] = angles.reshape(-1, 3).T
    unique, inverse = np.unique(records, return_inverse=True)
    unique_angles = np.column_stack([unique[name] for name in ("phi1", "Phi", "phi2")])
    return inverse.reshape(angles.shape[:2]).astype(np.int32), unique_angles


def connected_components_by_exact_label(
    orientation_labels: np.ndarray, *, connectivity: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split exact source labels with scikit-image connectivity.

    Returns the component map, the component-to-orientation map and areas.
    Component IDs are deterministic in increasing exact orientation-label
    order and in scikit-image's scan order within each label.
    """

    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (4-neighbour) or 2 (8-neighbour)")
    labels = np.asarray(orientation_labels, dtype=np.int32)
    component_map = np.full(labels.shape, -1, dtype=np.int32)
    slices = ndimage.find_objects(labels + 1, max_label=int(labels.max()) + 1)
    component_orientation: list[int] = []
    component_area: list[int] = []
    next_id = 0
    for orientation_id, region in enumerate(slices):
        if region is None:
            continue
        view = labels[region] == orientation_id
        local = skimage_label(view, connectivity=connectivity, background=0)
        count = int(local.max())
        if count == 0:
            continue
        target = component_map[region]
        target[view] = local[view] + next_id - 1
        sizes = np.bincount(local.ravel(), minlength=count + 1)[1:]
        component_orientation.extend([orientation_id] * count)
        component_area.extend(int(value) for value in sizes)
        next_id += count
    return (
        component_map,
        np.asarray(component_orientation, dtype=np.int32),
        np.asarray(component_area, dtype=np.int64),
    )


def boundary_adjacency(component_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique component pairs and their 4-neighbour contact lengths."""

    n_components = int(component_map.max()) + 1
    keys: list[np.ndarray] = []
    for first, second in (
        (component_map[:, :-1], component_map[:, 1:]),
        (component_map[:-1, :], component_map[1:, :]),
    ):
        mask = (first >= 0) & (second >= 0) & (first != second)
        low = np.minimum(first[mask], second[mask]).astype(np.int64)
        high = np.maximum(first[mask], second[mask]).astype(np.int64)
        keys.append(low * n_components + high)
    encoded = np.concatenate(keys)
    unique, counts = np.unique(encoded, return_counts=True)
    return unique // n_components, unique % n_components, counts.astype(np.int64)


def _neighbour_lists(
    first: np.ndarray, second: np.ndarray, contact: np.ndarray, n_components: int
) -> list[list[tuple[int, int]]]:
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(n_components)]
    for left, right, length in zip(first, second, contact, strict=True):
        left_i, right_i, length_i = int(left), int(right), int(length)
        neighbours[left_i].append((right_i, length_i))
        neighbours[right_i].append((left_i, length_i))
    return neighbours


def clean_components(
    component_map: np.ndarray,
    component_orientation: np.ndarray,
    component_area: np.ndarray,
    rotations: np.ndarray,
    minimum_area: int,
) -> dict[str, object]:
    """Perform one conservative seed-only cleanup pass."""

    n_components = len(component_area)
    seeds = component_area > minimum_area
    neighbours = boundary_adjacency(component_map)
    first, second, contact = neighbours
    adjacency = _neighbour_lists(first, second, contact, n_components)
    target = np.full(n_components, -1, dtype=np.int32)
    reason = np.full(n_components, "seed", dtype="U32")
    fmax = np.full(n_components, np.nan, dtype=np.float64)
    fusion_misorientation = np.full(n_components, np.nan, dtype=np.float64)
    ambiguous = np.zeros(n_components, dtype=bool)
    cache: dict[tuple[int, int], float] = {}

    for component_id, _area in enumerate(component_area):
        if seeds[component_id]:
            target[component_id] = component_id
            continue
        candidates = [
            (other, length)
            for other, length in adjacency[component_id]
            if seeds[other]
        ]
        if not candidates:
            ambiguous[component_id] = True
            reason[component_id] = "ambiguous_no_seed"
            continue
        total_contact = sum(length for _, length in candidates)
        max_contact = max(length for _, length in candidates)
        fmax[component_id] = max_contact / total_contact
        top = [other for other, length in candidates if length == max_contact]
        if len(top) == 1 and (len(candidates) == 1 or fmax[component_id] >= 2.0 / 3.0):
            target[component_id] = top[0]
            reason[component_id] = "single_neighbor" if len(candidates) == 1 else "dominant_contact"
            continue

        # A unique topological maximum below 2/3 is still resolved by the
        # crystallographic tie-break, as prescribed by the lexicographic rule.
        if len(top) == 1:
            top = [other for other, _ in candidates]
            reason_prefix = "misorientation_tiebreak"
        else:
            reason_prefix = "contact_tie_misorientation"
        orientation_id = int(component_orientation[component_id])
        distances: list[tuple[float, int]] = []
        for other in top:
            other_orientation = int(component_orientation[other])
            key = tuple(sorted((orientation_id, other_orientation)))
            if key not in cache:
                cache[key] = cubic_misorientation_angle(rotations[key[0]], rotations[key[1]])
            distances.append((cache[key], other))
        distances.sort(key=lambda item: (item[0], item[1]))
        fusion_misorientation[component_id] = distances[0][0]
        if len(distances) > 1 and np.isclose(
            distances[0][0], distances[1][0], rtol=0.0, atol=1e-12
        ):
            ambiguous[component_id] = True
            reason[component_id] = "ambiguous_crystallographic_tie"
        else:
            target[component_id] = distances[0][1]
            reason[component_id] = reason_prefix

    final_component = component_map.copy()
    assigned = target[component_map]
    final_component[assigned >= 0] = assigned[assigned >= 0]
    reassigned = final_component != component_map
    final_orientation = component_orientation[final_component]
    final_area = np.bincount(final_component.ravel(), minlength=n_components)
    small = component_area <= minimum_area
    return {
        "final_component": final_component,
        "final_orientation": final_orientation.astype(np.int32),
        "target_component": target,
        "reason": reason,
        "ambiguous_component": ambiguous,
        "fmax": fmax,
        "fusion_misorientation_deg": fusion_misorientation,
        "reassigned_mask": reassigned,
        "final_area": final_area.astype(np.int64),
        "seed_component": seeds,
        "small_component": small,
        "adjacency_pairs": neighbours,
    }


def _stats(
    result: dict[str, object], orientation_labels: np.ndarray, minimum_area: int
) -> dict[str, object]:
    final_component = result["final_component"]
    reassigned = result["reassigned_mask"]
    ambiguous = result["ambiguous_component"]
    fmax = result["fmax"]
    fusion_angle = result["fusion_misorientation_deg"]
    final_area = result["final_area"]
    small = result["small_component"]
    eligible = small & np.isfinite(fmax)
    r0, r1, c0, c1 = M20_CROP
    raw_m20 = orientation_labels[r0:r1, c0:c1]
    clean_m20 = final_component[r0:r1, c0:c1]
    finite_fmax = fmax[eligible]
    finite_angles = fusion_angle[np.isfinite(fusion_angle)]
    def quantile(values: np.ndarray) -> list[float]:
        return (
            np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
            if len(values)
            else []
        )
    return {
        "minimum_area_px": minimum_area,
        "raw_component_count": len(final_area),
        "final_grain_count": int(np.unique(final_component).size),
        "small_component_count": int(np.count_nonzero(small)),
        "eligible_small_component_count": int(np.count_nonzero(eligible)),
        "ambiguous_component_count": int(np.count_nonzero(ambiguous & small)),
        "reassigned_pixel_count": int(np.count_nonzero(reassigned)),
        "reassigned_pixel_fraction": float(np.mean(reassigned)),
        "ambiguous_pixel_fraction": float(np.mean(ambiguous[final_component])),
        "fmax_quantiles": quantile(finite_fmax),
        "fusion_misorientation_quantiles_deg": quantile(finite_angles),
        "final_area_quantiles_px": quantile(final_area[final_area > 0]),
        "m20_raw_orientation_count": int(np.unique(raw_m20).size),
        "m20_clean_grain_count": int(np.unique(clean_m20).size),
        "m20_reassigned_pixel_fraction": float(np.mean(reassigned[r0:r1, c0:c1])),
        "m20_ambiguous_pixel_fraction": float(np.mean(ambiguous[clean_m20])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--minimum-area", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    orientation_labels, unique_angles = _read_orientation(args.source)
    print(f"loaded source map {orientation_labels.shape}, {len(unique_angles)} exact labels")
    rotations = rotations_from_euler_bunge_deg(unique_angles)
    component_map, component_orientation, component_area = connected_components_by_exact_label(
        orientation_labels, connectivity=1
    )
    print(f"4-connected components: {len(component_area)}")
    component_map_8, _, component_area_8 = connected_components_by_exact_label(
        orientation_labels, connectivity=2
    )
    del component_map_8
    print(f"8-connected components: {len(component_area_8)}")
    report: dict[str, object] = {
        "schema_version": "ebsd_cleanup_screening_v1",
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "grid_shape": list(orientation_labels.shape),
        "exact_orientation_label_count": len(unique_angles),
        "four_connected_component_count": len(component_area),
        "eight_connected_component_count": len(component_area_8),
        "source_label_component_count_delta": int(len(component_area) - len(unique_angles)),
        "variants": {},
    }
    np.savez_compressed(
        args.work / "raw_maps.npz",
        orientation_label=orientation_labels,
        raw_connected_component_id=component_map,
        component_orientation_id=component_orientation,
        component_area_px=component_area,
    )
    for minimum_area in args.minimum_area:
        result = clean_components(
            component_map, component_orientation, component_area, rotations, minimum_area
        )
        variant_name = f"amin_{minimum_area}"
        np.savez_compressed(
            args.work / f"{variant_name}.npz",
            raw_orientation_label=orientation_labels,
            raw_connected_component_id=component_map,
            clean_grain_id=result["final_component"],
            clean_orientation_label=result["final_orientation"],
            cleanup_target_grain=result["target_component"],
            cleanup_ambiguous_component=result["ambiguous_component"],
            cleanup_reason=result["reason"],
            cleanup_fmax=result["fmax"],
            cleanup_fusion_misorientation_deg=result["fusion_misorientation_deg"],
        )
        report["variants"][variant_name] = _stats(result, orientation_labels, minimum_area)
        print(variant_name, report["variants"][variant_name])
    report["artifact_policy"] = "local compressed maps; no HDF5/golden promotion"
    (args.work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
