"""Extract band geometry from the DIC field alone.

Every object here is built from the experimental field and then frozen. No
candidate field is read, so no candidate can influence the geometry it is later
measured against — that asymmetry is deliberate and is what makes the
comparison fair.

Array convention follows the project: fields are ``(nx, ny)`` on canonical
``(x, y)`` axes, so index 0 is x and index 1 is y.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
SkeletonNode = tuple[int, int]

#: Eight-connectivity, so a diagonal band is one object rather than a dotted line.
_CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True, slots=True)
class BandObject:
    """A connected region of the thresholded DIC field."""

    identifier: int
    threshold_quantile: float
    threshold_value: float
    area_pixels: int
    centroid: tuple[float, float]
    orientation_degrees: float
    major_axis_pixels: float
    minor_axis_pixels: float
    elongation: float
    compactness: float
    bounding_box: tuple[int, int, int, int]


def quantile_thresholds(
    field: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
    quantiles: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[float, float]:
    """Return absolute thresholds computed on the valid DIC values only.

    Computing them on the valid subset matters: including invalid entries would
    move the threshold and silently change every downstream object.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if valid_mask is None:
        selected = values.ravel()
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match the field shape")
        selected = values[mask]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("the valid field must be finite and non-empty")
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must lie strictly between zero and one")
    return {float(q): float(np.quantile(selected, q)) for q in quantiles}


def _object_metrics(component: BoolArray) -> tuple[float, float, float, float, float]:
    """Return orientation, axis lengths, elongation and compactness."""

    coordinates = np.argwhere(component).astype(np.float64)
    centred = coordinates - coordinates.mean(axis=0)
    covariance = (centred.T @ centred) / max(len(centred), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major_direction = eigenvectors[:, order[0]]
    orientation = float(np.degrees(np.arctan2(major_direction[1], major_direction[0])))
    major = float(4.0 * np.sqrt(max(eigenvalues[0], 0.0)))
    minor = float(4.0 * np.sqrt(max(eigenvalues[1], 0.0)))
    elongation = float(major / minor) if minor > 0.0 else float("inf")
    perimeter = float(np.count_nonzero(component ^ ndimage.binary_erosion(component)))
    area = float(np.count_nonzero(component))
    compactness = float(4.0 * np.pi * area / perimeter**2) if perimeter > 0.0 else 0.0
    return orientation, major, minor, elongation, compactness


def label_band_objects(
    field: NDArray[np.generic],
    *,
    threshold_value: float,
    threshold_quantile: float,
    valid_mask: NDArray[np.bool_] | None = None,
    minimum_area_pixels: int = 64,
) -> tuple[IntArray, list[BandObject]]:
    """Threshold the field and describe every surviving connected object.

    Only objects below ``minimum_area_pixels`` are dropped, and that bound is
    meant to be preregistered: it is the one place where a size choice can
    change which bands exist.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if minimum_area_pixels < 1:
        raise ValueError("minimum_area_pixels must be positive")
    active = values >= threshold_value
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match the field shape")
        active &= mask

    labels, count = ndimage.label(active, structure=_CONNECTIVITY)
    objects: list[BandObject] = []
    retained = np.zeros_like(labels)
    next_identifier = 0
    for index in range(1, count + 1):
        component = labels == index
        area = int(np.count_nonzero(component))
        if area < minimum_area_pixels:
            continue
        next_identifier += 1
        retained[component] = next_identifier
        orientation, major, minor, elongation, compactness = _object_metrics(component)
        rows, columns = np.where(component)
        objects.append(
            BandObject(
                identifier=next_identifier,
                threshold_quantile=float(threshold_quantile),
                threshold_value=float(threshold_value),
                area_pixels=area,
                centroid=(float(rows.mean()), float(columns.mean())),
                orientation_degrees=orientation,
                major_axis_pixels=major,
                minor_axis_pixels=minor,
                elongation=elongation,
                compactness=compactness,
                bounding_box=(
                    int(rows.min()),
                    int(rows.max()),
                    int(columns.min()),
                    int(columns.max()),
                ),
            )
        )
    # Largest first, so "the main bands" is a stable notion across thresholds.
    order = sorted(range(len(objects)), key=lambda i: -objects[i].area_pixels)
    remap = np.zeros(next_identifier + 1, dtype=int)
    ordered: list[BandObject] = []
    for new_index, old_index in enumerate(order, start=1):
        item = objects[old_index]
        remap[item.identifier] = new_index
        ordered.append(
            BandObject(
                identifier=new_index,
                threshold_quantile=item.threshold_quantile,
                threshold_value=item.threshold_value,
                area_pixels=item.area_pixels,
                centroid=item.centroid,
                orientation_degrees=item.orientation_degrees,
                major_axis_pixels=item.major_axis_pixels,
                minor_axis_pixels=item.minor_axis_pixels,
                elongation=item.elongation,
                compactness=item.compactness,
                bounding_box=item.bounding_box,
            )
        )
    return remap[retained], ordered


def zhang_suen_thinning(mask: NDArray[np.bool_], *, maximum_passes: int = 200) -> BoolArray:
    """Thin a binary mask to an eight-connected one-pixel skeleton.

    Implemented here rather than taken from scikit-image, which is not a
    dependency of this project. The algorithm is deterministic: the same mask
    always gives the same skeleton.
    """

    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = image

    def _neighbours(grid: BoolArray) -> list[BoolArray]:
        # P2..P9 clockwise from north, the ordering the algorithm is stated in.
        return [
            grid[:-2, 1:-1],
            grid[:-2, 2:],
            grid[1:-1, 2:],
            grid[2:, 2:],
            grid[2:, 1:-1],
            grid[2:, :-2],
            grid[1:-1, :-2],
            grid[:-2, :-2],
        ]

    for _ in range(maximum_passes):
        changed = False
        for step in (0, 1):
            neighbours = _neighbours(padded)
            centre = padded[1:-1, 1:-1]
            total = sum(n.astype(np.int_) for n in neighbours)
            transitions = sum(
                (~neighbours[i] & neighbours[(i + 1) % 8]).astype(np.int_)
                for i in range(8)
            )
            if step == 0:
                first = neighbours[0] & neighbours[2] & neighbours[4]
                second = neighbours[2] & neighbours[4] & neighbours[6]
            else:
                first = neighbours[0] & neighbours[2] & neighbours[6]
                second = neighbours[0] & neighbours[4] & neighbours[6]
            removable = (
                centre
                & (total >= 2)
                & (total <= 6)
                & (transitions == 1)
                & ~first
                & ~second
            )
            if removable.any():
                padded[1:-1, 1:-1] &= ~removable
                changed = True
        if not changed:
            break
    return np.ascontiguousarray(padded[1:-1, 1:-1])


def _skeleton_neighbours(
    skeleton: BoolArray,
) -> dict[SkeletonNode, list[SkeletonNode]]:
    points = [tuple(int(v) for v in p) for p in np.argwhere(skeleton)]
    lookup = set(points)
    graph: dict[SkeletonNode, list[SkeletonNode]] = {}
    for row, column in points:
        neighbours = [
            (row + dr, column + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (row + dr, column + dc) in lookup
        ]
        graph[(row, column)] = neighbours
    return graph


def prune_skeleton_spurs(
    skeleton: NDArray[np.bool_],
    *,
    minimum_branch_pixels: int = 8,
) -> BoolArray:
    """Remove short branches, keeping the main path intact.

    A spur is a chain that ends at a degree-one pixel and is shorter than the
    declared bound. Pruning is iterated because removing one spur can expose
    another.
    """

    if minimum_branch_pixels < 1:
        raise ValueError("minimum_branch_pixels must be positive")
    current = np.asarray(skeleton, dtype=bool).copy()
    for _ in range(64):
        graph = _skeleton_neighbours(current)
        endpoints = [p for p, n in graph.items() if len(n) == 1]
        removed = False
        for endpoint in endpoints:
            chain = [endpoint]
            previous, node = endpoint, graph[endpoint][0]
            while len(graph.get(node, [])) == 2 and len(chain) <= minimum_branch_pixels:
                chain.append(node)
                following = [n for n in graph[node] if n != previous]
                if not following:
                    break
                previous, node = node, following[0]
            if len(graph.get(node, [])) > 2 and len(chain) < minimum_branch_pixels:
                for row, column in chain:
                    current[row, column] = False
                removed = True
        if not removed:
            break
    return current


def order_centreline(skeleton: NDArray[np.bool_]) -> FloatArray:
    """Return the main path through the skeleton, as an ordered polyline.

    Two breadth-first passes locate the two extremities, then the geometrically
    shortest route between them is taken. Hop count alone is not enough: a
    one-pixel residue beside the trunk offers a diagonal detour with the same
    number of nodes, and picking it would put a spurious kink in the centreline.
    Weighting edges by Euclidean length makes the straight trunk win.
    """

    graph = _skeleton_neighbours(np.asarray(skeleton, dtype=bool))
    if not graph:
        raise ValueError("the skeleton is empty")

    def _farthest(start: SkeletonNode) -> SkeletonNode:
        seen = {start}
        queue = deque([start])
        last = start
        while queue:
            node = queue.popleft()
            last = node
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return last

    first = _farthest(min(graph))
    second = _farthest(first)

    distances: dict[SkeletonNode, float] = {first: 0.0}
    previous: dict[SkeletonNode, SkeletonNode | None] = {first: None}
    heap: list[tuple[float, SkeletonNode]] = [(0.0, first)]
    while heap:
        cost, node = heappop(heap)
        if cost > distances.get(node, float("inf")):
            continue
        if node == second:
            break
        for neighbour in graph[node]:
            step = float(np.hypot(neighbour[0] - node[0], neighbour[1] - node[1]))
            candidate = cost + step
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                previous[neighbour] = node
                heappush(heap, (candidate, neighbour))
    if second not in previous:
        raise ValueError("the skeleton is not connected")

    path: list[SkeletonNode] = [second]
    while True:
        parent = previous[path[-1]]
        if parent is None:
            break
        path.append(parent)
    return np.asarray(path[::-1], dtype=np.float64)


def smooth_centreline(path: NDArray[np.generic], *, window: int = 9) -> FloatArray:
    """Apply a declared moving-average smoothing, endpoints preserved."""

    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("path must have shape (n, 2)")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number of samples")
    if window == 1 or len(points) <= window:
        return np.ascontiguousarray(points)
    kernel = np.ones(window) / window
    smoothed = points.copy()
    for axis in (0, 1):
        padded = np.pad(points[:, axis], window // 2, mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    # The ends anchor the band; smoothing must not shorten it.
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return np.ascontiguousarray(smoothed)


def resample_polyline(path: NDArray[np.generic], *, spacing_pixels: float) -> FloatArray:
    """Resample a polyline at a regular arc-length interval."""

    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must have shape (n, 2) with at least two points")
    if not np.isfinite(spacing_pixels) or spacing_pixels <= 0.0:
        raise ValueError("spacing_pixels must be finite and positive")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    if arc[-1] == 0.0:
        raise ValueError("the path has zero length")
    targets = np.arange(0.0, arc[-1] + 0.5 * spacing_pixels, spacing_pixels)
    targets = targets[targets <= arc[-1]]
    return np.ascontiguousarray(
        np.stack([np.interp(targets, arc, points[:, axis]) for axis in (0, 1)], axis=-1)
    )


def tangents_and_normals(path: NDArray[np.generic]) -> tuple[FloatArray, FloatArray]:
    """Return unit tangents and left normals along a polyline."""

    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must have shape (n, 2) with at least two points")
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    if not np.all(lengths > 0.0):
        raise ValueError("the path contains repeated points")
    tangents = tangents / lengths
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=-1)
    return np.ascontiguousarray(tangents), np.ascontiguousarray(normals)


def band_corridor(
    shape: tuple[int, int],
    centreline: NDArray[np.generic],
    *,
    half_width_pixels: float,
) -> BoolArray:
    """Return the set of pixels within a declared distance of the centreline."""

    points = np.asarray(centreline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("centreline must have shape (n, 2)")
    if not np.isfinite(half_width_pixels) or half_width_pixels <= 0.0:
        raise ValueError("half_width_pixels must be finite and positive")
    seeds = np.ones(shape, dtype=bool)
    rows = np.clip(np.round(points[:, 0]).astype(int), 0, shape[0] - 1)
    columns = np.clip(np.round(points[:, 1]).astype(int), 0, shape[1] - 1)
    seeds[rows, columns] = False
    distance = ndimage.distance_transform_edt(seeds)
    return np.ascontiguousarray(np.asarray(distance) <= half_width_pixels)
