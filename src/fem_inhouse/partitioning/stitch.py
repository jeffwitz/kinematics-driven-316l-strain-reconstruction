"""Extract padded partition data and stitch uniquely owned cores."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.partitioning.layout import Partition, PartitionLayout, Slice2D

FieldLocation = Literal["element", "node"]


def _expected_global_shape(
    layout: PartitionLayout,
    location: FieldLocation,
) -> tuple[int, int]:
    nx, ny = layout.global_shape
    if location == "element":
        return nx, ny
    if location == "node":
        return nx + 1, ny + 1
    raise ValueError("location must be 'element' or 'node'")


def _expected_local_shape(partition: Partition, location: FieldLocation) -> tuple[int, int]:
    nx, ny = partition.solve_shape
    if location == "element":
        return nx, ny
    if location == "node":
        return nx + 1, ny + 1
    raise ValueError("location must be 'element' or 'node'")


def _global_solve_slice(partition: Partition, location: FieldLocation) -> Slice2D:
    if location == "element":
        return partition.solve_element_slice_global
    if location == "node":
        return partition.solve_node_slice_global
    raise ValueError("location must be 'element' or 'node'")


def _global_core_slice(partition: Partition, location: FieldLocation) -> Slice2D:
    if location == "element":
        return partition.core_element_slice_global
    if location == "node":
        return partition.owned_node_slice_global
    raise ValueError("location must be 'element' or 'node'")


def _local_core_slice(partition: Partition, location: FieldLocation) -> Slice2D:
    if location == "element":
        return partition.core_element_slice_local
    if location == "node":
        return partition.owned_node_slice_local
    raise ValueError("location must be 'element' or 'node'")


def _with_trailing(slice_2d: Slice2D, ndim: int) -> tuple[slice, ...]:
    return (*slice_2d, *(slice(None) for _ in range(ndim - 2)))


def extract_partition_field(
    global_field: ArrayLike,
    *,
    layout: PartitionLayout,
    partition: Partition,
    location: FieldLocation,
) -> NDArray:
    """Extract the padded solve region from an element or nodal global field."""

    values = np.asarray(global_field)
    expected = _expected_global_shape(layout, location)
    if values.ndim < 2 or values.shape[:2] != expected:
        raise ValueError(
            f"global {location} field starts with shape {values.shape[:2]}, expected {expected}"
        )
    selection = _with_trailing(_global_solve_slice(partition, location), values.ndim)
    return values[selection]


def _validate_partition_fields(
    layout: PartitionLayout,
    partition_fields: Mapping[int, ArrayLike],
    location: FieldLocation,
) -> tuple[tuple[int, ...], np.dtype]:
    expected_ids = set(range(layout.count))
    actual_ids = set(partition_fields)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"partition field ids mismatch: missing={missing}, extra={extra}")

    trailing_shape: tuple[int, ...] | None = None
    dtype: np.dtype | None = None
    for partition in layout:
        values = np.asanyarray(partition_fields[partition.partition_id])
        expected_local = _expected_local_shape(partition, location)
        if values.ndim < 2 or values.shape[:2] != expected_local:
            raise ValueError(
                f"partition {partition.partition_id} {location} field starts with "
                f"shape {values.shape[:2]}, expected {expected_local}"
            )
        if trailing_shape is None:
            trailing_shape = values.shape[2:]
            dtype = values.dtype
        elif values.shape[2:] != trailing_shape:
            raise ValueError("all partition fields must share the same trailing shape")
    assert trailing_shape is not None
    assert dtype is not None
    return trailing_shape, dtype


def stitch_partition_fields(
    layout: PartitionLayout,
    partition_fields: Mapping[int, ArrayLike],
    *,
    location: FieldLocation,
    output: NDArray | None = None,
) -> NDArray:
    """Stitch solve-local partition fields into one uniquely owned global field."""

    trailing_shape, dtype = _validate_partition_fields(layout, partition_fields, location)
    expected_shape = (*_expected_global_shape(layout, location), *trailing_shape)
    if output is None:
        output = np.empty(expected_shape, dtype=dtype)
    elif output.shape != expected_shape:
        raise ValueError(f"output shape {output.shape} does not match {expected_shape}")

    for partition in layout:
        values = np.asanyarray(partition_fields[partition.partition_id])
        local = _with_trailing(_local_core_slice(partition, location), values.ndim)
        global_ = _with_trailing(_global_core_slice(partition, location), output.ndim)
        output[global_] = values[local]
    return output


def stitch_partition_files(
    layout: PartitionLayout,
    partition_files: Mapping[int, str | PathLike[str]],
    *,
    location: FieldLocation,
    output_path: str | PathLike[str],
) -> np.memmap:
    """Stitch `.npy` partition files into a memory-mapped global `.npy` file."""

    mapped_fields = {
        partition_id: np.load(Path(path), mmap_mode="r")
        for partition_id, path in partition_files.items()
    }
    trailing_shape, dtype = _validate_partition_fields(layout, mapped_fields, location)
    output_shape = (*_expected_global_shape(layout, location), *trailing_shape)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(destination, mode="w+", dtype=dtype, shape=output_shape)
    stitch_partition_fields(layout, mapped_fields, location=location, output=output)
    output.flush()
    return output
