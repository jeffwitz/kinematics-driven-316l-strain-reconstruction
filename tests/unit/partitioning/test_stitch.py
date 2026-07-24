from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.partitioning.layout import PartitionLayout
from fem_inhouse.partitioning.stitch import (
    extract_partition_field,
    stitch_partition_fields,
    stitch_partition_files,
)


def _partition_fields(global_field, layout, location):
    return {
        partition.partition_id: extract_partition_field(
            global_field,
            layout=layout,
            partition=partition,
            location=location,
        )
        for partition in layout
    }


def test_element_field_round_trip_with_padding_and_components() -> None:
    layout = PartitionLayout(global_shape=(11, 7), partition_shape=(3, 2), padding=2)
    field = np.arange(11 * 7 * 3, dtype=float).reshape(11, 7, 3)
    partition_fields = _partition_fields(field, layout, "element")

    stitched = stitch_partition_fields(layout, partition_fields, location="element")
    np.testing.assert_array_equal(stitched, field)


def test_nodal_field_round_trip_has_unique_interface_ownership() -> None:
    layout = PartitionLayout(global_shape=(11, 7), partition_shape=(3, 2), padding=2)
    field = np.arange(12 * 8 * 2, dtype=float).reshape(12, 8, 2)
    partition_fields = _partition_fields(field, layout, "node")

    stitched = stitch_partition_fields(layout, partition_fields, location="node")
    np.testing.assert_array_equal(stitched, field)


def test_stitch_result_is_independent_of_mapping_insertion_order() -> None:
    layout = PartitionLayout(global_shape=(11, 7), partition_shape=(3, 2), padding=2)
    field = np.arange(11 * 7, dtype=float).reshape(11, 7)
    partition_fields = _partition_fields(field, layout, "element")
    reversed_fields = dict(reversed(partition_fields.items()))

    stitched = stitch_partition_fields(layout, reversed_fields, location="element")

    np.testing.assert_array_equal(stitched, field)


def test_stitch_to_memory_mapped_npy(tmp_path) -> None:
    layout = PartitionLayout(global_shape=(8, 6), partition_shape=(2, 3), padding=1)
    field = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
    partition_fields = _partition_fields(field, layout, "element")
    paths: dict[int, Path] = {}
    for partition_id, values in partition_fields.items():
        path = tmp_path / f"partition_{partition_id}.npy"
        np.save(path, values)
        paths[partition_id] = path

    output_path = tmp_path / "stitched.npy"
    output = stitch_partition_files(
        layout,
        paths,
        location="element",
        output_path=output_path,
    )
    assert isinstance(output, np.memmap)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(np.load(output_path), field)


def test_missing_and_extra_partition_ids_are_reported() -> None:
    layout = PartitionLayout(global_shape=(4, 4), partition_shape=(2, 1))
    field = np.zeros((4, 4))
    partition_fields = _partition_fields(field, layout, "element")

    partition_fields.pop(1)
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        stitch_partition_fields(layout, partition_fields, location="element")

    partition_fields[1] = np.zeros((2, 4))
    partition_fields[2] = np.zeros((2, 4))
    with pytest.raises(ValueError, match=r"extra=\[2\]"):
        stitch_partition_fields(layout, partition_fields, location="element")


def test_shape_and_location_contracts_are_enforced() -> None:
    layout = PartitionLayout(global_shape=(4, 4), partition_shape=(2, 1))
    first = layout.get(0)

    with pytest.raises(ValueError, match="global element field"):
        extract_partition_field(
            np.zeros((5, 4)),
            layout=layout,
            partition=first,
            location="element",
        )
    with pytest.raises(ValueError, match="location"):
        extract_partition_field(
            np.zeros((4, 4)),
            layout=layout,
            partition=first,
            location="invalid",
        )

    partition_fields = _partition_fields(np.zeros((4, 4)), layout, "element")
    partition_fields[0] = np.zeros((3, 4))
    with pytest.raises(ValueError, match="partition 0"):
        stitch_partition_fields(layout, partition_fields, location="element")


def test_trailing_shape_and_output_shape_must_match() -> None:
    layout = PartitionLayout(global_shape=(4, 4), partition_shape=(2, 1))
    partition_fields = _partition_fields(np.zeros((4, 4, 2)), layout, "element")
    partition_fields[1] = np.zeros((2, 4, 3))
    with pytest.raises(ValueError, match="trailing shape"):
        stitch_partition_fields(layout, partition_fields, location="element")

    partition_fields = _partition_fields(np.zeros((4, 4, 2)), layout, "element")
    with pytest.raises(ValueError, match="output shape"):
        stitch_partition_fields(
            layout,
            partition_fields,
            location="element",
            output=np.empty((4, 4)),
        )
