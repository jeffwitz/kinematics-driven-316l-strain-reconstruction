import json

import numpy as np
import pytest

from fem_inhouse.partitioning.layout import PartitionLayout


def test_balanced_cores_cover_every_element_exactly_once() -> None:
    layout = PartitionLayout(global_shape=(10, 8), partition_shape=(3, 2), padding=2)
    coverage = np.zeros(layout.global_shape, dtype=int)

    partitions = list(layout)
    assert len(partitions) == 6
    assert [partition.partition_id for partition in partitions] == list(range(6))
    assert [partition.core_shape for partition in partitions] == [
        (4, 4),
        (4, 4),
        (3, 4),
        (3, 4),
        (3, 4),
        (3, 4),
    ]

    for partition in partitions:
        coverage[partition.core_element_slice_global] += 1
    np.testing.assert_array_equal(coverage, np.ones(layout.global_shape, dtype=int))


def test_padding_is_clipped_at_global_boundaries() -> None:
    layout = PartitionLayout(global_shape=(10, 8), partition_shape=(2, 2), padding=3)
    first = layout.get(0)
    last = layout.get(3)

    assert first.core_bounds == (0, 5, 0, 4)
    assert first.solve_bounds == (0, 8, 0, 7)
    assert last.core_bounds == (5, 10, 4, 8)
    assert last.solve_bounds == (2, 10, 1, 8)


def test_owned_nodes_cover_global_nodal_grid_once() -> None:
    layout = PartitionLayout(global_shape=(7, 5), partition_shape=(3, 2), padding=1)
    coverage = np.zeros((8, 6), dtype=int)
    for partition in layout:
        coverage[partition.owned_node_slice_global] += 1
    np.testing.assert_array_equal(coverage, np.ones((8, 6), dtype=int))


def test_manifest_records_reproducible_partition_bounds(tmp_path) -> None:
    layout = PartitionLayout(global_shape=(9, 7), partition_shape=(2, 3), padding=2)
    path = tmp_path / "layout.json"
    layout.write_manifest(path)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["global_shape"] == [9, 7]
    assert manifest["partition_shape"] == [2, 3]
    assert manifest["padding"] == 2
    assert len(manifest["partitions"]) == 6
    assert manifest["partitions"][0]["partition_id"] == 0


@pytest.mark.parametrize(
    ("partition_shape", "expected_count", "expected_core_shape"),
    [
        ((5, 5), 25, (720, 620)),
        ((10, 10), 100, (360, 310)),
    ],
)
def test_article_partition_schemes(
    partition_shape,
    expected_count,
    expected_core_shape,
) -> None:
    layout = PartitionLayout(
        global_shape=(3_600, 3_100),
        partition_shape=partition_shape,
        padding=150,
    )

    partitions = list(layout)
    assert len(partitions) == expected_count
    assert {partition.core_shape for partition in partitions} == {expected_core_shape}
    assert partitions[0].solve_bounds[:2] == (0, expected_core_shape[0] + 150)
    assert partitions[-1].solve_bounds[1] == 3_600
    assert partitions[-1].solve_bounds[3] == 3_100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_shape": (0, 2), "partition_shape": (1, 1)},
        {"global_shape": (2, 2), "partition_shape": (0, 1)},
        {"global_shape": (2, 2), "partition_shape": (3, 1)},
        {"global_shape": (2, 2), "partition_shape": (1, 1), "padding": -1},
    ],
)
def test_invalid_layout_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        PartitionLayout(**kwargs)


def test_unknown_partition_id_is_rejected() -> None:
    layout = PartitionLayout(global_shape=(2, 2), partition_shape=(1, 1))
    with pytest.raises(KeyError, match="unknown partition_id"):
        layout.get(1)
