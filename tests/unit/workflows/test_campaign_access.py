"""The read-only door onto a saved campaign, and every guard behind it.

Six workflow modules read archived campaigns exclusively through this one. Its
happy paths were exercised indirectly; none of its refusals was. Those refusals
are the point of the module -- it exists so that a truncated, mismatched or
silently regenerated archive fails loudly instead of producing a plausible
number, and an untested guard is a guard nobody has seen work.

These tests double as the specification of the on-disk contract: what a campaign
directory must contain, and what each field of `status.json` is checked against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    load_partition_status,
    load_verified_partition_field,
    partition_from_manifest,
    validate_mechanical_campaign_pair,
)

MANIFEST_SHA = "a" * 64
PARTITION_ID = 2


def _layout() -> PartitionLayout:
    return PartitionLayout(global_shape=(12, 8), partition_shape=(2, 2), padding=1)


def _layout_data() -> dict[str, Any]:
    layout = _layout()
    return {
        "global_shape": list(layout.global_shape),
        "partition_shape": list(layout.partition_shape),
        "padding": layout.padding,
        "partitions": [
            {
                "partition_id": index,
                "core_bounds": list(layout.get(index).core_bounds),
                "solve_bounds": list(layout.get(index).solve_bounds),
            }
            for index in range(layout.count)
        ],
    }


def _campaign(root: Path, *, field: np.ndarray | None = None) -> Path:
    """A minimal complete campaign directory with one archived field."""

    values = np.arange(6.0).reshape(2, 3) if field is None else field
    directory = root / "partitions" / f"{PARTITION_ID:04d}"
    directory.mkdir(parents=True)
    path = directory / "PEEQ.npy"
    np.save(path, values)
    (directory / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "partition_id": PARTITION_ID,
                "manifest_sha256": MANIFEST_SHA,
                "outputs": {"PEEQ": fingerprint_file(path)},
            }
        )
    )
    return root


def _status(root: Path) -> dict[str, Any]:
    return load_partition_status(
        root, partition_id=PARTITION_ID, manifest_sha256=MANIFEST_SHA
    )


class TestLoadJsonObject:
    def test_a_missing_file_is_named_in_the_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing JSON file"):
            load_json_object(tmp_path / "absent.json")

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json_object(tmp_path)

    def test_a_json_array_is_refused(self, tmp_path: Path) -> None:
        """The callers index by key. A list would fail later and less clearly."""

        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="expected a JSON object"):
            load_json_object(path)

    def test_an_object_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"a": 1}))

        assert load_json_object(path) == {"a": 1}


class TestPartitionFromManifest:
    def test_a_manifest_without_layout_metadata_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lacks layout metadata"):
            partition_from_manifest({}, PARTITION_ID)

    def test_a_non_object_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lacks layout metadata"):
            partition_from_manifest({"layout": "flat"}, PARTITION_ID)

    def test_a_three_dimensional_shape_is_refused(self) -> None:
        data = _layout_data()
        data["global_shape"] = [12, 8, 4]

        with pytest.raises(ValueError, match="two entries"):
            partition_from_manifest({"layout": data}, PARTITION_ID)

    def test_an_unlisted_partition_is_refused(self) -> None:
        data = _layout_data()
        data["partitions"] = [
            item for item in data["partitions"] if item["partition_id"] != PARTITION_ID
        ]

        with pytest.raises(ValueError, match=f"does not identify partition {PARTITION_ID}"):
            partition_from_manifest({"layout": data}, PARTITION_ID)

    def test_a_duplicated_partition_entry_is_refused(self) -> None:
        """Two declarations mean the archive cannot say which bounds were used."""

        data = _layout_data()
        data["partitions"].append(dict(data["partitions"][PARTITION_ID]))

        with pytest.raises(ValueError, match="does not identify partition"):
            partition_from_manifest({"layout": data}, PARTITION_ID)

    def test_bounds_that_contradict_the_recomputed_layout_are_refused(self) -> None:
        """The guard that matters: the layout is recomputed, not trusted.

        A manifest declaring bounds that its own `global_shape` and
        `partition_shape` do not produce means the archive was written by a
        different partitioning than the one it claims.
        """

        data = _layout_data()
        data["partitions"][PARTITION_ID]["core_bounds"] = [0, 1, 0, 1]

        with pytest.raises(ValueError, match="bounds disagree with the declared layout"):
            partition_from_manifest({"layout": data}, PARTITION_ID)

    def test_a_consistent_manifest_returns_the_recomputed_partition(self) -> None:
        layout, partition = partition_from_manifest({"layout": _layout_data()}, PARTITION_ID)

        assert layout == _layout()
        assert partition.core_bounds == _layout().get(PARTITION_ID).core_bounds
        assert partition.solve_bounds == _layout().get(PARTITION_ID).solve_bounds


class TestLoadPartitionStatus:
    def test_an_incomplete_partition_is_refused(self, tmp_path: Path) -> None:
        """Reading a partition still being solved would report a partial field."""

        root = _campaign(tmp_path)
        path = root / "partitions" / f"{PARTITION_ID:04d}" / "status.json"
        status = json.loads(path.read_text())
        status["complete"] = False
        path.write_text(json.dumps(status))

        with pytest.raises(RuntimeError, match="partition is not complete"):
            _status(root)

    def test_a_status_naming_another_partition_is_refused(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path)
        path = root / "partitions" / f"{PARTITION_ID:04d}" / "status.json"
        status = json.loads(path.read_text())
        status["partition_id"] = PARTITION_ID + 1
        path.write_text(json.dumps(status))

        with pytest.raises(ValueError, match="identifies another partition"):
            _status(root)

    def test_a_status_from_a_different_campaign_is_refused(self, tmp_path: Path) -> None:
        """The digest is what stops results of two configurations being mixed."""

        root = _campaign(tmp_path)

        with pytest.raises(RuntimeError, match="does not match campaign manifest"):
            load_partition_status(root, partition_id=PARTITION_ID, manifest_sha256="b" * 64)

    def test_a_missing_status_reports_the_path_it_looked_for(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"status\.json"):
            _status(tmp_path)

    def test_a_matching_status_is_returned(self, tmp_path: Path) -> None:
        status = _status(_campaign(tmp_path))

        assert status["partition_id"] == PARTITION_ID
        assert status["manifest_sha256"] == MANIFEST_SHA


class TestLoadVerifiedPartitionField:
    def test_a_field_matching_its_hash_is_returned_as_float64(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path)

        values = load_verified_partition_field(
            root, partition_id=PARTITION_ID, status=_status(root), name="PEEQ"
        )

        assert values.dtype == np.float64
        np.testing.assert_array_equal(values, np.arange(6.0).reshape(2, 3))

    def test_a_missing_field_is_named(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path)

        with pytest.raises(FileNotFoundError, match="missing saved partition field ABSENT"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=_status(root), name="ABSENT"
            )

    def test_a_field_the_status_does_not_declare_is_refused(self, tmp_path: Path) -> None:
        """Present on disk but undeclared: nothing vouches for it."""

        root = _campaign(tmp_path)
        status = _status(root)
        undeclared = root / "partitions" / f"{PARTITION_ID:04d}" / "EXTRA.npy"
        np.save(undeclared, np.zeros((2, 3)))

        with pytest.raises(ValueError, match="does not declare output EXTRA"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=status, name="EXTRA"
            )

    def test_a_field_edited_after_the_run_fails_its_hash(self, tmp_path: Path) -> None:
        """The guard the whole archive rests on. Rewriting the array with the
        same shape and dtype must still be caught."""

        root = _campaign(tmp_path)
        status = _status(root)
        np.save(
            root / "partitions" / f"{PARTITION_ID:04d}" / "PEEQ.npy",
            np.arange(6.0).reshape(2, 3) + 1.0,
        )

        with pytest.raises(RuntimeError, match="fails its status hash"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=status, name="PEEQ"
            )

    def test_a_non_numeric_field_is_refused(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path, field=np.array([["a", "b"], ["c", "d"]]))

        with pytest.raises(ValueError, match="is not numeric"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=_status(root), name="PEEQ"
            )

    def test_a_field_carrying_nan_is_refused(self, tmp_path: Path) -> None:
        """A diverged solve that was archived anyway must not read back clean."""

        root = _campaign(tmp_path, field=np.array([[1.0, np.nan], [3.0, 4.0]]))

        with pytest.raises(ValueError, match="contains non-finite values"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=_status(root), name="PEEQ"
            )

    def test_an_infinite_value_is_refused_too(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path, field=np.array([[1.0, np.inf], [3.0, 4.0]]))

        with pytest.raises(ValueError, match="contains non-finite values"):
            load_verified_partition_field(
                root, partition_id=PARTITION_ID, status=_status(root), name="PEEQ"
            )

    def test_reading_without_mmap_is_supported(self, tmp_path: Path) -> None:
        root = _campaign(tmp_path)

        values = load_verified_partition_field(
            root,
            partition_id=PARTITION_ID,
            status=_status(root),
            name="PEEQ",
            mmap_mode=None,
        )

        np.testing.assert_array_equal(values, np.arange(6.0).reshape(2, 3))


class TestValidateMechanicalCampaignPair:
    """A local and a coupled campaign may differ in exactly one thing."""

    @staticmethod
    def _pair() -> tuple[dict[str, Any], dict[str, Any]]:
        base_solver = {"increments": 20, "constitutive_backend": "mfront", "mfront_threads": 4}
        local = {
            "inputs": {"evm": "sha-1"},
            "layout": _layout_data(),
            "config": {
                "mesh": {"nx": 12, "ny": 8},
                "material": {"young_modulus_mpa": 205000.0},
                "solver": dict(base_solver),
                "nonlocal_plasticity": {"enabled": False},
            },
        }
        coupled = json.loads(json.dumps(local))
        coupled["config"]["nonlocal_plasticity"] = {
            "enabled": True,
            "coupling_modulus_mpa": 200.0,
        }
        return local, coupled

    def test_an_otherwise_identical_pair_is_accepted(self) -> None:
        local, coupled = self._pair()

        validate_mechanical_campaign_pair(local, coupled)

    def test_the_thread_count_alone_may_differ(self) -> None:
        """Threads change the wall time, not the answer."""

        local, coupled = self._pair()
        coupled["config"]["solver"]["mfront_threads"] = 1

        validate_mechanical_campaign_pair(local, coupled)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda c: c.__setitem__("inputs", {"evm": "sha-2"}), "identical input fields"),
            (lambda c: c.__setitem__("layout", {"padding": 9}), "same partition layout"),
            (lambda c: c["config"].__setitem__("mesh", {"nx": 1}), "differ in mesh"),
            (lambda c: c["config"].__setitem__("material", {}), "differ in material"),
            (
                lambda c: c["config"]["solver"].__setitem__("increments", 40),
                "differ in mechanical solver",
            ),
        ],
    )
    def test_any_other_mechanical_difference_is_refused(self, mutate, message: str) -> None:
        local, coupled = self._pair()
        mutate(coupled)

        with pytest.raises(ValueError, match=message):
            validate_mechanical_campaign_pair(local, coupled)

    def test_a_reference_that_is_itself_coupled_is_refused(self) -> None:
        """Comparing two coupled runs would not isolate the coupling."""

        local, coupled = self._pair()
        local["config"]["nonlocal_plasticity"] = {
            "enabled": True,
            "coupling_modulus_mpa": 50.0,
        }

        with pytest.raises(ValueError, match="must be local or use H_chi=0"):
            validate_mechanical_campaign_pair(local, coupled)

    def test_a_reference_coupled_with_zero_modulus_is_accepted(self) -> None:
        """H_chi = 0 is mechanically local, whatever the flag says. Accepting it
        is what allows the pair to share one solver configuration."""

        local, coupled = self._pair()
        local["config"]["nonlocal_plasticity"] = {
            "enabled": True,
            "coupling_modulus_mpa": 0.0,
        }

        validate_mechanical_campaign_pair(local, coupled)

    def test_a_candidate_without_coupling_is_refused(self) -> None:
        local, coupled = self._pair()
        coupled["config"]["nonlocal_plasticity"] = {"enabled": False}

        with pytest.raises(ValueError, match="must enable nonlocal plasticity"):
            validate_mechanical_campaign_pair(local, coupled)
