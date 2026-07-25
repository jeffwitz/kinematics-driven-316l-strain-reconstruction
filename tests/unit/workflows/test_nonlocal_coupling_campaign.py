from __future__ import annotations

import json

import numpy as np
import pytest

from fem_inhouse.workflows.nonlocal_coupling_campaign import (
    compute_reference_hardening_modulus,
    estimate_reference_hardening_from_campaign,
)
from fem_inhouse.workflows.partitioned import fingerprint_array


def test_reference_modulus_is_median_ludwik_derivative_on_plastic_core() -> None:
    peeq = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.02],
            [0.0, 0.04, 0.08],
            [0.0, 0.0, 0.0],
        ]
    )
    hardening = np.full_like(peeq, 400.0)

    reference, samples = compute_reference_hardening_modulus(
        peeq,
        hardening,
        core_slice=(slice(1, 3), slice(1, 3)),
        hardening_exponent=0.25,
        first_positive_plastic_strain=1e-6,
    )

    expected = 400.0 * 0.25 * peeq[1:3, 1:3].ravel() ** -0.75
    np.testing.assert_allclose(np.sort(samples), np.sort(expected))
    assert reference == pytest.approx(float(np.median(expected)))


def test_reference_modulus_requires_plastic_elements() -> None:
    with pytest.raises(ValueError, match="contains no elements"):
        compute_reference_hardening_modulus(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            core_slice=(slice(None), slice(None)),
            hardening_exponent=0.245,
            first_positive_plastic_strain=1e-6,
        )


def test_completed_local_campaign_produces_reproducible_reference_report(
    tmp_path,
) -> None:
    inputs = tmp_path / "inputs"
    campaign = tmp_path / "campaign"
    partition_directory = campaign / "partitions" / "0154"
    inputs.mkdir()
    partition_directory.mkdir(parents=True)
    hardening_global = np.full((8, 8), 400.0)
    np.save(inputs / "hardening_coefficient_mpa.npy", hardening_global)
    peeq = np.zeros((4, 3))
    peeq[1:3, 1:3] = [[0.01, 0.02], [0.04, 0.08]]
    np.save(partition_directory / "PEEQ.npy", peeq)
    manifest = {
        "config": {
            "material": {
                "first_positive_plastic_strain": 1e-6,
                "hardening_exponent": 0.25,
            },
            "nonlocal_plasticity": {"enabled": False},
        },
        "inputs": {
            "hardening_coefficient_mpa": fingerprint_array(hardening_global),
        },
        "layout": {
            "partitions": [
                {
                    "partition_id": 154,
                    "solve_bounds": [2, 6, 3, 6],
                    "core_bounds": [3, 5, 4, 6],
                }
            ]
        },
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest))
    (partition_directory / "status.json").write_text(
        json.dumps({"complete": True})
    )
    output = tmp_path / "reference.json"

    report = estimate_reference_hardening_from_campaign(
        input_directory=inputs,
        campaign_directory=campaign,
        partition_id=154,
        output_path=output,
    )

    assert report.partition_id == 154
    assert report.core_element_count == 4
    assert report.plastic_element_count == 4
    assert report.plastic_element_fraction == 1.0
    assert report.coupling_moduli_mpa[0] == 0.0
    assert report.coupling_moduli_mpa[-1] == pytest.approx(
        2.0 * report.reference_hardening_modulus_mpa
    )
    assert json.loads(output.read_text())["source_peeq_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        estimate_reference_hardening_from_campaign(
            input_directory=inputs,
            campaign_directory=campaign,
            partition_id=154,
            output_path=output,
        )
