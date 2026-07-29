from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import measurement_windows


def test_windows_are_parsed_extracted_and_hashed_deterministically() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    window = measurement_windows(
        [
            {
                "id": "central",
                "bounds": [2, 8, 3, 9],
                "justification": "fixed before correlation",
            }
        ]
    )[0]

    np.testing.assert_array_equal(window.extract(image), image[2:8, 3:9])
    assert window.manifest(image) == window.manifest(image.copy())
    assert window.manifest(image)["shape"] == [6, 6]


def test_windows_reject_duplicate_ids_and_out_of_bounds() -> None:
    row = {"id": "same", "bounds": [0, 2, 0, 2], "justification": "control"}
    with pytest.raises(ValueError, match="unique"):
        measurement_windows([row, row])
    window = measurement_windows([row])[0]
    with pytest.raises(ValueError, match="exceeds"):
        window.extract(np.zeros((1, 1)))
