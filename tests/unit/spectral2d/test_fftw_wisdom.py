from __future__ import annotations

import json

from fem_inhouse.spectral2d.wisdom import load_wisdom, save_wisdom, wisdom_file


class FakeFFTW:
    def __init__(self) -> None:
        self.imported = None

    def export_wisdom(self):
        return (b"a", b"b", b"c")

    def import_wisdom(self, wisdom):
        self.imported = wisdom
        return True


def test_wisdom_round_trip_and_metadata_rejection(tmp_path) -> None:
    metadata = {"shape": [7, 7, 2], "threads": 1, "dtype": "float64"}
    fftw = FakeFFTW()
    path = save_wisdom(tmp_path, metadata, fftw)
    assert path == wisdom_file(tmp_path, metadata)
    assert load_wisdom(tmp_path, metadata, fftw)
    assert fftw.imported == (b"a", b"b", b"c")
    assert not load_wisdom(tmp_path, {**metadata, "threads": 2}, fftw)


def test_corrupt_wisdom_is_ignored(tmp_path) -> None:
    metadata = {"shape": [7, 7, 2]}
    path = wisdom_file(tmp_path, metadata)
    path.write_text(json.dumps({"metadata": metadata, "wisdom": "not-base64"}))
    assert not load_wisdom(tmp_path, metadata, FakeFFTW())
