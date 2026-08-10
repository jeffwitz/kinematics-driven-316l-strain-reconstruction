"""Regression test for the generic SPS/raw-3D same-state identity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.mfront
def test_generic_structural_plane_stress_matches_live_raw_schur(tmp_path: Path) -> None:
    """Run the material-point qualification for both crystal laws."""

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "same_state_schur.json"
    command = (
        "set +u; source /home/jeff/.local/share/tfel/env/env.sh; set -u; "
        f".venv/bin/python scripts/qualify_structural_plane_stress_same_state_schur.py "
        f"--output {output}"
    )
    # The qualification script imports `scripts.diagnose_gps_tangent_blocks`,
    # and `scripts` is a repository package rather than an installed one. Under
    # pytest the root is on the path because `pythonpath = ["."]`, but a
    # subprocess inherits none of that: running the script puts `scripts/` on
    # sys.path, not the root, so the import fails. Pass the root explicitly.
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
    subprocess.run(
        ["bash", "-lc", command],
        cwd=root,
        env=environment,
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    for law in ("srix", "meric"):
        assert report[law]["max_stress_relative_error"] < 1.0e-9
        assert report[law]["max_tangent_relative_error"] < 1.0e-9
        assert len(report[law]["states"]) >= 4
