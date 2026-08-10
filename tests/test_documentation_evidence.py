from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_documentation_evidence.py"
SPEC = importlib.util.spec_from_file_location("generate_documentation_evidence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate = MODULE.generate


def test_documentation_evidence_generation_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    static = tmp_path / "static"
    generate(output_directory=output, static_directory=static)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    generate(output_directory=output, static_directory=static)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first == second
    assert "Current conclusion" in first["current_conclusion.inc"].decode()
    assert "1.437e-04" in first["local_fem_metrics.inc"].decode()
    assert (static / "evidence" / "local_morphological_defect.png").is_file()


def test_documentation_evidence_rejects_missing_source(tmp_path: Path) -> None:
    registry = {
        "schema_version": 2,
        "current_conclusion": "test",
        "claims": [],
        "evidence": [
            {
                "id": "missing",
                "question": "test",
                "fidelity": "test",
                "parameters": "test",
                "status": "test",
                "conclusion": "test",
                "sources": [
                    {
                        "path": "validation/does-not-exist.json",
                        "role": "primary",
                        "assertions": [],
                    }
                ],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing evidence source"):
        generate(registry_path, tmp_path / "generated", tmp_path / "static")


def test_documentation_evidence_rejects_failed_semantic_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"passed": false}\n', encoding="utf-8")
    monkeypatch.setattr(MODULE, "REPOSITORY_ROOT", tmp_path)
    registry = {
        "schema_version": 2,
        "current_conclusion": "test",
        "claims": [],
        "evidence": [
            {
                "id": "failed",
                "question": "test",
                "fidelity": "test",
                "parameters": "test",
                "status": "test",
                "conclusion": "test",
                "sources": [
                    {
                        "path": "source.json",
                        "assertions": [
                            {
                                "path": "/passed",
                                "operator": "equals",
                                "expected": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence assertion failed"):
        generate(registry_path, tmp_path / "generated", tmp_path / "static")


def test_public_documentation_architecture() -> None:
    root = Path(__file__).resolve().parents[1]
    explanation = root / "docs" / "explanation"
    required_explanations = {
        "from_dic_to_mechanics.md",
        "local_baseline.md",
        "missing_spatial_interaction.md",
        "micromorphic_model.md",
        "parameter_identification.md",
        "current_evidence.md",
        "scope_and_prediction.md",
    }
    assert required_explanations <= {path.name for path in explanation.iterdir()}
    assert len((root / "README.md").read_text(encoding="utf-8").splitlines()) <= 85
    conf = (root / "docs" / "conf.py").read_text(encoding="utf-8")
    assert '"archive/**"' in conf


HOW_TO = Path(__file__).resolve().parents[1] / "docs" / "how-to"
REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference"


def test_every_accepted_constitutive_option_is_documented() -> None:
    """No `constitutive_options` key may exist without a line in the reference.

    These keys are the only way to steer a backend, and they are added in the
    module that consumes them, far from the page that lists them. Three of them
    -- the SRIX smoothing pair and the failure diagnostics -- were shipped and
    used in campaigns while the reference still described the previous set, so
    a reader could not learn they existed. This closes that gap by construction
    rather than by vigilance.
    """

    import re

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fem_inhouse"
        / "core"
        / "plane_stress_material.py"
    ).read_text(encoding="utf-8")
    accepted = set(re.findall(r'options\.pop\(\s*"([a-z_]+)"', source))
    assert accepted, "the option keys are no longer read with options.pop"

    reference = (REFERENCE / "configuration.md").read_text(encoding="utf-8")
    undocumented = sorted(key for key in accepted if f"`{key}`" not in reference)
    assert not undocumented, (
        "constitutive_options accepted by the code but absent from "
        f"docs/reference/configuration.md: {undocumented}"
    )


def _python_blocks(page: Path) -> list[str]:
    """Every fenced ```python block of a page, in order."""

    import re

    return re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)


def test_the_onboarding_page_offers_the_recommended_backend() -> None:
    """The short path must not send a newcomer to the slower route.

    Cheap to check and easy to lose: the SRIX how-to said "the 3D behaviour is
    condensed" and offered only that backend long after the GPS route had been
    qualified, so every new reader was pointed at the wrong one.
    """

    page = (HOW_TO / "run_316l_crystal_plasticity.md").read_text(encoding="utf-8")
    assert "mfront-native-generalised-plane-stress" in page
    assert "gps_composite_fd_tangent" in page
    # And it must still name the reference, or the reader cannot check anything.
    assert "mfront-3d-condensed-plane-stress" in page


@pytest.mark.mfront
def test_the_onboarding_python_block_runs_verbatim() -> None:
    """Execute the page's own snippet, exactly as a reader would copy it.

    A documented block that has drifted from the API is worse than no block:
    it fails on the first thing a newcomer tries. This runs the real one, so
    the page cannot rot silently.
    """

    import os

    if os.environ.get("MFRONT_BEHAVIOUR_LIBRARY") is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis")

    blocks = _python_blocks(HOW_TO / "run_316l_crystal_plasticity.md")
    assert blocks, "the onboarding page lost its runnable block"
    namespace: dict[str, object] = {}
    exec(compile(blocks[0], "run_316l_crystal_plasticity.md", "exec"), namespace)
    stress = namespace["trial"].stress_in_plane_mpa[0]  # type: ignore[attr-defined]
    assert stress.shape == (3,)
    assert bool(np.isfinite(stress).all())
    # Driven to 2 percent in-plane strain on 316L, the response is plastic and
    # of the order of a few hundred MPa; a zero or a NaN would mean the block
    # ran but integrated nothing.
    assert 50.0 < float(np.abs(stress).max()) < 2000.0
