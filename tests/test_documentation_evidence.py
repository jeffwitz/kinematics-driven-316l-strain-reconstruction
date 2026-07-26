from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    assert "maximum stress relative L2" in first["local_baseline_metrics.inc"].decode()
    assert "21 / 23" in first["current_identification_metrics.inc"].decode()
    assert (static / "evidence" / "band_roi_alpha_summary.png").is_file()


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
                "status": "supported",
                "conclusion": "test",
                "sources": [{"path": "does-not-exist.json"}],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing evidence source"):
        generate(registry_path, tmp_path / "generated", tmp_path / "static")


def test_documentation_evidence_rejects_failed_assertion(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"passed": False}), encoding="utf-8")
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
                "status": "verified",
                "conclusion": "test",
                "sources": [{"path": "source.json"}],
                "assertions": [
                    {
                        "source": "source.json",
                        "path": "passed",
                        "operator": "equals",
                        "expected": True,
                    }
                ],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence assertion failed"):
        generate(registry_path, tmp_path / "generated", tmp_path / "static")


def test_documentation_evidence_rejects_unknown_claim_evidence(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("evidence", encoding="utf-8")
    registry = {
        "schema_version": 2,
        "current_conclusion": "test",
        "claims": [
            {
                "question": "test",
                "status": "supported",
                "boundary": "test",
                "evidence_ids": ["unknown"],
            }
        ],
        "evidence": [
            {
                "id": "known",
                "question": "test",
                "fidelity": "test",
                "parameters": "test",
                "status": "supported",
                "conclusion": "test",
                "sources": [{"path": "source.txt"}],
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown evidence ids"):
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
