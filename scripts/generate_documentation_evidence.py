#!/usr/bin/env python3
"""Generate stable documentation fragments from the evidence registry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "validation" / "documentation_evidence_registry.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs" / "_generated"
DEFAULT_STATIC = REPOSITORY_ROOT / "docs" / "_static"


def _load_registry(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported documentation evidence registry schema")
    for evidence in data["evidence"]:
        source = REPOSITORY_ROOT / evidence["source"]
        if not source.exists():
            raise FileNotFoundError(f"missing evidence source: {source}")
    return data


def _write_current_conclusion(data: dict[str, Any], output: Path) -> None:
    text = (
        "<!-- Generated from validation/documentation_evidence_registry.json. -->\n\n"
        ":::{admonition} Current conclusion\n"
        ":class: important\n\n"
        f"{data['current_conclusion']}\n"
        ":::\n"
    )
    (output / "current_conclusion.inc").write_text(text, encoding="utf-8")


def _write_claims(data: dict[str, Any], output: Path) -> None:
    lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
        "| Question | Status | Claim boundary | Evidence |",
        "|---|---|---|---|",
    ]
    for claim in data["claims"]:
        evidence = ", ".join(f"`{item}`" for item in claim["evidence_ids"])
        lines.append(
            f"| {claim['question']} | **{claim['status']}** | "
            f"{claim['boundary']} | {evidence} |"
        )
    lines.append("")
    (output / "claims_matrix.inc").write_text("\n".join(lines), encoding="utf-8")


def _write_evidence(data: dict[str, Any], output: Path) -> None:
    lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
        "| Evidence ID | Question | Fidelity | Parameters | Status | Main conclusion | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in data["evidence"]:
        source = item["source"]
        lines.append(
            f"| `{item['id']}` | {item['question']} | {item['fidelity']} | "
            f"{item['parameters']} | **{item['status']}** | {item['conclusion']} | "
            f"`{source}` |"
        )
    lines.append("")
    (output / "evidence_registry.inc").write_text("\n".join(lines), encoding="utf-8")
    print_lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
    ]
    for item in data["evidence"]:
        print_lines.extend(
            [
                f"**{item['id']} — {item['question']}**",
                f": Fidelity: {item['fidelity']}. Parameters: {item['parameters']}.",
                f": Status: **{item['status']}**.",
                f": Conclusion: {item['conclusion']}",
                f": Source: `{item['source']}`.",
                "",
            ]
        )
    (output / "evidence_registry_print.inc").write_text(
        "\n".join(print_lines), encoding="utf-8"
    )


def generate(
    registry_path: Path = DEFAULT_REGISTRY,
    output_directory: Path = DEFAULT_OUTPUT,
    static_directory: Path = DEFAULT_STATIC,
) -> None:
    """Generate Markdown fragments and copy registry-owned evidence figures."""
    data = _load_registry(registry_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_current_conclusion(data, output_directory)
    _write_claims(data, output_directory)
    _write_evidence(data, output_directory)
    for figure in data.get("figures", []):
        source = REPOSITORY_ROOT / figure["source"]
        target = static_directory / figure["target"]
        if not source.exists():
            raise FileNotFoundError(f"missing documentation figure source: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    args = parser.parse_args()
    generate(args.registry, args.output, args.static)


if __name__ == "__main__":
    main()
