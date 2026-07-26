#!/usr/bin/env python3
"""Verify evidence assertions and generate stable documentation fragments."""

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


def _json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    value = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    operations = {
        "equals": lambda: actual == expected,
        "not_equals": lambda: actual != expected,
        "less_than": lambda: actual < expected,
        "less_than_or_equal": lambda: actual <= expected,
        "greater_than": lambda: actual > expected,
        "greater_than_or_equal": lambda: actual >= expected,
    }
    try:
        return bool(operations[operator]())
    except KeyError as error:
        raise ValueError(f"unsupported evidence operator: {operator}") from error


def _load_and_verify_sources(
    evidence: dict[str, Any],
) -> list[tuple[dict[str, Any], Any]]:
    loaded: list[tuple[dict[str, Any], Any]] = []
    for source_spec in evidence.get("sources", []):
        source = REPOSITORY_ROOT / source_spec["path"]
        if not source.exists():
            raise FileNotFoundError(f"missing evidence source: {source}")
        document = json.loads(source.read_text(encoding="utf-8"))
        for assertion in source_spec.get("assertions", []):
            actual = _json_pointer(document, assertion["path"])
            expected = (
                _json_pointer(document, assertion["expected_path"])
                if "expected_path" in assertion
                else assertion["expected"]
            )
            if not _compare(actual, assertion["operator"], expected):
                raise ValueError(
                    f"evidence assertion failed for {evidence['id']} at "
                    f"{source_spec['path']}{assertion['path']}: "
                    f"{actual!r} {assertion['operator']} {expected!r}"
                )
        loaded.append((source_spec, document))
    return loaded


def _load_registry(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError("unsupported documentation evidence registry schema")
    for evidence in data["evidence"]:
        evidence["_loaded_sources"] = _load_and_verify_sources(evidence)
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
    print_lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
    ]
    for item in data["evidence"]:
        sources = item.get("sources", [])
        source_text = "<br>".join(
            f"{source.get('role', 'primary')}: `{source['path']}`" for source in sources
        )
        lines.append(
            f"| `{item['id']}` | {item['question']} | {item['fidelity']} | "
            f"{item['parameters']} | **{item['status']}** | {item['conclusion']} | "
            f"{source_text} |"
        )
        print_lines.extend(
            [
                f"**{item['id']} — {item['question']}**",
                f": Fidelity: {item['fidelity']}. Parameters: {item['parameters']}.",
                f": Status: **{item['status']}**.",
                f": Conclusion: {item['conclusion']}",
                f": Sources: {source_text}.",
                "",
            ]
        )
    lines.append("")
    (output / "evidence_registry.inc").write_text("\n".join(lines), encoding="utf-8")
    (output / "evidence_registry_print.inc").write_text(
        "\n".join(print_lines), encoding="utf-8"
    )


def _format_table_cell(cell: Any, documents: dict[str, Any]) -> str:
    if not isinstance(cell, dict):
        return str(cell)
    value = _json_pointer(documents[cell["source"]], cell["path"])
    if "format" in cell:
        value = format(value, cell["format"])
    return f"{value}{cell.get('suffix', '')}"


def _write_tables(data: dict[str, Any], output: Path) -> None:
    documents = {
        source["path"]: document
        for evidence in data["evidence"]
        for source, document in evidence["_loaded_sources"]
    }
    for table in data.get("tables", []):
        headings = table["columns"]
        lines = [
            "<!-- Generated and verified from machine-readable evidence. -->",
            "",
            "| " + " | ".join(headings) + " |",
            "|" + "|".join("---" for _ in headings) + "|",
        ]
        for row in table["rows"]:
            lines.append(
                "| "
                + " | ".join(_format_table_cell(cell, documents) for cell in row)
                + " |"
            )
        lines.append("")
        (output / table["target"]).write_text("\n".join(lines), encoding="utf-8")


def generate(
    registry_path: Path = DEFAULT_REGISTRY,
    output_directory: Path = DEFAULT_OUTPUT,
    static_directory: Path = DEFAULT_STATIC,
) -> None:
    """Verify sources, generate Markdown fragments and copy evidence figures."""
    data = _load_registry(registry_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_current_conclusion(data, output_directory)
    _write_claims(data, output_directory)
    _write_evidence(data, output_directory)
    _write_tables(data, output_directory)
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
