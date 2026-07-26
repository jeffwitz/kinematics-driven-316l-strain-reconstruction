#!/usr/bin/env python3
"""Generate stable documentation fragments from the evidence registry."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "validation" / "documentation_evidence_registry.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs" / "_generated"
DEFAULT_STATIC = REPOSITORY_ROOT / "docs" / "_static"

ALLOWED_STATUSES = {
    "verified",
    "supported",
    "provisional",
    "not demonstrated",
    "not claimed",
}


def _registry_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved.parent
    return REPOSITORY_ROOT


def _source_entries(item: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_sources = item.get("sources")
    if raw_sources is None:
        source = item.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("evidence must declare source or sources")
        return [{"path": source, "role": "primary"}]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("evidence sources must be a non-empty list")
    entries: list[dict[str, str]] = []
    for raw in raw_sources:
        if isinstance(raw, str):
            entries.append({"path": raw, "role": "primary"})
            continue
        if not isinstance(raw, dict):
            raise ValueError("evidence source entries must be strings or mappings")
        source_path = raw.get("path")
        role = raw.get("role", "primary")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("evidence source path must be a non-empty string")
        if not isinstance(role, str) or not role:
            raise ValueError("evidence source role must be a non-empty string")
        entries.append({"path": source_path, "role": role})
    return entries


def _value_at_path(data: Any, path: str) -> Any:
    current = data
    for token in path.split("."):
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(f"cannot descend through {type(current).__name__} at {token!r}")
    return current


def _load_json_source(
    source: str,
    *,
    root: Path,
    cache: dict[str, Any],
) -> Any:
    if source in cache:
        return cache[source]
    path = root / source
    if path.suffix.lower() != ".json":
        raise ValueError(f"source is not JSON and cannot be queried: {source}")
    cache[source] = json.loads(path.read_text(encoding="utf-8"))
    return cache[source]


def _assertion_value(
    assertion: Mapping[str, Any],
    *,
    root: Path,
    cache: dict[str, Any],
    source_key: str = "source",
    path_key: str = "path",
) -> Any:
    source = assertion.get(source_key)
    path = assertion.get(path_key)
    if not isinstance(source, str) or not source:
        raise ValueError(f"assertion {source_key} must be a non-empty string")
    if not isinstance(path, str) or not path:
        raise ValueError(f"assertion {path_key} must be a non-empty string")
    data = _load_json_source(source, root=root, cache=cache)
    return _value_at_path(data, path)


def _validate_assertion(
    assertion: Mapping[str, Any],
    *,
    root: Path,
    cache: dict[str, Any],
) -> None:
    actual = _assertion_value(assertion, root=root, cache=cache)
    operator = assertion.get("operator")
    if operator == "equals":
        passed = actual == assertion.get("expected")
        expectation = repr(assertion.get("expected"))
    elif operator in {"less_than", "less_than_or_equal", "greater_than", "greater_than_or_equal"}:
        expected = assertion.get("expected")
        if not isinstance(expected, int | float):
            raise ValueError(f"{operator} assertion requires a numeric expected value")
        comparisons = {
            "less_than": actual < expected,
            "less_than_or_equal": actual <= expected,
            "greater_than": actual > expected,
            "greater_than_or_equal": actual >= expected,
        }
        passed = bool(comparisons[operator])
        expectation = f"{operator} {expected!r}"
    elif operator in {"less_than_path", "greater_than_path"}:
        other = _assertion_value(
            assertion,
            root=root,
            cache=cache,
            source_key="other_source",
            path_key="other_path",
        )
        factor = float(assertion.get("factor", 1.0))
        threshold = factor * float(other)
        if operator == "less_than_path":
            passed = float(actual) < threshold
        else:
            passed = float(actual) > threshold
        expectation = f"{operator} {factor:g} × {other!r}"
    else:
        raise ValueError(f"unsupported evidence assertion operator: {operator!r}")
    if not passed:
        source = assertion["source"]
        path = assertion["path"]
        raise ValueError(
            f"evidence assertion failed for {source}:{path}: "
            f"actual={actual!r}, expected {expectation}"
        )


def _validate_registry(
    data: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if data.get("schema_version") != 2:
        raise ValueError("unsupported documentation evidence registry schema")
    claims = data.get("claims")
    evidence = data.get("evidence")
    if not isinstance(claims, list) or not isinstance(evidence, list):
        raise ValueError("registry claims and evidence must be lists")

    evidence_ids: set[str] = set()
    source_cache: dict[str, Any] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("evidence entries must be mappings")
        evidence_id = item.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence id must be a non-empty string")
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)
        if item.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"invalid evidence status for {evidence_id}")
        for source in _source_entries(item):
            path = root / source["path"]
            if not path.exists():
                raise FileNotFoundError(f"missing evidence source: {path}")
        assertions = item.get("assertions", [])
        if not isinstance(assertions, list):
            raise ValueError(f"assertions for {evidence_id} must be a list")
        for assertion in assertions:
            if not isinstance(assertion, dict):
                raise ValueError(f"assertions for {evidence_id} must be mappings")
            _validate_assertion(assertion, root=root, cache=source_cache)

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("claim entries must be mappings")
        if claim.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"invalid claim status for {claim.get('question')!r}")
        identifiers = claim.get("evidence_ids")
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError("every claim must cite at least one evidence id")
        missing = sorted(set(identifiers) - evidence_ids)
        if missing:
            raise ValueError(f"claim references unknown evidence ids: {missing}")

    tables = data.get("tables", [])
    if not isinstance(tables, list):
        raise ValueError("registry tables must be a list")
    for table in tables:
        if not isinstance(table, dict):
            raise ValueError("table entries must be mappings")
        output = table.get("output")
        columns = table.get("columns")
        rows = table.get("rows")
        if not isinstance(output, str) or not output.endswith(".inc"):
            raise ValueError("table output must be an .inc file")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"table {output} must define columns")
        if not isinstance(rows, list):
            raise ValueError(f"table {output} must define rows")
    return source_cache


def _load_registry(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = _registry_root(path)
    cache = _validate_registry(data, root=root)
    return data, root, cache


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


def _source_text(item: Mapping[str, Any]) -> str:
    sources = []
    for source in _source_entries(item):
        role = source["role"]
        suffix = "" if role == "primary" else f" ({role})"
        sources.append(f"`{source['path']}`{suffix}")
    return "<br>".join(sources)


def _write_evidence(data: dict[str, Any], output: Path) -> None:
    lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
        "| Evidence ID | Question | Fidelity | Parameters | Status | Main conclusion | Sources |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in data["evidence"]:
        lines.append(
            f"| `{item['id']}` | {item['question']} | {item['fidelity']} | "
            f"{item['parameters']} | **{item['status']}** | {item['conclusion']} | "
            f"{_source_text(item)} |"
        )
    lines.append("")
    (output / "evidence_registry.inc").write_text("\n".join(lines), encoding="utf-8")

    print_lines = [
        "<!-- Generated from validation/documentation_evidence_registry.json. -->",
        "",
    ]
    for item in data["evidence"]:
        sources = "; ".join(
            f"{source['path']} ({source['role']})" for source in _source_entries(item)
        )
        print_lines.extend(
            [
                f"**{item['id']} — {item['question']}**",
                f": Fidelity: {item['fidelity']}. Parameters: {item['parameters']}.",
                f": Status: **{item['status']}**.",
                f": Conclusion: {item['conclusion']}",
                f": Sources: `{sources}`.",
                "",
            ]
        )
    (output / "evidence_registry_print.inc").write_text(
        "\n".join(print_lines), encoding="utf-8"
    )


def _format_table_cell(
    cell: Any,
    *,
    root: Path,
    cache: dict[str, Any],
) -> str:
    if isinstance(cell, str):
        return cell
    if isinstance(cell, int | float):
        return str(cell)
    if not isinstance(cell, dict):
        raise ValueError("table cells must be strings, numbers or mappings")
    source = cell.get("source")
    paths = cell.get("paths")
    if paths is None:
        path = cell.get("path")
        paths = [path]
    if not isinstance(source, str) or not source:
        raise ValueError("dynamic table cells require a source")
    if not isinstance(paths, list) or not paths or any(
        not isinstance(path, str) or not path for path in paths
    ):
        raise ValueError("dynamic table cells require path or paths")
    data = _load_json_source(source, root=root, cache=cache)
    values = [_value_at_path(data, path) for path in paths]

    template = cell.get("template")
    if template is not None:
        if not isinstance(template, str):
            raise ValueError("table cell template must be a string")
        text = template.format(*values)
    else:
        aggregate = cell.get("aggregate")
        if aggregate is None:
            if len(values) != 1:
                raise ValueError("multiple table values require aggregate or template")
            value = values[0]
        elif aggregate == "max":
            value = max(values)
        elif aggregate == "min":
            value = min(values)
        else:
            raise ValueError(f"unsupported table aggregate: {aggregate!r}")
        format_spec = cell.get("format", "")
        if not isinstance(format_spec, str):
            raise ValueError("table cell format must be a string")
        text = format(value, format_spec)
    return f"{cell.get('prefix', '')}{text}{cell.get('suffix', '')}"


def _write_tables(
    data: dict[str, Any],
    output: Path,
    *,
    root: Path,
    cache: dict[str, Any],
) -> None:
    for table in data.get("tables", []):
        columns = [str(column) for column in table["columns"]]
        lines = [
            "<!-- Generated from validation/documentation_evidence_registry.json. -->",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for row in table["rows"]:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(
                    f"table row in {table['output']} must match the column count"
                )
            cells = [
                _format_table_cell(cell, root=root, cache=cache)
                for cell in row
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        (output / table["output"]).write_text("\n".join(lines), encoding="utf-8")


def generate(
    registry_path: Path = DEFAULT_REGISTRY,
    output_directory: Path = DEFAULT_OUTPUT,
    static_directory: Path = DEFAULT_STATIC,
) -> None:
    """Generate Markdown fragments and copy registry-owned evidence figures."""
    data, root, cache = _load_registry(registry_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_current_conclusion(data, output_directory)
    _write_claims(data, output_directory)
    _write_evidence(data, output_directory)
    _write_tables(data, output_directory, root=root, cache=cache)
    for figure in data.get("figures", []):
        source = root / figure["source"]
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
