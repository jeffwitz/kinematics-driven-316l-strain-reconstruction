#!/usr/bin/env python3
"""Validate the Diátaxis manifest and local Sphinx navigation targets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

import yaml

DOC_ROOT = Path(__file__).resolve().parents[1] / "docs"
EXCLUDED_DIRS = {"_build", "_generated", "_static", "_inventories", "_exports"}
VALID_MODES = {"tutorial", "how-to", "reference", "explanation", "portal"}


def pages() -> list[str]:
    return sorted(
        str(path.relative_to(DOC_ROOT))
        for path in DOC_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".rst"}
        and not any(part in EXCLUDED_DIRS for part in path.relative_to(DOC_ROOT).parts)
    )


def load_manifest() -> list[dict[str, str]]:
    data = yaml.safe_load((DOC_ROOT / "diataxis_manifest.yml").read_text())
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ValueError("manifest must contain a pages list")
    return data["pages"]


def matches(entry: dict[str, str], path: str) -> bool:
    pattern = entry.get("path") or entry.get("glob")
    if not pattern:
        return False
    # A basename glob such as ``*.md`` is only for the docs root; fnmatch's
    # '*' also matches slashes, which would make every declaration ambiguous.
    if "/" not in pattern and "/" in path:
        return False
    return PurePosixPath(path).match(pattern)


def resolve_target(source: str, target: str) -> bool:
    target = target.strip().strip("`")
    if not target or target.startswith(":") or target.startswith("#"):
        return True
    if target.startswith("/"):
        candidate = DOC_ROOT / target.lstrip("/")
    else:
        candidate = DOC_ROOT / Path(source).parent / target
    candidates = [candidate]
    if candidate.suffix == "":
        candidates += [
            candidate.with_suffix(".md"),
            candidate.with_suffix(".rst"),
            candidate / "index.md",
            candidate / "index.rst",
        ]
    return any(item.is_file() for item in candidates)


def navigation_targets(path: str, text: str) -> list[str]:
    targets: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        myst = bool(re.match(r"^\s*```\{toctree\}", line))
        rst = bool(re.match(r"^\s*\.\.\s+toctree::", line))
        if not (myst or rst):
            continue
        for candidate in lines[index + 1 :]:
            if myst and candidate.strip().startswith("```"):
                break
            if rst and candidate and not candidate[0].isspace():
                break
            value = candidate.strip()
            if value and not value.startswith(":") and not value.startswith("#"):
                targets.append(value)
    for match in re.finditer(r"\{doc\}`(?:[^<`]*<)?([^>`\s]+)(?:>)?`", text):
        targets.append(match.group(1))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DOC_ROOT / "diataxis_manifest.yml")
    args = parser.parse_args()
    del args  # the canonical location is intentionally not configurable yet
    errors: list[str] = []
    try:
        entries = load_manifest()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"documentation structure: {exc}", file=sys.stderr)
        return 1

    for path in pages():
        exact = [entry for entry in entries if entry.get("path") == path]
        found = exact or [entry for entry in entries if matches(entry, path)]
        if len(found) != 1:
            errors.append(f"{path}: expected one manifest entry, found {len(found)}")
            continue
        entry = found[0]
        if entry.get("mode") not in VALID_MODES:
            errors.append(f"{path}: invalid mode {entry.get('mode')!r}")
        if not entry.get("domain") or not entry.get("status") or not entry.get("navigation"):
            errors.append(f"{path}: mode/domain/status/navigation are required")
        if (
            Path(path).parent == Path(".")
            and path != "index.rst"
            and entry.get("status") == "current"
        ):
            errors.append(f"{path}: current substantive page cannot remain at docs root")
        text = (DOC_ROOT / path).read_text(errors="replace")
        for target in navigation_targets(path, text):
            if not resolve_target(path, target):
                errors.append(f"{path}: navigation target does not exist: {target}")

    if errors:
        print("Documentation structure violations:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Documentation structure OK ({len(pages())} pages checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
