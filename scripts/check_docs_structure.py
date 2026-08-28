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
VALID_DOMAINS = {
    "dic",
    "data",
    "measurement",
    "reconstruction",
    "constitutive",
    "crystal-plasticity",
    "plane-stress",
    "spectral",
    "identification",
    "evidence",
    "software",
    "architecture",
    "operations",
    "scientific",
    "numerics",
    "legacy",
}


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


def load_coverage() -> list[dict[str, str]]:
    source = DOC_ROOT / "_audit" / "scientific_coverage.yml"
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("subjects"), list):
        raise ValueError("scientific coverage must contain a subjects list")
    return data["subjects"]


def matches(entry: dict[str, str], path: str) -> bool:
    pattern = entry.get("path") or entry.get("glob")
    if not pattern:
        return False
    # A basename glob such as ``*.md`` is only for the docs root; fnmatch's
    # '*' also matches slashes, which would make every declaration ambiguous.
    if "/" not in pattern and "/" in path:
        return False
    if "**" not in pattern and len(pattern.split("/")) != len(path.split("/")):
        return False
    return PurePosixPath(path).match(pattern)


def resolve_target(source: str, target: str) -> bool:
    return resolve_path(source, target) is not None


def resolve_path(source: str, target: str) -> str | None:
    target = target.strip().strip("`")
    if not target or target.startswith(":") or target.startswith("#"):
        return None
    if target.startswith("/"):
        candidate = DOC_ROOT / target.lstrip("/")
    else:
        candidate = DOC_ROOT / Path(source).parent / target
    candidate = candidate.resolve()
    candidates = [candidate]
    if candidate.suffix == "":
        candidates += [
            candidate.with_suffix(".md"),
            candidate.with_suffix(".rst"),
            candidate / "index.md",
            candidate / "index.rst",
        ]
    for item in candidates:
        if item.is_file():
            return str(item.relative_to(DOC_ROOT))
    return None


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


def manifest_entry(entries: list[dict[str, str]], path: str) -> dict[str, str] | None:
    exact = [entry for entry in entries if entry.get("path") == path]
    found = exact or [entry for entry in entries if matches(entry, path)]
    return found[0] if len(found) == 1 else None


def reachable_from(entries: list[dict[str, str]], start: str, target: str) -> bool:
    pending = [start]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == target:
            return True
        source = DOC_ROOT / current
        if not source.is_file():
            continue
        for raw_target in navigation_targets(current, source.read_text(errors="replace")):
            resolved = resolve_path(current, raw_target)
            if resolved is not None:
                pending.append(resolved)
    return False


def declared_marker(text: str, label: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(label)}:\*\*\s+(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


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
        if entry.get("domain") not in VALID_DOMAINS:
            errors.append(f"{path}: invalid domain {entry.get('domain')!r}")
        top_level_mode = {
            "tutorials": "tutorial",
            "how-to": "how-to",
            "reference": "reference",
            "explanation": "explanation",
        }.get(path.split("/", 1)[0])
        if (
            top_level_mode
            and entry.get("status") == "current"
            and entry.get("mode") != top_level_mode
        ):
            errors.append(
                f"{path}: current page mode {entry.get('mode')!r} disagrees "
                f"with its {top_level_mode!r} tree"
            )
        if not entry.get("domain") or not entry.get("status") or not entry.get("navigation"):
            errors.append(f"{path}: mode/domain/status/navigation are required")
        if (
            Path(path).parent == Path(".")
            and path != "index.rst"
            and entry.get("status") == "current"
        ):
            errors.append(f"{path}: current substantive page cannot remain at docs root")
        text = (DOC_ROOT / path).read_text(errors="replace")
        for label, value in (("Mode", entry.get("mode")), ("Domain", entry.get("domain"))):
            marker = declared_marker(text, label)
            if marker is not None and marker != value:
                errors.append(
                    f"{path}: declared {label} {marker!r} disagrees with manifest {value!r}"
                )
        for target in navigation_targets(path, text):
            if not resolve_target(path, target):
                errors.append(f"{path}: navigation target does not exist: {target}")

    try:
        coverage = load_coverage()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"scientific coverage: {exc}")
    else:
        for subject in coverage:
            if not isinstance(subject, dict) or not subject.get("name"):
                errors.append("scientific coverage: every subject needs a name")
                continue
            required = {"explanation", "reference", "how_to", "evidence", "status"}
            missing = sorted(required - subject.keys())
            if missing:
                errors.append(
                    f"coverage {subject.get('name')}: missing paths: {', '.join(missing)}"
                )
            status = subject.get("status")
            if status not in {"complete", "routed", "partial"}:
                errors.append(
                    f"coverage {subject.get('name')}: invalid status {status!r}"
                )
            for key, target in subject.items():
                if key in {"name", "status"}:
                    continue
                if not isinstance(target, str) or not (DOC_ROOT / target).is_file():
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target does not exist: {target}"
                    )
                    continue
                entry = manifest_entry(entries, target)
                if entry is None:
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target has no unique "
                        f"manifest entry: {target}"
                    )
                    continue
                if status == "complete" and entry.get("status") in {
                    "historical",
                    "internal",
                    "provisional",
                }:
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target is not current: {target}"
                    )
                if status == "complete" and entry.get("navigation") == "legacy":
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target uses legacy "
                        f"navigation: {target}"
                    )
                expected_modes = {
                    "tutorial": {"tutorial"},
                    "how_to": {"how-to"},
                    "reference": {"reference"},
                    "explanation": {"explanation"},
                    "evidence": {"reference", "explanation", "portal"},
                }
                if key in expected_modes and entry.get("mode") not in expected_modes[key]:
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target has mode "
                        f"{entry.get('mode')!r}, expected {sorted(expected_modes[key])}"
                    )
                roots = {
                    "tutorial": "tutorials/index.md",
                    "how_to": "how-to/index.md",
                    "reference": "reference/index.md",
                    "explanation": "explanation/index.md",
                    "evidence": "evidence/index.md",
                }
                root = roots.get(key)
                if status == "complete" and root and not reachable_from(entries, root, target):
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target is not reachable "
                        f"from {root}: {target}"
                    )

    if errors:
        print("Documentation structure violations:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Documentation structure OK ({len(pages())} pages checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
