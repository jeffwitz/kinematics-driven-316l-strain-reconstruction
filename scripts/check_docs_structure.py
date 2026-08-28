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
VALID_BLOCKERS = {
    "missing_tutorial",
    "missing_how_to",
    "missing_reference",
    "missing_explanation",
    "missing_evidence",
    "content_review_required",
    "insufficient_specificity",
    "historical_content_not_fully_migrated",
    "scientific_status_not_frozen",
}
VALID_ROUTING_STATUSES = {"complete", "incomplete", "routed", "partial"}
VALID_CONTENT_STATUSES = {"reviewed", "partial", "stub", "blocked"}
VALID_SCIENTIFIC_STATUSES = {
    "verified",
    "supported",
    "negative",
    "provisional",
    "open",
    "historical",
}
VALID_CLAIM_STATUSES = {
    "verified",
    "supported",
    "negative",
    "provisional",
    "open",
    "not_claimed",
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


def toctree_targets(text: str) -> list[str]:
    """Return only primary menu entries, excluding inline cross-references."""
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
    return targets


def how_to_is_actionable(path: str) -> bool:
    """Require minimum operational content for a reviewed how-to."""
    text = (DOC_ROOT / path).read_text(errors="replace").lower()
    has_inputs = any(token in text for token in ("## prerequisites", "## inputs", "input paths"))
    has_procedure = any(token in text for token in ("```bash", "```console", "```python"))
    has_outputs = any(
        token in text
        for token in ("expected artifact", "expected output", "expected report", "outputs")
    )
    has_verification = any(
        token in text
        for token in ("## verify", "verification", "compare", "check that", "criteria")
    )
    has_boundary = any(
        token in text
        for token in ("failure", "do not", "not qualified", "claim boundary", "error")
    )
    return all((has_inputs, has_procedure, has_outputs, has_verification, has_boundary))


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
        # Reachability for a public menu is deliberately stricter than inline
        # cross-reference reachability: only primary ``toctree`` edges count.
        for raw_target in toctree_targets(source.read_text(errors="replace")):
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
            resolved = resolve_path(path, target)
            if resolved is None:
                errors.append(f"{path}: navigation target does not exist: {target}")
                continue
            target_entry = manifest_entry(entries, resolved)
            if (
                entry.get("status") == "current"
                and target_entry is not None
                and (
                    target_entry.get("status") in {"historical", "legacy"}
                    or target_entry.get("navigation") == "legacy"
                )
            ):
                errors.append(
                    f"{path}: current page links to legacy target: {resolved}"
                )
        if entry.get("status") == "current":
            for target in toctree_targets(text):
                resolved = resolve_path(path, target)
                if resolved is None:
                    continue
                target_entry = manifest_entry(entries, resolved)
                if target_entry is None:
                    continue
                if target_entry.get("status") in {"historical", "internal", "provisional"}:
                    errors.append(
                        f"{path}: current toctree exposes non-current target: {resolved}"
                    )
                if target_entry.get("navigation") == "legacy":
                    errors.append(
                        f"{path}: current toctree exposes legacy target: {resolved}"
                    )

    try:
        coverage = load_coverage()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"scientific coverage: {exc}")
    else:
        for subject in coverage:
            if not isinstance(subject, dict) or not subject.get("name"):
                errors.append("scientific coverage: every subject needs a name")
                continue
            required = {
                "explanation",
                "reference",
                "evidence",
                "status",
                "routing_status",
                "content_status",
                "scientific_status",
                "claim_statuses",
            }
            missing = sorted(required - subject.keys())
            if missing:
                errors.append(
                    f"coverage {subject.get('name')}: missing paths: {', '.join(missing)}"
                )
            status = subject.get("status")
            routing_status = subject.get("routing_status")
            if status not in VALID_ROUTING_STATUSES:
                errors.append(
                    f"coverage {subject.get('name')}: invalid status {status!r}"
                )
            if routing_status not in VALID_ROUTING_STATUSES:
                errors.append(
                    f"coverage {subject.get('name')}: invalid routing_status "
                    f"{routing_status!r}"
                )
            elif status != routing_status:
                errors.append(
                    f"coverage {subject.get('name')}: status and routing_status "
                    "must agree"
                )
            if subject.get("content_status") not in VALID_CONTENT_STATUSES:
                errors.append(
                    f"coverage {subject.get('name')}: invalid content_status "
                    f"{subject.get('content_status')!r}"
                )
            if subject.get("scientific_status") not in VALID_SCIENTIFIC_STATUSES:
                errors.append(
                    f"coverage {subject.get('name')}: invalid scientific_status "
                    f"{subject.get('scientific_status')!r}"
                )
            claims = subject.get("claim_statuses")
            if not isinstance(claims, dict) or not claims:
                errors.append(
                    f"coverage {subject.get('name')}: claim_statuses must be a non-empty mapping"
                )
            elif any(status not in VALID_CLAIM_STATUSES for status in claims.values()):
                errors.append(
                    f"coverage {subject.get('name')}: invalid claim status values {claims!r}"
                )
            blockers = subject.get("blockers", [])
            if not isinstance(blockers, list) or any(
                blocker not in VALID_BLOCKERS for blocker in blockers
            ):
                errors.append(
                    f"coverage {subject.get('name')}: invalid blockers {blockers!r}"
                )
            for key, target in subject.items():
                if key in {
                    "name",
                    "status",
                    "routing_status",
                    "content_status",
                    "scientific_status",
                    "claim_statuses",
                    "blockers",
                }:
                    continue
                if isinstance(target, dict):
                    if target.get("applicable") is False:
                        if not target.get("reason"):
                            errors.append(
                                f"coverage {subject.get('name')}: {key} marked "
                                "not applicable without a reason"
                            )
                        continue
                    target = target.get("path")
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
                if routing_status == "complete" and entry.get("status") in {
                    "historical",
                    "internal",
                    "provisional",
                }:
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target is not current: {target}"
                    )
                if routing_status == "complete" and entry.get("navigation") == "legacy":
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
                if (
                    routing_status == "complete"
                    and root
                    and not reachable_from(entries, root, target)
                ):
                    errors.append(
                        f"coverage {subject.get('name')}: {key} target is not reachable "
                        f"from {root}: {target}"
                    )
            how_to = subject.get("how_to")
            if (
                subject.get("content_status") == "reviewed"
                and isinstance(how_to, str)
                and not how_to_is_actionable(how_to)
            ):
                errors.append(
                    "coverage "
                    f"{subject.get('name')}: reviewed how-to is not actionable: {how_to}"
                )

    if errors:
        print("Documentation structure violations:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Documentation structure OK ({len(pages())} pages checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
