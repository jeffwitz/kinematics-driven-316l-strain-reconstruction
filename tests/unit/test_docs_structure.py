import re

from scripts.check_docs_structure import (
    DOC_ROOT,
    declared_marker,
    how_to_is_actionable,
    load_coverage,
    load_manifest,
    manifest_entry,
    navigation_targets,
    pages,
    reachable_from,
    resolve_path,
    toctree_targets,
)


def test_manifest_covers_documentation_inventory() -> None:
    assert load_manifest()
    assert len(pages()) >= 140


def test_scientific_coverage_declares_core_quadrants() -> None:
    subjects = load_coverage()
    assert len(subjects) >= 10
    for subject in subjects:
        assert subject["name"]
        for quadrant in ("explanation", "reference", "evidence"):
            assert subject[quadrant].endswith((".md", ".rst"))
        how_to = subject.get("how_to")
        if isinstance(how_to, dict) and how_to.get("applicable") is False:
            assert how_to.get("reason")
        else:
            assert isinstance(how_to, str) and how_to.endswith((".md", ".rst"))


def test_coverage_targets_use_current_manifest_entries() -> None:
    entries = load_manifest()
    subjects = load_coverage()
    for subject in subjects:
        for key in ("explanation", "reference", "how_to", "evidence"):
            target = subject.get(key)
            if isinstance(target, dict) and target.get("applicable") is False:
                continue
            entry = manifest_entry(entries, target)
            assert entry is not None
            if subject["routing_status"] == "complete":
                assert entry["status"] == "current"
                assert entry["navigation"] != "legacy"


def test_explicit_markers_and_canonical_reachability() -> None:
    entries = load_manifest()
    assert declared_marker(
        DOC_ROOT.joinpath("explanation/native-srix/optimization_strategy.md")
        .read_text(),
        "Domain",
    ) == manifest_entry(entries, "explanation/native-srix/optimization_strategy.md")["domain"]
    assert reachable_from(
        entries,
        "explanation/index.md",
        "explanation/native-srix/optimization_strategy.md",
    )


def test_coverage_declares_separate_semantic_statuses() -> None:
    for subject in load_coverage():
        assert subject["status"] == subject["routing_status"]
        assert subject["content_status"] in {"reviewed", "partial", "stub", "blocked"}
        assert subject["scientific_status"] in {
            "verified",
            "supported",
            "negative",
            "provisional",
            "open",
            "historical",
        }
        assert isinstance(subject["claim_statuses"], dict)
        assert subject["claim_statuses"]
        assert set(subject["claim_statuses"].values()) <= {
            "verified",
            "supported",
            "negative",
            "provisional",
            "open",
            "not_claimed",
        }


def test_reviewed_how_tos_are_actionable() -> None:
    for subject in load_coverage():
        if subject["content_status"] != "reviewed":
            continue
        how_to = subject.get("how_to")
        if isinstance(how_to, str):
            assert how_to_is_actionable(how_to)


def test_phase3_canonical_reference_targets_are_current() -> None:
    entries = load_manifest()
    canonical = {
        "reference/evidence/evidence_registry.md": "evidence",
        "reference/evidence/claims_matrix.md": "evidence",
        "reference/evidence/qualification_vocabulary.md": "evidence",
        "reference/evidence/selection_indicators.md": "evidence",
        "reference/scientific/constitutive_models.md": "constitutive",
        "reference/scientific/observation_operator.md": "measurement",
        "reference/scientific/ebsd_orientation_contract.md": "crystal-plasticity",
    }
    for path, domain in canonical.items():
        entry = manifest_entry(entries, path)
        assert entry is not None
        assert entry["mode"] == "reference"
        assert entry["status"] == "current"
        assert entry["navigation"] != "legacy"
        assert entry["domain"] == domain


def test_current_toctrees_do_not_expose_noncurrent_pages() -> None:
    entries = load_manifest()
    for path in pages():
        entry = manifest_entry(entries, path)
        assert entry is not None
        if entry["status"] != "current":
            continue
        text = (DOC_ROOT / path).read_text()
        for raw_target in toctree_targets(text):
            target = resolve_path(path, raw_target)
            if target is None:
                continue
            target_entry = manifest_entry(entries, target)
            assert target_entry is not None
            assert target_entry["status"] == "current"
            assert target_entry["navigation"] != "legacy"


def test_current_pages_do_not_link_to_legacy_pages() -> None:
    entries = load_manifest()
    for path in pages():
        entry = manifest_entry(entries, path)
        assert entry is not None
        if entry["status"] != "current":
            continue
        text = (DOC_ROOT / path).read_text()
        for raw_target in navigation_targets(path, text):
            target = resolve_path(path, raw_target)
            if target is None:
                continue
            target_entry = manifest_entry(entries, target)
            assert target_entry is not None
            assert target_entry["status"] not in {"historical", "legacy"}
            assert target_entry["navigation"] != "legacy"


def test_agent_read_first_routes_are_not_legacy() -> None:
    entries = load_manifest()
    source = "agent/README.md"
    text = (DOC_ROOT / source).read_text()
    for raw_target in re.findall(r"\]\(([^)]+)\)", text):
        target = resolve_path(source, raw_target)
        if target is None:
            continue
        entry = manifest_entry(entries, target)
        assert entry is not None
        assert entry["status"] not in {"historical", "legacy"}
        assert entry["navigation"] != "legacy"
