from scripts.check_docs_structure import (
    DOC_ROOT,
    declared_marker,
    load_coverage,
    load_manifest,
    manifest_entry,
    pages,
    reachable_from,
)


def test_manifest_covers_documentation_inventory() -> None:
    assert load_manifest()
    assert len(pages()) >= 140


def test_scientific_coverage_declares_core_quadrants() -> None:
    subjects = load_coverage()
    assert len(subjects) >= 10
    for subject in subjects:
        assert subject["name"]
        for quadrant in ("explanation", "reference", "how_to", "evidence"):
            assert subject[quadrant].endswith((".md", ".rst"))


def test_coverage_targets_use_current_manifest_entries() -> None:
    entries = load_manifest()
    subjects = load_coverage()
    for subject in subjects:
        for key in ("explanation", "reference", "how_to", "evidence"):
            entry = manifest_entry(entries, subject[key])
            assert entry is not None
            if subject["status"] == "complete":
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
