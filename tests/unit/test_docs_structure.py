from scripts.check_docs_structure import (
    DOC_ROOT,
    declared_marker,
    load_coverage,
    load_manifest,
    manifest_entry,
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
