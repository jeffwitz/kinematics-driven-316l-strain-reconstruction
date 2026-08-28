from scripts.check_docs_structure import load_coverage, load_manifest, pages


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
