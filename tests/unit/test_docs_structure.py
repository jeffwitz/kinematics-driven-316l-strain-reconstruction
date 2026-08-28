from scripts.check_docs_structure import load_manifest, pages


def test_manifest_covers_documentation_inventory() -> None:
    assert load_manifest()
    assert len(pages()) >= 140
