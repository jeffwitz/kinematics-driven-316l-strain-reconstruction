import hashlib
import json
from pathlib import Path


def test_article_source_matches_versioned_manifest() -> None:
    repository = Path(__file__).resolve().parents[2]
    manifest_path = repository / "ArticleSource" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    article = manifest["article"]
    article_path = manifest_path.parent / article["file"]

    assert article_path.stat().st_size == article["size_bytes"]
    assert hashlib.sha256(article_path.read_bytes()).hexdigest() == article["sha256"]
    assert article["doi"] is None
