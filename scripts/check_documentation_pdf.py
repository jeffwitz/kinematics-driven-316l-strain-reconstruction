#!/usr/bin/env python3
"""Check the generated documentation PDF for publication-blocking failures."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "docs" / "_build" / "latex"
PDF = LATEX / "kinematics-driven-316l-strain-reconstruction.pdf"
TEX = LATEX / "kinematics-driven-316l-strain-reconstruction.tex"
LOG = LATEX / "kinematics-driven-316l-strain-reconstruction.log"


def main() -> None:
    if not PDF.is_file() or PDF.stat().st_size == 0:
        raise SystemExit("documentation PDF is missing or empty")
    text_path = Path("/tmp/spectral-documentation-pdf.txt")
    subprocess.run(["pdftotext", str(PDF), str(text_path)], check=True)
    text = text_path.read_text(errors="replace")
    raw_markers = (r"\begin{bmatrix}", r"\varepsilon", "```{math}")
    found = [marker for marker in raw_markers if marker in text]
    if found:
        raise SystemExit(f"raw mathematical markup found in PDF text: {found}")

    log = LOG.read_text(errors="replace")
    unresolved = re.findall(
        r"(?:undefined references|undefined citations|There were undefined)",
        log,
        flags=re.IGNORECASE,
    )
    if unresolved:
        raise SystemExit(f"unresolved PDF references: {unresolved}")

    missing = []
    for match in re.finditer(r"\\sphinxincludegraphics[^\n]*?\{+([^{}]+)\}+", TEX.read_text()):
        stem = match.group(1)
        candidates = [LATEX / stem]
        if "." not in Path(stem).name:
            candidates.extend(LATEX / f"{stem}{suffix}" for suffix in (".pdf", ".png", ".jpg"))
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(str(candidates[0]))
    if missing:
        raise SystemExit(f"missing PDF figures: {missing}")
    print(f"validated PDF: {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
