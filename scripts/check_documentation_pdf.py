#!/usr/bin/env python3
"""Check the generated documentation PDF for publication-blocking failures."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "docs" / "_build" / "latex"
PDF = LATEX / "kinematics-driven-316l-strain-reconstruction.pdf"
TEX = LATEX / "kinematics-driven-316l-strain-reconstruction.tex"
LOG = LATEX / "kinematics-driven-316l-strain-reconstruction.log"
REPORT = LATEX / "pdf_validation.json"


def main() -> None:
    required_files = (PDF, TEX, LOG)
    missing_required = [str(path) for path in required_files if not path.is_file()]
    if missing_required:
        raise SystemExit(f"documentation build files are missing: {missing_required}")
    if PDF.stat().st_size == 0:
        raise SystemExit("documentation PDF is empty")
    pdfinfo = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)$", pdfinfo, flags=re.MULTILINE)
    if page_match is None or int(page_match.group(1)) <= 0:
        raise SystemExit("documentation PDF has no positive page count")
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

    missing_characters = len(re.findall(r"Missing character", log))
    overfull_hboxes = len(re.findall(r"Overfull \\hbox", log))
    overfull_vboxes = len(re.findall(r"Overfull \\vbox", log))

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
    report = {
        "pdf": str(PDF.relative_to(ROOT)),
        "size_bytes": PDF.stat().st_size,
        "pages": int(page_match.group(1)),
        "missing_characters": missing_characters,
        "overfull_hboxes": overfull_hboxes,
        "overfull_vboxes": overfull_vboxes,
        "missing_figures": [],
        "unresolved_references": [],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"validated PDF: {PDF} ({PDF.stat().st_size} bytes, "
        f"{page_match.group(1)} pages); "
        f"missing_characters={missing_characters}, "
        f"overfull_hboxes={overfull_hboxes}, "
        f"overfull_vboxes={overfull_vboxes}"
    )


if __name__ == "__main__":
    main()
