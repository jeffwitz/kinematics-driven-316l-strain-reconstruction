"""Refresh the committed intersphinx inventories used as offline fallbacks.

`docs/conf.py` lists each project's remote inventory first and a copy under
`docs/_inventories/` second. Intersphinx only warns when every location for a
project fails, so the copy is what keeps a strict `-W` documentation build green
during an upstream outage. It is a safety net, not the source of truth: the
remote is still consulted first, so references resolve against current upstream
documentation.

Run this when the copies drift, and commit the result.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

#: Same projects as `intersphinx_mapping` in `docs/conf.py`.
INVENTORIES = {
    "numpy": "https://numpy.org/doc/stable/objects.inv",
    "python": "https://docs.python.org/3/objects.inv",
}

DESTINATION = Path(__file__).resolve().parent.parent / "docs" / "_inventories"
TIMEOUT_SECONDS = 60


def refresh(destination: Path = DESTINATION, *, timeout: int = TIMEOUT_SECONDS) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in INVENTORIES.items():
        target = destination / f"{name}.inv"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            # Leave the existing copy in place: a stale fallback is better than
            # none, and this script must not empty the net it maintains.
            print(f"{name}: FAILED to fetch {url} ({error}); keeping the existing copy")
            failures += 1
            continue
        if not payload.startswith(b"# Sphinx inventory version"):
            print(f"{name}: refused, {url} did not return a Sphinx inventory")
            failures += 1
            continue
        target.write_bytes(payload)
        print(f"{name}: {len(payload)} bytes from {url}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    return 1 if refresh(args.destination) else 0


if __name__ == "__main__":
    raise SystemExit(main())
