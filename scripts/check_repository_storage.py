#!/usr/bin/env python3
"""Reject accidental large Git payloads and non-golden LFS additions."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys

DEFAULT_LIMIT = 20 * 1024 * 1024
LFS_WHITELIST = ("validation/golden/**",)


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=False
    )
    return result.stdout.decode().rstrip("\0\n")


def is_golden(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in LFS_WHITELIST)


def tracked_paths(base: str | None) -> list[str]:
    if base is None:
        raw = _git("diff", "--cached", "--name-only", "--diff-filter=AMRT", "-z")
    else:
        raw = _git("diff", "--name-only", "--diff-filter=AMRT", "-z", f"{base}...HEAD")
    return [item for item in raw.split("\0") if item]


def index_size(path: str) -> int:
    return int(_git("cat-file", "-s", f":{path}"))


def index_filter(path: str) -> str:
    result = subprocess.run(
        ("git", "check-attr", "--cached", "filter", "--", path),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rsplit(":", 1)[-1].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit; inspect only files added since it")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    paths = tracked_paths(args.base)
    violations: list[str] = []
    for path in paths:
        size = index_size(path)
        lfs = index_filter(path) == "lfs"
        if size > args.max_bytes and not is_golden(path) and not lfs:
            violations.append(
                f"large Git-normal file: {path} ({size / 1024**2:.1f} MiB); "
                "archive it externally or whitelist a golden reference"
            )
        if args.base is not None and lfs and not is_golden(path):
            violations.append(
                f"new non-golden LFS path: {path}; use validation/golden/** "
                "or keep the result outside the repository"
            )
    if violations:
        print("Repository storage policy violation:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in violations), file=sys.stderr)
        return 1
    print(f"Repository storage policy OK ({len(paths)} paths checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
