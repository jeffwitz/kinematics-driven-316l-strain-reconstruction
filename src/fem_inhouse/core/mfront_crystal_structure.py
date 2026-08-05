"""Structural fingerprints for the compile-time FCC declarations in MFront."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CrystalStructureFingerprint:
    crystal_structure: str
    sliding_system: str
    interaction_matrix: tuple[float, ...]
    source_sha256: str

    def structure_contract_sha256(self) -> str:
        payload = (
            self.crystal_structure,
            self.sliding_system,
            self.interaction_matrix,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


def _one(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} declaration, found {len(matches)}")
    return matches[0]


def read_crystal_structure_fingerprint(source_path: Path) -> CrystalStructureFingerprint:
    text = source_path.read_text(encoding="utf-8")
    crystal = _one(r"@CrystalStructure\s+([A-Za-z0-9_]+)\s*;", text, "CrystalStructure")
    sliding_match = _one(
        r"@SlidingSystem\s*<([^>]+)>\s*\{([^}]+)\}\s*;",
        text,
        "SlidingSystem",
    )
    sliding = (
        "<"
        + ",".join(part.strip() for part in sliding_match[0].split(","))
        + ">{"
        + ",".join(part.strip() for part in sliding_match[1].split(","))
        + "}"
    )
    values = _one(r"@InteractionMatrix\s*\{([^}]+)\}\s*;", text, "InteractionMatrix")
    coefficients = tuple(float(value.strip()) for value in values.split(","))
    if len(coefficients) != 7 or not all(math.isfinite(value) for value in coefficients):
        raise ValueError("InteractionMatrix must contain seven finite coefficients")
    return CrystalStructureFingerprint(
        crystal_structure=crystal,
        sliding_system=sliding,
        interaction_matrix=coefficients,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
