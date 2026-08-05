"""Machine-local pyFFTW wisdom persistence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any


def wisdom_file(directory: Path, metadata: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return directory / f"wisdom-{key}.json"


def load_wisdom(directory: Path, metadata: dict[str, Any], pyfftw: Any) -> bool:
    path = wisdom_file(directory, metadata)
    try:
        payload = json.loads(path.read_text())
        if payload.get("metadata") != metadata:
            return False
        wisdom = pickle.loads(base64.b64decode(payload["wisdom"]))
        return bool(pyfftw.import_wisdom(wisdom))
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        binascii.Error,
        pickle.PickleError,
    ):
        return False


def save_wisdom(directory: Path, metadata: dict[str, Any], pyfftw: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = wisdom_file(directory, metadata)
    payload = {
        "metadata": metadata,
        "wisdom": base64.b64encode(pickle.dumps(pyfftw.export_wisdom())).decode("ascii"),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path
