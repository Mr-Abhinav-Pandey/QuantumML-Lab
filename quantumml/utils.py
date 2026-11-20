"""Shared utility functions for QuantumML Lab."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    """Return the project root directory (parent of ``quantumml/``)."""
    return Path(__file__).resolve().parent.parent


def research_dir() -> Path:
    return project_root() / "research"


def reference_library_dir() -> Path:
    return research_dir() / "reference_library"


def original_experiments_dir() -> Path:
    return research_dir() / "original_experiments"


def benchmarking_dir() -> Path:
    return research_dir() / "benchmarking"


def generate_experiment_id() -> str:
    """Generate a unique experiment identifier.

    Format: ``EXP-YYYYMMDDTHHMMSSz-XXXXXX``
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"EXP-{ts}-{suffix}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: object) -> int:
    """Derive a reproducible 32-bit seed from structured input parts."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def save_json(data: dict, path: Path) -> None:
    """Atomically write JSON to *path*, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
