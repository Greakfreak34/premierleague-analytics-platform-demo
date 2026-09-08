"""Local artifact utilities only; no public social-data collector."""
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def immutable(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Immutable artifact conflict")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
