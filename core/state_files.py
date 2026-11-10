"""Content-addressed JSON state transport for large native checkpoints.

Files are immutable, checksum verified and live outside the backend database.
Keep referenced snapshot files together with an exported checkpoint.
"""
from hashlib import sha256
import json
from pathlib import Path

from mematlas.core.files import atomic_write_json

KEY = "__mematlas_state_file_v1__"
INLINE_LIMIT = 1024 * 1024


def pack_state(state, directory, *, limit=INLINE_LIMIT):
    if isinstance(state, dict) and set(state) == {KEY}:
        return state
    raw = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    if len(raw) <= limit:
        return state
    checksum = sha256(raw).hexdigest()
    target = Path(directory).resolve() / "state-snapshots" / (checksum + ".json")
    if not target.exists():
        atomic_write_json(target, state)
    # atomic_write_json uses different whitespace: hash logical canonical bytes,
    # not filesystem encoding details; size records the same canonical payload.
    return {KEY: {"path": str(target), "sha256": checksum, "canonical_bytes": len(raw)}}


def unpack_state(state):
    if not isinstance(state, dict) or set(state) != {KEY}:
        return state
    reference = state[KEY]
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "canonical_bytes"}:
        raise ValueError("Invalid state snapshot reference")
    checksum = reference["sha256"]
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)):
        raise ValueError("Invalid state snapshot SHA-256")
    value = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    if sha256(raw).hexdigest() != checksum or len(raw) != reference["canonical_bytes"]:
        raise ValueError("State snapshot checksum/size mismatch")
    return value


def fingerprint_state(state, *, resolve_files=True):
    """Hash logical state independently of immutable snapshot file locations.

    Source files are still checksum checked. JSON blobs may themselves contain
    snapshot references; binary databases/vectors are represented by their hash.
    No pickle/database is opened and no provider is constructed.
    """
    def normalize(value):
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {KEY}:
            if not resolve_files:
                return {KEY: {key: item for key, item in value[KEY].items() if key != "path"}}
            return normalize(unpack_state(value))
        if set(value) == {"path", "sha256", "size"}:
            if not resolve_files:
                return {"sha256": value["sha256"], "size": value["size"]}
            raw = Path(value["path"]).read_bytes()
            if len(raw) != value["size"] or sha256(raw).hexdigest() != value["sha256"]:
                raise ValueError("State blob checksum/size mismatch")
            try:
                content = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"sha256": value["sha256"], "size": value["size"]}
            return {"json_blob": normalize(content)}
        # A compressed native snapshot already carries a verified logical hash
        # and a separate archive checksum. Its location is transport metadata.
        if {"snapshot_file", "file_sha256", "sha256", "size_bytes"}.issubset(value):
            if resolve_files:
                raw = Path(value["snapshot_file"]).read_bytes()
                if len(raw) != value["size_bytes"] or sha256(raw).hexdigest() != value["file_sha256"]:
                    raise ValueError("Native snapshot checksum/size mismatch")
            value = {key: item for key, item in value.items() if key != "snapshot_file"}
        return {key: normalize(item) for key, item in value.items()}

    raw = json.dumps(normalize(state), sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()
