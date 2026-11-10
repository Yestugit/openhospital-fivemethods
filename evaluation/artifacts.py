"""Read results and verify a recovery checkpoint through a common public API."""
import json
from pathlib import Path

from mematlas.core.provenance import json_sha256


def read_results(path):
    from mematlas.core.evaluation import EvaluationResult
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        target = target / 'results.json'
    value = json.loads(target.read_text(encoding='utf-8-sig'))
    if not isinstance(value, list):
        raise ValueError('Expected results.json containing a list of method results')
    return [EvaluationResult.from_dict(row) for row in value]


def recovery_checkpoint(directory, *, expected_signature=None):
    source = Path(directory).expanduser().resolve()
    candidates = []
    for path in (source / "snapshots").glob("learned_*.json"):
        count = path.stem.removeprefix("learned_")
        if count.isdigit():
            candidates.append((int(count), path))
    payload = None
    # Full recovery payloads can be hundreds of MiB. Read only the newest
    # parseable candidate, never retain every historical payload in memory.
    for expected_position, path in sorted(candidates, reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count = value.get("schedule_position")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            if count != expected_position:
                raise ValueError("Recovery snapshot filename and position differ")
            position, payload = count, value
            break
    if payload is None:
        raise ValueError(f"No valid recovery snapshots found under {source}")
    if payload.get("schema_version") != 6:
        raise ValueError("Unsupported recovery checkpoint schema; expected version 6")
    checked = dict(payload)
    digest = checked.pop("payload_sha256", None)
    if not checked.get("committed") or json_sha256(checked) != digest:
        raise ValueError("Resume snapshot is incomplete or has an invalid checksum")
    if expected_signature is not None and payload.get("plan_signature") != expected_signature:
        raise ValueError("Resume snapshot does not match the current method, providers, plan or execution contract")
    return position, path, payload
