"""Audit and migrate paused Mem0 checkpoints for reviewed recovery patches.

Original contracts and snapshot manifests are backed up before any write. Memory
payloads, clinical records, model settings and schedules never change. The
exam-names profile also versions hospital parser contracts without changing data.
Run without --apply to validate and preview the exact migration.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from mematlas import ExperimentConfig
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import file_sha256, json_sha256
from mematlas.core.registry import default_registry
from mematlas.evaluation.artifacts import recovery_checkpoint
from mematlas.integrations.openhospital.config import HospitalOptions
from mematlas.integrations.openhospital.planning import build_plan
from mematlas.integrations.openhospital.rsi import _HospitalRunner

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "integrations/openhospital/rsi.py",
    "integrations/openhospital/suite.py",
    "integrations/openhospital_runtime/hospital-agent-sdk/hospital_agent_sdk/llm.py",
}
EXAM_GENERATOR = "integrations/openhospital_runtime/hospital_service/hospital_service/service/exam/generator.py"
EXAM_CLIENT = "integrations/openhospital_runtime/hospital-agent-sdk/hospital_agent_sdk/client.py"
PARALLEL_TEST_SOURCES = {
    "integrations/openhospital/execution.py", "integrations/openhospital/suite.py",
    "integrations/openhospital/monitor.py",
}
SDK_PATH_SOURCES = {"integrations/openhospital/execution.py"}


def table_digest(db):
    """Check the entire persisted clinical/accounting state without loading it."""
    import hashlib
    digest = hashlib.sha256()
    for table in ("examinations", "usage", "requests"):
        digest.update(table.encode())
        for row in db.execute(f'SELECT * FROM "{table}" ORDER BY id'):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def exam_world_updates(run):
    """Only update the reviewed parser hash; keep every existing database row."""
    import sqlite3
    updates = []
    for path in sorted((run / "world").glob("*/world.sqlite3")):
        with sqlite3.connect(path) as db:
            old = json.loads(db.execute("SELECT payload FROM metadata WHERE id='contract'").fetchone()[0])
            key = db.execute("SELECT payload FROM metadata WHERE id='world_key'").fetchone()[0]
            unchanged = table_digest(db)
        new = deepcopy(old)
        changes = []
        base = ROOT / "integrations/openhospital_runtime/hospital_service/hospital_service"
        for name, previous in old["files"].items():
            source = base / name.replace("\\", "/")
            current = file_sha256(source)
            if previous != current:
                if source.relative_to(ROOT).as_posix() != EXAM_GENERATOR:
                    raise ValueError(f"Unexpected hospital source change: {source}")
                new["files"][name] = current
                changes.append({"file": EXAM_GENERATOR, "before": previous, "after": current})
        for field, source in (("local_adapter", "run_openhospital_local.py"), ("world_adapter", "local_world.py")):
            if old[field] != file_sha256(ROOT / "integrations/openhospital_runtime" / source):
                raise ValueError(f"Unexpected hospital adapter change: {field}")
        if len(changes) != 1:
            raise ValueError("Expected exactly the reviewed examination parser patch")
        updates.append({"path": str(path), "split": path.parent.name, "before": old, "after": new,
            "world_id_before": json_sha256({"contract": old, "world_key": key}),
            "world_id_after": json_sha256({"contract": new, "world_key": key}),
            "unchanged_clinical_state": unchanged, "source_changes": changes})
    if {row["split"] for row in updates} != {"train", "primary_test", "advanced_test"}:
        raise ValueError("Expected all three existing hospital worlds")
    return updates


def updated_contract(old, *, patch="api-retry", worlds=()):
    new = deepcopy(old)
    changes = []
    allowed = {"api-retry": ALLOWED, "parallel-tests": PARALLEL_TEST_SOURCES,
               "sdk-paths": SDK_PATH_SOURCES, "exam-names": {EXAM_CLIENT}}[patch]
    for section, base in (("framework_sources", ROOT),
                          ("scene_sources", ROOT / "integrations/openhospital_runtime")):
        for name, previous in old[section].items():
            path = base / name.replace("\\", "/")
            current = file_sha256(path)
            if previous != current:
                relative = path.relative_to(ROOT).as_posix()
                if relative not in allowed:
                    raise ValueError(f"Non-operational source changed: {relative}")
                new[section][name] = current
                changes.append({"file": relative, "before": previous, "after": current})
    if patch == "exam-names":
        for world in worlds:
            if world["split"] not in old["services"]:
                continue
            service = new["services"][world["split"]]
            if service != {"contract": world["before"], "world_id": world["world_id_before"]}:
                raise ValueError("Original hospital contract does not match the persisted world")
            service.update(contract=world["after"], world_id=world["world_id_after"])
            changes.extend(world["source_changes"])
    expected = allowed | ({EXAM_GENERATOR} if patch == "exam-names" else set())
    if {row["file"] for row in changes} != expected:
        raise ValueError("Expected exactly the selected reviewed recovery patch")
    return new, changes


def signature(config, output, contract):
    options = HospitalOptions.from_experiment(config)
    prepared = build_plan(options, checkpoints=config.checkpoints, seed=config.seed or 0)
    frozen = json.loads((output / "frozen_cohort.json").read_text(encoding="utf-8"))
    method = config.methods[0]
    core = SimpleNamespace(config=method, method_id="mem0",
                           plugin=default_registry().get("mem0"))
    memory = SimpleNamespace(core_memory=core,
        department=options.departments[0] if len(options.departments) == 1 else "shared_doctor")
    engine = _HospitalRunner(prepared.plan, memory, output, None, None,
        execution_metadata={"resume_contract": contract, "protocol_fingerprint": json_sha256({
            "evaluation_contract": 2, "scene": json_sha256(frozen), "metrics": list(config.metrics)})})
    return engine.plan_signature()


def migrate(run, apply=False, patch="api-retry"):
    import psutil
    run = Path(run).resolve()
    manifest = json.loads((run / "suite.json").read_text(encoding="utf-8"))
    owner = manifest.get("owner", {})
    try:
        if psutil.Process(owner["pid"]).create_time() == owner["created"]:
            raise ValueError("Stop the suite at durable boundaries before migration")
    except (psutil.NoSuchProcess, KeyError):
        pass
    worlds = exam_world_updates(run) if patch == "exam-names" else []
    updates, receipts = [], []
    for condition, job in manifest["jobs"].items():
        if job["status"] == "completed":
            continue
        config = ExperimentConfig.load(job["config"])
        output = Path(job["output"])
        execution_path = output / "execution.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        contract, changes = updated_contract(execution["contract"], patch=patch, worlds=worlds)
        old_sig = signature(config, output, execution["contract"])
        new_sig = signature(config, output, contract)
        position, _, _ = recovery_checkpoint(output / "runs/001-mem0", expected_signature=old_sig)
        updates.append((execution_path, {**execution, "contract": contract}))
        memories = {}
        for path in (output / "runs/001-mem0/snapshots").glob("learned_*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            checksum = payload.pop("payload_sha256")
            if json_sha256(payload) != checksum or payload["plan_signature"] != old_sig:
                raise ValueError(f"Invalid original checkpoint: {path}")
            memory = path.parent / payload["memory_file"]
            if file_sha256(memory) != payload["memory_sha256"]:
                raise ValueError(f"Invalid original memory: {memory}")
            memories[str(memory)] = payload["memory_sha256"]
            payload["plan_signature"] = new_sig
            payload["payload_sha256"] = json_sha256(payload)
            updates.append((path, payload))
        receipts.append({"condition": condition, "position": position, "source_changes": changes,
            "contract_before": json_sha256(execution["contract"]), "contract_after": json_sha256(contract),
            "signature_before": old_sig, "signature_after": new_sig, "unchanged_memories": memories})
    report = {"reason": patch + "; resume from the same schedule positions", "world_updates": worlds,
              "created_at": datetime.now(timezone.utc).isoformat(), "conditions": receipts,
              "provider_sources": {name: file_sha256(ROOT / name) for name in ("providers/requests.py", "providers/llm.py")}}
    if apply:
        backup = run / "operational-migrations" / (datetime.now().strftime("%Y%m%d-%H%M%S-") + patch)
        backup.mkdir(parents=True, exist_ok=False)
        for path, payload in updates:
            target = backup / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        import sqlite3
        for world in worlds:
            path = Path(world["path"])
            target = backup / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as source, sqlite3.connect(target) as destination:
                source.backup(destination)
        report["files"] = [{"path": str(path), "before": file_sha256(path), "after_payload": json_sha256(value)}
                           for path, value in updates]
        atomic_write_json(backup / "receipt.json", report)
        for world in worlds:
            with sqlite3.connect(world["path"]) as db:
                if table_digest(db) != world["unchanged_clinical_state"]:
                    raise ValueError("Hospital data changed during migration")
                db.execute("UPDATE metadata SET payload=? WHERE id='contract'", (json.dumps(world["after"]),))
                if table_digest(db) != world["unchanged_clinical_state"]:
                    raise ValueError("Hospital data changed during migration")
        for path, payload in updates:
            atomic_write_json(path, payload)
        for receipt in receipts:
            for name, digest in receipt["unchanged_memories"].items():
                if file_sha256(name) != digest:
                    raise ValueError("Memory changed during migration")
        print(f"Backup and receipt: {backup}")
    print(json.dumps({"apply": apply, "conditions": [{k: v for k, v in row.items() if k != "unchanged_memories"}
                                                    for row in receipts]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--patch", choices=("api-retry", "exam-names", "parallel-tests", "sdk-paths"), default="api-retry")
    args = parser.parse_args()
    migrate(args.run_dir, args.apply, args.patch)
