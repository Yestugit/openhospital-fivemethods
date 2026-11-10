"""The parser migration preserves hospital rows and refuses unrelated changes."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def migration(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/openhospital-retry-migration.py"
    spec = importlib.util.spec_from_file_location("exam_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "source")
    root = module.ROOT
    for relative in (module.EXAM_GENERATOR, "integrations/openhospital_runtime/run_openhospital_local.py",
                     "integrations/openhospital_runtime/local_world.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("reviewed source", encoding="utf-8")
    return module


def test_world_preview_does_not_mutate_database_and_keeps_all_records(migration, tmp_path):
    module = migration
    run = tmp_path / "run"
    for split in ("train", "primary_test", "advanced_test"):
        path = run / "world" / split / "world.sqlite3"
        path.parent.mkdir(parents=True)
        contract = {"local_adapter": module.file_sha256(module.ROOT / "integrations/openhospital_runtime/run_openhospital_local.py"),
                    "world_adapter": module.file_sha256(module.ROOT / "integrations/openhospital_runtime/local_world.py"),
                    "files": {"service\\exam\\generator.py": "old-reviewed-parser"}}
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE metadata (id TEXT PRIMARY KEY, payload TEXT)")
            db.executemany("INSERT INTO metadata VALUES (?, ?)", [("contract", json.dumps(contract)), ("world_key", split)])
            for table in ("examinations", "usage", "requests"):
                db.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, payload TEXT)")
                db.execute(f"INSERT INTO {table} VALUES (?, ?)", ("existing", "real clinical evidence"))
    before = {path: module.file_sha256(path) for path in (run / "world").glob("*/world.sqlite3")}
    updates = module.exam_world_updates(run)
    assert len(updates) == 3
    assert all(module.file_sha256(path) == digest for path, digest in before.items())
    for update in updates:
        assert update["world_id_before"] != update["world_id_after"]
        with sqlite3.connect(update["path"]) as db:
            assert module.table_digest(db) == update["unchanged_clinical_state"]
    # An adapter change must not be disguised as the reviewed name-format repair.
    (module.ROOT / "integrations/openhospital_runtime/local_world.py").write_text("unreviewed change")
    with pytest.raises(ValueError, match="Unexpected hospital adapter"):
        module.exam_world_updates(run)
