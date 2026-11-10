"""Durable local hospital world and reported service usage (no credentials)."""
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
from uuid import uuid4
import hashlib
from collections.abc import Mapping


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def redact_sensitive(value, *, key=""):
    compact = "".join(character for character in key.lower() if character.isalnum())
    suffixes = ("apikey", "accesstoken", "authtoken", "bearertoken", "clientsecret", "password", "secret", "token")
    if not compact.endswith("env") and compact.endswith(suffixes) and value is not None:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child_key): redact_sensitive(child_value, key=str(child_key))
                for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value


class HospitalWorld:
    def __init__(self, directory, contract, record_type):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract, self.record_type = contract, record_type
        self.path = self.directory / "world.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            previous = db.execute("SELECT payload FROM metadata WHERE id='contract'").fetchone()
            if previous and json.loads(previous[0]) != contract:
                raise ValueError("Hospital world configuration changed; select a new world directory")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (json.dumps(contract),))
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('world_key', ?)", (uuid4().hex,))
            world_key = db.execute("SELECT payload FROM metadata WHERE id='world_key'").fetchone()[0]
            self.world_id = json_sha256({"contract":contract, "world_key":world_key})
            db.execute("CREATE TABLE IF NOT EXISTS examinations (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS usage (id TEXT PRIMARY KEY, team TEXT, model TEXT, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS response_cache (id TEXT PRIMARY KEY, signature TEXT NOT NULL, response TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, team TEXT, path TEXT, payload TEXT)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, dataset_key, patient_id, examination_name):
        key = json_sha256((dataset_key, patient_id, examination_name))
        with self.connect() as db:
            row = db.execute("SELECT payload FROM examinations WHERE id=?", (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["abnormal_indicators"] = tuple(data["abnormal_indicators"])
        return self.record_type(**data)

    def list_patient(self, dataset_key, patient_id):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM examinations").fetchall()
        data = [json.loads(row[0]) for row in rows]
        return {row["examination_name"]: self.get(dataset_key, patient_id, row["examination_name"])
                for row in data if row["dataset_key"] == dataset_key and row["patient_id"] == patient_id}

    def put_if_absent(self, record):
        key = json_sha256((record.dataset_key, record.patient_id, record.examination_name))
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO examinations VALUES (?,?)", (key, json.dumps(asdict(record), ensure_ascii=False)))
        return self.get(record.dataset_key, record.patient_id, record.examination_name)

    def record_usage(self, team, model, usage, *, known):
        payload = {"input_tokens": int(usage.get("prompt_tokens") or 0),
                   "output_tokens": int(usage.get("completion_tokens") or 0), "known": known}
        with self.connect() as db:
            db.execute("INSERT INTO usage VALUES (?,?,?,?)", (uuid4().hex, team, model, json.dumps(payload)))

    def usage(self, team):
        with self.connect() as db:
            rows = db.execute("SELECT model, payload FROM usage WHERE team=?", (team,)).fetchall()
        data = [json.loads(payload) for _, payload in rows]
        inputs, outputs = sum(r["input_tokens"] for r in data), sum(r["output_tokens"] for r in data)
        return {"requests": len(rows), "input_tokens": inputs, "output_tokens": outputs,
                "total_tokens": inputs + outputs, "unknown_calls": sum(not r["known"] for r in data)}

    def record_request(self, team, path, payload, response):
        safe = redact_sensitive({"request": payload, "response": response})
        with self.connect() as db:
            db.execute("INSERT INTO requests VALUES (?,?,?,?)", (uuid4().hex, team, path, json.dumps(safe, ensure_ascii=False)))

    def cached_response(self, identity, payload):
        signature = json_sha256(redact_sensitive(payload))
        with self.connect() as db:
            row = db.execute("SELECT signature,response FROM response_cache WHERE id=?", (identity,)).fetchone()
        if row is None:
            return None
        if row[0] != signature:
            raise ValueError("Business request ID reused with different payload")
        return json.loads(row[1])

    def cache_response(self, identity, payload, response):
        signature = json_sha256(redact_sensitive(payload))
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO response_cache VALUES (?,?,?)",
                (identity, signature, json.dumps(response, ensure_ascii=False)))
        self.cached_response(identity, payload)

    def close(self):
        pass
