"""Dashboard accounting and liveness checks without models or hospital services."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import psutil

from mematlas.core.telemetry import UsageLedger, UsageRecord
from mematlas.integrations.openhospital.monitor import background_monitor, status_data
from mematlas.integrations.openhospital.progress import HospitalProgress


def test_monitor_waits_for_launcher_without_creating_experiment_directory(tmp_path):
    root = tmp_path / "not-yet-launched"
    with background_monitor(root, port=0) as url:
        with urlopen(url + "/status") as response:
            data = json.loads(response.read())
            assert response.headers["Cache-Control"] == "no-store"
        assert data["status"] == "starting" and data["completed"] == 0
        assert data["planned_encounters"] == 1100 and not root.exists()
        root.mkdir()
        (root / "suite.json").write_text(json.dumps({"status": "completed", "pilot": True, "jobs": {}}))
        with urlopen(url + "/status") as response:
            assert json.loads(response.read())["planned_encounters"] == 22
        with urlopen(url) as response:
            assert b"Completed frozen checkpoints" in response.read()


def test_monitor_distinguishes_stale_heartbeat_and_finished_run(tmp_path):
    now = datetime.now(timezone.utc)
    progress = {"training_completed": 26, "test_completed": 50, "updated_at": (now - timedelta(seconds=60)).isoformat(),
        "usage": {"llm_calls": 8, "embedding_calls": 3, "known_total_tokens": 100, "failed_calls": 1,
                  "untracked_calls": 1, "token_coverage": "partial"}}
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    owner = psutil.Process()
    manifest = {"status": "running", "owner": {"pid": owner.pid, "created": owner.create_time()},
        "preflight": {"conditions": {"novel_departments": {"train_exposure_count": 250,
             "primary_test_count": 50, "checkpoints": [0, 50, 100, 150, 200, 250], "advanced_test_count": 50}}},
        "jobs": {"novel_departments": {"output": str(tmp_path), "status": "running", "attempt": 2,
             "pid": owner.pid, "created": owner.create_time()}}}
    data = status_data(manifest)
    assert data["owner_alive"] and data["completed"] == 76
    job = data["jobs"][0]
    assert job["stale"] and job["worker_alive"]
    assert job["training_total"] + job["test_total"] == 600
    assert data["usage"]["untracked_calls"] == 1 and data["usage"]["token_coverage"] != "complete"
    manifest.pop("owner")
    assert status_data(manifest)["status"] == "launcher offline"
    manifest.update(status="completed")
    manifest["jobs"]["novel_departments"]["status"] = "completed"
    assert not status_data(manifest)["jobs"][0]["stale"]


def test_progress_keeps_usage_recovery_and_checkpoint_metrics_across_heartbeats(tmp_path):
    ledger = UsageLedger()
    ledger.add(UsageRecord("test", "test", prompt_tokens=10, completion_tokens=2, phase="doctor"))
    ledger.add(UsageRecord("test", "test", usage_known=False, status="failed", phase="doctor"))
    ledger.add(UsageRecord("test", "embed", prompt_tokens=7, phase="memory", resource_kind="embedding"))
    plan = SimpleNamespace(training_count=20, primary_test_ids=("a", "b"), advanced_test_ids=(), effective_checkpoints=lambda: (20,))
    progress = HospitalProgress(tmp_path, plan)
    progress.bridge = SimpleNamespace(core_memory=SimpleNamespace(method_id="mem0", services=SimpleNamespace(usage=ledger),
        _method=SimpleNamespace(memory=SimpleNamespace(memories=[1, 2, 3]))), history=SimpleNamespace(counts=lambda: {"learned_exposures": 2}))
    progress.training = {str(i): {"position": i, "response_time_s": i * 10, "json_recovery": {"repairs": 1, "fallbacks": 0}} for i in reversed(range(1, 7))}
    progress.tests = {"test": {"json_recovery": {"repairs": 3, "fallbacks": 1}}}
    progress.record_checkpoint(100, "primary_test", {"diagnosis_accuracy": .5, "evaluated_cases": 2, "case_reports": ["omitted"]})
    progress.record_checkpoint(50, "primary_test", {"diagnosis_accuracy": .1})
    assert progress.last_test["checkpoint"] == 100
    progress.write(status="running", phase="train", patient_id="a", position=7)
    first = json.loads((tmp_path / "progress.json").read_text())
    progress.write()
    second = json.loads((tmp_path / "progress.json").read_text())
    assert first["case_started_at"] == second["case_started_at"]
    assert second["case_elapsed_s"] >= first["case_elapsed_s"]
    assert second["json_recovery"] == {"repairs": 9, "fallbacks": 1}
    assert second["timing"]["mean_last_5_training_s"] == 40
    assert second["usage_breakdown"]["Doctor"]["tracked_calls"] == 2
    assert second["usage_breakdown"]["Doctor"]["untracked_calls"] == 1
    assert second["usage_breakdown"]["Embedding"]["prompt_tokens"] == 7
    assert second["memory"]["memories"] == 3 and len(second["test_checkpoints"]) == 2
    assert "case_reports" not in second["test_checkpoints"][0]
