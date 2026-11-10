"""Exercise the relocated hospital components without model calls."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import pytest

HOSPITAL_ROOT = Path(__file__).resolve().parents[1] / "integrations/openhospital_runtime"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, HOSPITAL_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_case_data_rebuilds_the_patient_index():
    expected = json.loads((HOSPITAL_ROOT / "department_patient_ids.json").read_text(encoding="utf-8"))
    builder = load_script("hospital_department_builder", "build_openhospital_department_ids.py")
    assert builder.build(HOSPITAL_ROOT) == expected


@pytest.mark.asyncio
async def test_hospital_llm_pool_reuses_socket_without_sharing_credentials(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.syspath_prepend(str(HOSPITAL_ROOT / "hospital_service"))
    from hospital_service.service import llm_client
    requests = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.client_address, self.headers["Authorization"], payload))
            response = json.dumps({"choices": [{"message": {"content": "ok"}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    messages = [{"role": "user", "content": "unchanged clinical request"}]
    try:
        async with llm_client.llm_http_pool():
            shared = llm_client._shared_http_client[1]
            for key in ("first-test-key", "second-test-key"):
                caller = llm_client.OpenAICompatibleChatClient(
                    model_api=f"http://127.0.0.1:{server.server_port}/v1", api_key=key,
                    model="configured-model", timeout_seconds=3, retries=1)
                result = await caller.chat(messages=messages, temperature=0.2)
                assert result.content == "ok"
            assert not shared.is_closed
        assert shared.is_closed and llm_client._shared_http_client is None
        assert len(requests) == 2 and requests[0][0] == requests[1][0]
        assert [row[1] for row in requests] == ["Bearer first-test-key", "Bearer second-test-key"]
        assert all(row[2]["messages"] == messages for row in requests)
        assert requests[0][2] == requests[1][2]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", ("train", "primary_test", "advanced_test"))
async def test_bundled_service_lists_patients_and_preserves_its_world(tmp_path, dataset):
    build_app = load_script("hospital_local_service", "run_openhospital_local.py").build_app

    worlds = tmp_path / "worlds"
    app = build_app(dataset, world_dir=worlds, hospital_root=HOSPITAL_ROOT)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        health = (await client.get("/health")).json()
        assert health["dataset"] == dataset and health["world_id"]
        response = await client.get("/patients", params={"team_id": "migration-check", "patient_count": 1})
        assert response.status_code == 200
        assert len(response.json()["patients"]) == 1

    restarted = build_app(dataset, world_dir=worlds, hospital_root=HOSPITAL_ROOT)
    transport = httpx.ASGITransport(app=restarted, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        assert (await client.get("/health")).json()["world_id"] == health["world_id"]


def test_package_runs_without_the_outer_workspace(tmp_path):
    package = Path(__file__).resolve().parents[1]
    isolated_parent = tmp_path / "standalone"
    if os.name == "nt":
        isolated_parent = Path("\\\\?\\" + str(isolated_parent))
    ignored = shutil.ignore_patterns(".git", ".env*", ".venv", ".venv-*", ".mematlas", "__pycache__", "*.pyc", ".pytest_cache",
                                     ".vscode", "*.egg-info", "outputs", "dist", "build")
    shutil.copytree(package, isolated_parent / "mematlas", copy_function=shutil.copy2,
                    ignore=lambda directory, names: ignored(directory, names) |
                    ({"models"} if Path(directory) == package else set()))
    env = {key: value for key, value in os.environ.items() if key != "MEMATLAS_OPENHOSPITAL_ROOT"}
    env["PYTHONPATH"] = str(isolated_parent)
    env["PYTHONIOENCODING"] = "utf-8"
    independent = subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
         "from run_openhospital_local import build_app; "
         "app = build_app(world_dir=sys.argv[2]); assert 'mematlas' not in sys.modules",
         str(isolated_parent / "mematlas/integrations/openhospital_runtime"), str(tmp_path / "hospital-world")],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert independent.returncode == 0, independent.stderr
    doctor = subprocess.run(
        [sys.executable, "-c", "from mematlas.integrations.openhospital.application import load_doctor_runtime; "
         "agent, runtime = load_doctor_runtime(); assert agent.__name__ == 'MyDoctorAgent'"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert doctor.returncode == 0, doctor.stderr
    reports = []
    for condition in ("repeated", "novel_same_department", "novel_departments"):
        # Packaging/dry-run only: use an existing dependency-free v3 plugin.
        # Native Mem0 readiness is tested separately in its isolated interpreter.
        config = {"methods": [{"method": {"method_id": "memcon"},
                              "providers": {"llm": {"model": "preflight-model", "api_key_env": "OPENAI_API_KEY"},
                                            "embedding": {"model": "preflight-embedding", "api_key_env": "OPENAI_API_KEY"}}}],
                  "benchmark": {"kind": "openhospital", "condition": condition,
                                "departments": ["血液科", "呼吸内科"] if condition == "novel_departments" else ["血液科"],
                                "evaluation_departments": ["血液科", "呼吸内科"],
                                "cases_per_block": 2, "blocks": 2, "heldout_per_department": 1,
                                "advanced_per_department": 1},
                  "protocol": "experience_learning_curve", "output_dir": str(tmp_path / condition)}
        config_path = tmp_path / f"{condition}.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        env["OPENAI_API_KEY"] = "preflight-only-no-network"
        env["MODEL_API_KEY"] = "preflight-only-no-network"
        result = subprocess.run(
            [sys.executable, "-m", "mematlas.integrations.openhospital", "--config", str(config_path), "--dry-run"],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
        reports.append(json.loads(result.stdout))
    assert all(row["ready"] and row["train_exposure_count"] == 4 and row["checkpoints"] == [0, 2, 4]
               for row in reports)
